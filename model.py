"""
DMS2F-HAD: Dual-branch Mamba-based Spatial-Spectral Fusion Network
for Hyperspectral Anomaly Detection (Pant et al., WACV 2026).

A compact, pure-PyTorch re-implementation (no `mamba_ssm` CUDA kernels needed,
so it runs on Windows / CPU / any GPU).

Pipeline (paper Fig. 2):
    patch x (B, C, h, w)
      -> random spatial masking (training only)
      -> In-head: 1x1 conv + BN + GELU               -> F  (B, c1, h, w)
      -> Spatial branch : 3x3 & 5x5 convs -> pixel tokens -> Mamba  -> F_spa
      -> Spectral branch: 1-D spectral convs -> band grouping -> Mamba -> F_spe
      -> Gated fusion   : G = sigmoid(conv1x1([F_spa, F_spe]))
                          F_fus = Proj(G * F_spa + (1 - G) * F_spe)
      -> 2 x SS-Decoder block (Mamba || conv3 || conv5 -> concat -> conv1x1)
      -> Out-head 1x1 conv  -> reconstructed patch (B, C, h, w)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Mamba (selective state-space model) in plain PyTorch
# ---------------------------------------------------------------------------
def selective_scan(u, delta, A, B, C, D, chunk=16):
    """Mamba S6 recurrence (exact), computed with a two-level chunked scan.

    u, delta : (b, l, d)   input and step size
    A        : (d, n)      state matrix (diagonal, negative)
    B, C     : (b, l, n)   input-dependent projections
    D        : (d,)        skip connection
    h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t ;  y_t = <h_t, C_t> + D * u_t

    Instead of l sequential steps we (1) scan all chunks of length T in parallel from a
    zero state, (2) pass the state from chunk to chunk, (3) add the carried state back.
    That is ~2*sqrt(l) Python steps instead of l (16+16 instead of 256 for a 16x16 patch).
    """
    b, l, d = u.shape
    n = A.shape[1]
    dA = torch.exp(delta.unsqueeze(-1) * A)                      # (b, l, d, n) decay
    dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b, l, d, n) input
    T = min(chunk, l)
    pad = (-l) % T
    if pad:                                                      # identity steps at the end
        dA = F.pad(dA, (0, 0, 0, 0, 0, pad), value=1.0)
        dBu = F.pad(dBu, (0, 0, 0, 0, 0, pad))
    nc = (l + pad) // T
    dA, dBu = dA.view(b, nc, T, d, n), dBu.view(b, nc, T, d, n)

    h = u.new_zeros(b, nc, d, n)                                 # (1) local scans
    hs = []
    for t in range(T):
        h = torch.addcmul(dBu[:, :, t], dA[:, :, t], h)
        hs.append(h)
    hs = torch.stack(hs, dim=2)                                  # (b, nc, T, d, n)
    decay = torch.cumprod(dA, dim=2)                             # decay since chunk start

    carry, carries = u.new_zeros(b, d, n), []                    # (2) chunk-to-chunk state
    for c in range(nc):
        carries.append(carry)
        carry = torch.addcmul(hs[:, c, -1], decay[:, c, -1], carry)
    hs = hs + decay * torch.stack(carries, dim=1).unsqueeze(2)   # (3) add carried state

    hs = hs.reshape(b, nc * T, d, n)[:, :l]
    y = torch.einsum("bldn,bln->bld", hs, C)
    return y + u * D


class Mamba(nn.Module):
    """Standard Mamba block (Gu & Dao, 2023): in_proj -> conv1d -> S6 -> gate -> out_proj."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # dt initialisation so that softplus(bias) lies in [1e-3, 1e-1] (as in official Mamba)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):                                   # x: (b, l, d_model)
        l = x.shape[1]
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :l].transpose(1, 2))
        dt, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        y = selective_scan(x, dt, -torch.exp(self.A_log), B, C, self.D)
        return self.out_proj(y * F.silu(z))


class MambaLayer(nn.Module):
    """Pre-norm residual Mamba: x + Mamba(LayerNorm(x)), stacked `depth` times."""

    def __init__(self, dim, depth=1, d_state=16):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.mambas = nn.ModuleList([Mamba(dim, d_state=d_state) for _ in range(depth)])

    def forward(self, x):
        for norm, mamba in zip(self.norms, self.mambas):
            x = x + mamba(norm(x))
        return x


def to_tokens(x):            # (B, c, h, w) -> (B, h*w, c)   row-major pixel tokens
    return x.flatten(2).transpose(1, 2)


def from_tokens(t, h, w):    # (B, h*w, c) -> (B, c, h, w)
    return t.transpose(1, 2).reshape(t.shape[0], -1, h, w)


# ---------------------------------------------------------------------------
# 2. Building blocks of DMS2F-HAD
# ---------------------------------------------------------------------------
def conv_bn_gelu(cin, cout, k):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2), nn.BatchNorm2d(cout), nn.GELU())


def random_spatial_mask(x, p=0.5, min_size=2, max_size=5):
    """Paper Sec. 3.1.1: with probability p, zero a random rectangle across ALL bands
    of each training patch (size and position randomised per sample)."""
    B, _, H, W = x.shape
    mask = torch.ones(B, 1, H, W)                       # built on CPU, then moved once
    for b in range(B):
        if torch.rand(1).item() < p:
            mh, mw = torch.randint(min_size, max_size + 1, (2,)).tolist()
            i = torch.randint(0, H - mh + 1, (1,)).item()
            j = torch.randint(0, W - mw + 1, (1,)).item()
            mask[b, :, i:i + mh, j:j + mw] = 0
    return x * mask.to(x.device)


class SpatialBranch(nn.Module):
    """Multi-scale spatial feature extraction (parallel 3x3, 5x5) + Mamba over h*w pixel tokens."""

    def __init__(self, dim, d_state):
        super().__init__()
        self.conv3 = conv_bn_gelu(dim, dim, 3)
        self.conv5 = conv_bn_gelu(dim, dim, 5)
        self.mamba = MambaLayer(dim, depth=1, d_state=d_state)

    def forward(self, f):
        h, w = f.shape[-2:]
        f = self.conv3(f) + self.conv5(f)
        return from_tokens(self.mamba(to_tokens(f)), h, w)


class SpectralBranch(nn.Module):
    """Multi-scale 1-D spectral convs -> pool patch -> split c1 into g overlapping groups
    of c2 bands (stride k) -> Mamba over the g groups -> project back to c1."""

    def __init__(self, dim, group_size=16, group_stride=8, d_state=16):
        super().__init__()
        self.spec_conv7 = nn.Conv1d(1, 1, 7, padding=3)
        self.spec_conv3 = nn.Conv1d(1, 1, 3, padding=1)
        self.norm7, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.starts = list(range(0, dim - group_size + 1, group_stride))
        self.group_size = group_size
        self.mamba = MambaLayer(group_size, depth=1, d_state=d_state)
        self.proj = nn.Sequential(nn.Linear(len(self.starts) * group_size, dim), nn.LayerNorm(dim), nn.GELU())
        self.drop = nn.Dropout(0.1)

    def forward(self, f):
        B, c, h, w = f.shape
        s = f.permute(0, 2, 3, 1).reshape(-1, 1, c)                       # every pixel's spectrum
        s = s + F.gelu(self.norm7(self.spec_conv7(s))) + F.gelu(self.norm3(self.spec_conv3(s)))
        s = self.drop(s).reshape(B, h * w, c).mean(1)                      # pool patch -> (B, c1)
        groups = torch.stack([s[:, i:i + self.group_size] for i in self.starts], 1)  # (B, g, c2)
        s = self.proj(self.mamba(groups).flatten(1))                       # (B, c1)
        return s[:, :, None, None].expand(B, c, h, w)


class SSDecoderBlock(nn.Module):
    """Spatial-Spectral decoder block (paper Fig. 2c): global Mamba path + local 3x3 / 5x5 convs."""

    def __init__(self, dim, d_state, depth=2):
        super().__init__()
        self.mamba = MambaLayer(dim, depth=depth, d_state=d_state)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=2)
        self.fuse = nn.Conv2d(3 * dim, dim, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        g = from_tokens(self.mamba(to_tokens(x)), h, w)
        return F.gelu(self.fuse(torch.cat([g, self.conv3(x), self.conv5(x)], 1))) + x


# ---------------------------------------------------------------------------
# 3. Full model
# ---------------------------------------------------------------------------
class DMS2FHAD(nn.Module):
    """fusion: 'gated' (paper), 'add' (addition fusion), 'spatial' or 'spectral' (single branch).
    The last three are used for the ablation study (paper Table 5)."""

    def __init__(self, in_bands, dim=64, group_size=16, group_stride=8, d_state=16, fusion="gated"):
        super().__init__()
        self.fusion = fusion
        self.in_head = nn.Sequential(nn.Conv2d(in_bands, dim, 1), nn.BatchNorm2d(dim), nn.GELU())
        self.spatial = SpatialBranch(dim, d_state)
        self.spectral = SpectralBranch(dim, group_size, group_stride, d_state)
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, 1, 1), nn.Sigmoid())      # one weight per pixel
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.decoder = nn.Sequential(SSDecoderBlock(dim, d_state), SSDecoderBlock(dim, d_state))
        self.out_head = nn.Conv2d(dim, in_bands, 1)

    def forward(self, x):
        if self.training:
            x = random_spatial_mask(x)
        f = self.in_head(x)
        if self.fusion == "spatial":
            fused = self.spatial(f)
        elif self.fusion == "spectral":
            fused = self.spectral(f)
        else:
            f_spa, f_spe = self.spatial(f), self.spectral(f)
            if self.fusion == "add":
                fused = f_spa + f_spe
            else:  # Eq. (1): F_fusion = Proj(G * F_spa + (1 - G) * F_spe)
                g = self.gate(torch.cat([f_spa, f_spe], 1))
                fused = g * f_spa + (1 - g) * f_spe
        return self.out_head(self.decoder(self.proj(fused)))


if __name__ == "__main__":
    m = DMS2FHAD(in_bands=189)
    print("parameters: %.3f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
    print(m(torch.randn(2, 189, 16, 16)).shape)
