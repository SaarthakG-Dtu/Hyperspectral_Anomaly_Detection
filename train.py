"""
Train DMS2F-HAD on one hyperspectral image (unsupervised) and detect anomalies.

    python train.py --dataset aviris_1                 # full model (gated fusion)
    python train.py --dataset aviris_1 --fusion spatial  # ablation variants: spatial | spectral | add
    python train.py --dataset all --fusion all          # everything (sequential)

Rule: follow the paper where it is explicit, and the official code where the paper is silent.
Paper (Sec. 4.3): Adam, lr 5e-4, weight decay 1e-4, 100 epochs, 16x16 patches, stride 8, c1 = 64, c2 = 16, k = 8.
Official code: loss = MSE + 0.1 * L1, batch 32, Mamba state size 64, AUC checked every epoch, zero padding.
Anomaly score = L2 norm of the reconstruction residual of every pixel (Eq. 2).
Results go to results/runs/<dataset>_<fusion>.npz
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import DMS2FHAD
from utils import DATASETS, auc_score, extract_patches, fold_patches, load_hsi

RUN_DIR = Path(__file__).parent / "results" / "runs"


@torch.no_grad()
def detect(model, patches, H, W, cube_t, batch=64):
    """Reconstruct every patch, stitch the image back together and return the residual map R."""
    model.eval()
    recon = torch.cat([model(patches[i:i + batch]) for i in range(0, len(patches), batch)])
    recon = fold_patches(recon.cpu(), H, W)                          # (C, H, W)
    return torch.linalg.norm(cube_t - recon, dim=0).numpy()          # Eq. (2)


# Settings of the official code (github.com/Ayushma00/DMS2F-HAD), where they differ from the paper text
OFFICIAL = dict(d_state=64, conv5=False, group_size=12, group_stride=6, epochs=150, eval_every=1)


def train(dataset, fusion="gated", epochs=100, batch=32, lr=5e-4, wd=1e-4, eval_every=1, seed=0,
          d_state=64, conv5=True, group_size=16, group_stride=8, tag=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cube, gt = load_hsi(dataset)
    H, W, C = cube.shape
    cube_t = torch.from_numpy(cube).permute(2, 0, 1)
    patches = extract_patches(cube).to(dev)
    model = DMS2FHAD(C, fusion=fusion, d_state=d_state, conv5=conv5, group_size=group_size,
                     group_stride=group_stride).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    print(f"[{dataset} | {fusion}] image {H}x{W}x{C}, {len(patches)} patches, {n_params/1e6:.3f}M params", flush=True)

    losses, auc_hist = [], []
    best = {"auc": -1}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm, ep_loss = torch.randperm(len(patches), device=dev), 0.0
        for i in range(0, len(patches), batch):
            x = patches[perm[i:i + batch]]
            rec = model(x)
            loss = F.mse_loss(rec, x) + 0.1 * F.l1_loss(rec, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(x)
        losses.append(ep_loss / len(patches))

        if ep % eval_every == 0 or ep == epochs:
            score = detect(model, patches, H, W, cube_t)
            auc = auc_score(score, gt)
            auc_hist.append((ep, auc))
            if auc > best["auc"]:
                best = {"auc": auc, "epoch": ep, "map": score}
            print(f"  epoch {ep:3d}  loss {losses[-1]:.5f}  AUC {auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # inference time on the whole image (paper Table 3)
    if dev == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    final_map = detect(model, patches, H, W, cube_t)
    if dev == "cuda":
        torch.cuda.synchronize()
    infer_time = time.time() - t1

    out_dir = RUN_DIR / "extra" if tag else RUN_DIR      # extra experiments are kept apart
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dataset}_{fusion}{'_' + tag if tag else ''}.npz"
    np.savez_compressed(
        out, final_map=final_map, best_map=best["map"], gt=gt,
        final_auc=auc_score(final_map, gt), best_auc=best["auc"], best_epoch=best["epoch"],
        auc_hist=np.array(auc_hist), loss_hist=np.array(losses),
        params=n_params, infer_time=infer_time, train_time=time.time() - t0,
    )
    print(f"[{dataset} | {fusion}] final AUC {auc_score(final_map, gt):.4f} | best AUC {best['auc']:.4f} "
          f"(epoch {best['epoch']}) | inference {infer_time:.3f}s -> {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", help="Cri | aviris_1 | aviris_2 | San_Diego | all")
    p.add_argument("--fusion", default="gated", help="gated | add | spatial | spectral | all")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1, help="evaluate AUC every N epochs (official code: 1)")
    p.add_argument("--d_state", type=int, default=64, help="Mamba state size (official code: 64)")
    p.add_argument("--official", action="store_true",
                   help="use the official code's settings: Mamba state 64, 3x3 conv only, 12-band groups, "
                        "150 epochs, AUC checked every epoch")
    p.add_argument("--tag", default=None, help="name suffix; tagged runs go to results/runs/extra/")
    a = p.parse_args()
    datasets = list(DATASETS) if a.dataset == "all" else [a.dataset]
    fusions = ["gated", "add", "spatial", "spectral"] if a.fusion == "all" else [a.fusion]
    kw = dict(epochs=a.epochs, eval_every=a.eval_every, d_state=a.d_state)
    if a.official:
        kw.update(OFFICIAL)
    for d in datasets:
        for f in fusions:
            train(d, f, batch=a.batch, seed=a.seed, tag=a.tag, **kw)
