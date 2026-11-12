"""Data loading, patching, the RX baseline and evaluation helpers."""
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

DATA_DIR = Path(__file__).parent / "data"

# file name -> pretty name used in figures (same names as in the paper)
DATASETS = {
    "Cri": "CRI",
    "aviris_1": "AVIRIS-1",
    "aviris_2": "AVIRIS-2",
    "San_Diego": "San Diego",
}

# AUC values reported in the paper (Table 2 / Table 5) for the datasets we have
PAPER_AUC = {
    "Cri":       {"RX": 0.9989, "DMS2F-HAD": 0.9999, "spectral": 0.9997, "spatial": 0.9990, "add": 0.9999},
    "aviris_1":  {"RX": 0.8866, "DMS2F-HAD": 0.9891, "spectral": 0.9856, "spatial": 0.9716, "add": 0.9710},
    "aviris_2":  {"RX": 0.9181, "DMS2F-HAD": 0.9851, "spectral": 0.8701, "spatial": 0.9755, "add": 0.9999},
    "San_Diego": {"RX": 0.9403, "DMS2F-HAD": 0.9722, "spectral": 0.9158, "spatial": 0.9674, "add": 0.9532},
}


def load_hsi(name):
    """Return the HSI cube (H, W, C) min-max normalised to [0, 1] and the binary GT map (H, W)."""
    mat = sio.loadmat(DATA_DIR / f"{name}.mat")
    cube = mat["data"].astype(np.float32)
    cube = (cube - cube.min()) / (cube.max() - cube.min())
    gt = (mat["map"] > 0).astype(np.uint8)
    return cube, gt


def false_color(cube):
    """3-band false-colour image for display (bands at 25 %, 50 %, 75 % of the spectrum)."""
    C = cube.shape[2]
    rgb = cube[:, :, [int(C * 0.75), int(C * 0.5), int(C * 0.25)]].astype(np.float64)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


# ----------------------------------------------------------------------------- patching
def _pad_amount(n, size, stride):
    return 0 if (n - size) % stride == 0 else stride - (n - size) % stride


def extract_patches(cube, size=16, stride=8):
    """(H, W, C) numpy -> overlapping patches (N, C, size, size) tensor (paper Sec. 3.1.1)."""
    x = torch.from_numpy(cube).permute(2, 0, 1)[None]              # (1, C, H, W)
    H, W = cube.shape[:2]
    x = F.pad(x, (0, _pad_amount(W, size, stride), 0, _pad_amount(H, size, stride)))   # zeros, as in the official code
    cols = F.unfold(x, size, stride=stride)                         # (1, C*size*size, N)
    return cols[0].T.reshape(-1, cube.shape[2], size, size).contiguous()


def fold_patches(patches, H, W, size=16, stride=8):
    """Inverse of extract_patches: average the overlapping reconstructed patches -> (C, H, W)."""
    Hp, Wp = H + _pad_amount(H, size, stride), W + _pad_amount(W, size, stride)
    cols = patches.reshape(patches.shape[0], -1).T[None]
    summed = F.fold(cols, (Hp, Wp), size, stride=stride)
    count = F.fold(torch.ones_like(cols), (Hp, Wp), size, stride=stride)
    return (summed / count)[0, :, :H, :W]


# ----------------------------------------------------------------------------- detection
def rx_detector(cube):
    """Global Reed-Xiaoli detector: Mahalanobis distance of each pixel to the image background."""
    X = cube.reshape(-1, cube.shape[2]).astype(np.float64)
    Xc = X - X.mean(0)
    cov_inv = np.linalg.pinv(np.cov(Xc, rowvar=False))
    return np.einsum("ij,jk,ik->i", Xc, cov_inv, Xc).reshape(cube.shape[:2])


def auc_score(score_map, gt):
    return roc_auc_score(gt.ravel(), score_map.ravel())


def roc(score_map, gt):
    fpr, tpr, _ = roc_curve(gt.ravel(), score_map.ravel())
    return fpr, tpr


def normalize(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-12)
