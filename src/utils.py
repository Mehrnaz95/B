# utils.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# -----------------------------
# Randomness Control
# -----------------------------
def set_seed(seed: int = 0):
    """Ensure full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Directory Management
# -----------------------------
def ensure_out_dirs(cfg):
    """Create output directories for metrics, figs, and graphs."""
    for sub in ("", "figs", "metrics", "graphs"):
        os.makedirs(os.path.join(cfg.out_dir, sub), exist_ok=True)


# -----------------------------
# Visualization
# -----------------------------
def save_heatmap(M: np.ndarray, path: str, title: str,
                 labels=None, vmin=None, vmax=None):
    """Save matrix as heatmap (used for learned adjacencies)."""
    if labels is None:
        labels = [f"R{i}" for i in range(M.shape[0])]
    plt.figure(figsize=(6.4, 5.2))
    im = plt.imshow(M, aspect='equal', vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("To")
    plt.ylabel("From")
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right')
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
