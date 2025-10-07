# -*- coding: utf-8 -*-
"""
data_utils.py
--------------
Shared utilities for data handling across all experiments.

This module includes:
    • Synthetic structured suite generator  (Section 3.1, Fig. 7B)
    • Synthetic stochastic non-Gaussian suite generator  (Section 3.1, Fig. 7A)
    • Real neural dataset loader (placeholder example)
    • Sliding-window dataset and train/val/test splitting helpers
    • Normalization (per-channel / per-region)
    • Correlation initialization utilities

Synthetic datasets mirror the paper’s “Structured” and
“Stochastic non-Gaussian” suites (𝓓₁–𝓓₄), each containing
four behavioral regimes defined by distinct adjacency matrices.
"""

import os, math, random
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.signal import lfilter


# ==============================================================
#  Utility: reproducibility
# ==============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================
#  ----------  SYNTHETIC DATA GENERATION  ----------
# ==============================================================

# --------------------------------------------------------------
# Structured suite  (Section 3.1 – “structured suite”)
# --------------------------------------------------------------

def make_gt_graphs_structured(cfg) -> List[np.ndarray]:
    """
    Generate four structured ground-truth adjacency matrices A^(ϕ),
    each with the same row degree (k=2) but shifted nonzero locations.

    These mimic visually structured but distinct connectivity motifs
    as shown in Fig. 7B of the paper.
    """
    N, P = cfg.num_nodes, cfg.num_phases
    A_list = []
    for p in range(P):
        A = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            # two outgoing edges per row, shifted by phase index
            for k in range(2):
                j = (i + p + k + 1) % N
                A[i, j] = np.random.uniform(0.4, 0.8)
        np.fill_diagonal(A, 0.0)
        # scale to spectral radius
        eigs = np.linalg.eigvals(A)
        A = A / (max(abs(eigs)) / cfg.spectral_radius)
        A_list.append(A.astype(np.float32))
    return A_list


def simulate_structured_trials(cfg, A_list):
    """
    Simulate time series x_{t+1} = A * x_t + ε_t  (Gaussian ε).

    Returns:
        X : [P * trials_per_phase, N, seq_len]
        labels : [P * trials_per_phase]  (phase index)
    """
    N, P = cfg.num_nodes, cfg.num_phases
    T = cfg.seq_len
    n_trials = cfg.trials_per_phase
    all_x, all_y = [], []

    for p in range(P):
        A = A_list[p]
        for _ in range(n_trials):
            x = np.zeros((N, T), dtype=np.float32)
            x[:, 0] = np.random.randn(N) * 0.1
            for t in range(1, T):
                noise = np.random.randn(N) * cfg.noise_std
                x[:, t] = A @ x[:, t - 1] + noise
            all_x.append(x)
            all_y.append(p)

    X = np.stack(all_x, axis=0)
    y = np.array(all_y, dtype=np.int64)
    return X, y


# --------------------------------------------------------------
# Stochastic non-Gaussian suite  (Section 3.1 – “stochastic suite”)
# --------------------------------------------------------------

def make_gt_graphs_stochastic(cfg) -> List[np.ndarray]:
    """
    Generate four directed adjacency matrices for the
    stochastic non-Gaussian suite (A^(ϕ)), each sparse and random.
    """
    N, P = cfg.num_nodes, cfg.num_phases
    A_list = []
    for p in range(P):
        A = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            out_idx = np.random.choice(
                [j for j in range(N) if j != i], size=2, replace=False
            )
            A[i, out_idx] = np.random.uniform(0.4, 0.8, size=2)
        np.fill_diagonal(A, 0.0)
        # rescale spectral radius
        eigs = np.linalg.eigvals(A)
        A = A / (max(abs(eigs)) / cfg.spectral_radius)
        A_list.append(A.astype(np.float32))
    return A_list


def simulate_stochastic_trials(cfg, A_list):
    """
    Simulate non-Gaussian, colored-noise dynamics following the form:
        X_t = X_{t-1} + G(X_{t-1}, A) + μ_t,
    with   G(X, A) = (−λ I + γ A) X,
    and μ_t autoregressive Laplace noise (colored & heavy-tailed).

    Returns:
        X : [P * trials_per_phase, N, seq_len]
        labels : [P * trials_per_phase]
    """
    N, P = cfg.num_nodes, cfg.num_phases
    T = cfg.seq_len
    n_trials = cfg.trials_per_phase
    all_x, all_y = [], []

    leak, gain = 0.2, 0.3
    for p in range(P):
        A = A_list[p]
        for _ in range(n_trials):
            x = np.zeros((N, T), dtype=np.float32)
            noise_prev = np.zeros(N, dtype=np.float32)
            for t in range(1, T):
                # autoregressive colored noise
                eps = np.random.laplace(scale=cfg.laplace_scale, size=N).astype(np.float32)
                noise_prev = 0.5 * noise_prev + eps
                x[:, t] = x[:, t - 1] + (-leak * x[:, t - 1] + gain * (A @ x[:, t - 1])) + noise_prev
            all_x.append(x)
            all_y.append(p)

    X = np.stack(all_x, axis=0)
    y = np.array(all_y, dtype=np.int64)
    return X, y


# ==============================================================
#  ----------  REAL DATA HELPERS  ----------
# ==============================================================

def load_real_data_placeholder(mat_path: str):
    """
    Placeholder for real deep-brain dataset loader.

    The private dataset used in the paper is stored as a MATLAB `.mat` file
    with shape (phase, trial, channel, time). To keep the repository
    anonymous, this function includes only the expected structure:

        Example:
            data = np.loadmat(mat_path)["neuralDataAllPhases_reordered"]
            # shape: (4 phases, num_trials, 80 channels, 400 timepoints)

    Expected outputs:
        data_z : np.ndarray [T_tot, 80, 400]   (z-scored per channel)
        labels : np.ndarray [T_tot]            (phase index per trial)
    """
    raise NotImplementedError(
        "Real neural data are private. "
        "Please replace this placeholder with your own loader "
        "following the format described in config.RealDataConfig."
    )


# ==============================================================
#  ----------  SLIDING-WINDOW DATASET  ----------
# ==============================================================

class SlidingForecastDataset(Dataset):
    """
    Dataset yielding sliding windows (X_in, Y_out) per region.
    Compatible with both real and synthetic datasets.

    X_in  : [N, C, T_in]
    Y_out : [N, C, T_out]
    phase : scalar  (0–3)
    """
    def __init__(self, data, labels, N: int, C: int,
                 T_in: int, T_out: int, stride: int = 20):
        super().__init__()
        self.data = data
        self.labels = labels
        self.N = N
        self.C = C
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride

        self.starts = self._compute_starts(data.shape[-1], T_in, T_out, stride)
        self.num_pairs_per_trial = len(self.starts)

    @staticmethod
    def _compute_starts(T_total, T_in, T_out, stride):
        starts = []
        last = T_total - (T_in + T_out)
        s = 0
        while s <= last:
            starts.append(s)
            s += stride
        return starts

    def __len__(self):
        return self.data.shape[0] * self.num_pairs_per_trial

    def __getitem__(self, idx):
        trial_idx = idx // self.num_pairs_per_trial
        w_idx = idx % self.num_pairs_per_trial
        s = self.starts[w_idx]
        phase = int(self.labels[trial_idx])

        trial = self.data[trial_idx]  # [N, C, T_total]
        x = trial[:, :, s:s + self.T_in]
        y = trial[:, :, s + self.T_in:s + self.T_in + self.T_out]

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.tensor(phase, dtype=torch.long)


# ==============================================================
#  ----------  TRAIN / VAL / TEST SPLITTING  ----------
# ==============================================================

def split_by_trials(labels: np.ndarray, seed=0, train_frac=0.7, val_frac=0.15):
    """Split indices by phase, preserving balance."""
    set_seed(seed)
    num_phases = len(np.unique(labels))
    trials_per_phase = len(labels) // num_phases
    train, val, test = [], [], []
    for p in range(num_phases):
        start = p * trials_per_phase
        end = start + trials_per_phase
        ids = list(range(start, end))
        random.shuffle(ids)
        n_train = int(train_frac * trials_per_phase)
        n_val = int(val_frac * trials_per_phase)
        train += ids[:n_train]
        val += ids[n_train:n_train + n_val]
        test += ids[n_train + n_val:]
    return sorted(train), sorted(val), sorted(test)


def make_loaders_from_trials(ds: Dataset, train_ids, val_ids, test_ids, batch_size=64, device="cpu"):
    """Create PyTorch DataLoaders from trial index lists."""
    pin = (device == "cuda")
    train = Subset(ds, _indices_from_trials(ds, train_ids))
    val   = Subset(ds, _indices_from_trials(ds, val_ids))
    test  = Subset(ds, _indices_from_trials(ds, test_ids))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin)
    return train_loader, val_loader, test_loader


def _indices_from_trials(ds: SlidingForecastDataset, trial_list: List[int]) -> List[int]:
    idxs = []
    for t in trial_list:
        for w in range(ds.num_pairs_per_trial):
            idxs.append(t * ds.num_pairs_per_trial + w)
    return idxs


# ==============================================================
#  ----------  NORMALIZATION & CORRELATION INIT  ----------
# ==============================================================

def compute_channel_stats(data: np.ndarray, train_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std from training trials (leak-free).
    data : [T_tot, N, C, T]
    Returns:
        mean : [N, C, 1]
        std  : [N, C, 1]
    """
    train = data[train_ids]
    flat = train.reshape(-1, train.shape[2] * train.shape[3])  # merge N*C
    mean = train.mean(axis=(0, 3), keepdims=True)
    std = train.std(axis=(0, 3), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_norm(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel using train statistics."""
    return ((data - mean) / std).astype(np.float32)


def phase_corr_init_from_ds(ds, train_trials: List[int]) -> List[np.ndarray]:
    """
    Compute per-phase Pearson correlation across regions.
    Used to initialize PhaseGraphs (untrained adjacency priors).
    """
    C_list = []
    for p in range(4):
        xs = []
        for t in train_trials:
            if int(ds.labels[t]) != p:
                continue
            for s in ds.starts:
                xs.append(ds.data[t][:, :, s:s + ds.T_in].mean(1))  # region-mean signal
        if not xs:
            C_list.append(np.zeros((ds.N, ds.N), dtype=np.float32))
            continue
        Xp = np.stack(xs, axis=0)                      # [W, N, T_in]
        Xflat = Xp.transpose(1, 0, 2).reshape(ds.N, -1)
        C = np.corrcoef(Xflat)
        np.fill_diagonal(C, 0.0)
        C_list.append(np.nan_to_num(C).astype(np.float32))
    return C_list
