# data_utils.py
"""
Data loading and dataset utilities for BACE (Behavior-Adaptive Connectivity Estimation)
-------------------------------------------------------------------------------
This module handles:
- Loading trial-based data (either real intracranial recordings or synthetic samples)
- Normalization and train/val/test splitting
- Dataset class for sliding-window forecasting (region × channels × time)
-------------------------------------------------------------------------------
For open review, no real patient data is included.
Instead, reviewers can simulate small random data arrays to test functionality.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict, Tuple
from src.utils import set_seed

# =============================================================================
# 1. DATA LOADING (REAL OR SYNTHETIC)
# =============================================================================

def load_placeholder_data(num_trials=100, num_channels=80, timepoints=400, num_phases=4):
    """
    Placeholder data generator for open-source reproducibility.
    Returns synthetic 'neural' trials of shape [Ttot, 80, 400] and phase labels [Ttot].
    """
    total_trials = num_trials * num_phases
    # Simulated smooth random data per trial
    data = np.random.randn(total_trials, num_channels, timepoints).astype(np.float32)
    labels = np.repeat(np.arange(num_phases), num_trials).astype(np.int64)
    return data, labels


# =============================================================================
# 2. SPLITTING & NORMALIZATION
# =============================================================================

def split_by_trials(labels: np.ndarray, seed=0, train_frac=0.7, val_frac=0.15):
    """
    Split trial indices into train/val/test sets stratified by phase.
    """
    set_seed(seed)
    num_phases = len(np.unique(labels))
    trials_per_phase = len(labels) // num_phases

    train_trials, val_trials, test_trials = [], [], []
    for p in range(num_phases):
        start = p * trials_per_phase
        end = start + trials_per_phase
        phase_trials = list(range(start, end))
        random.shuffle(phase_trials)
        n_train = int(train_frac * trials_per_phase)
        n_val = int(val_frac * trials_per_phase)
        train_trials += phase_trials[:n_train]
        val_trials += phase_trials[n_train:n_train+n_val]
        test_trials += phase_trials[n_train+n_val:]

    return sorted(train_trials), sorted(val_trials), sorted(test_trials)


def compute_train_channel_stats(data: np.ndarray, train_trials: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std ONLY on training trials (leak-free normalization).
    """
    train = data[train_trials]  # [N_train, C, T]
    flat = train.transpose(1, 0, 2).reshape(data.shape[1], -1)
    ch_mean = flat.mean(axis=1, keepdims=True)
    ch_std = flat.std(axis=1, keepdims=True) + 1e-6
    return ch_mean.astype(np.float32), ch_std.astype(np.float32)


def apply_channel_norm(data: np.ndarray, ch_mean: np.ndarray, ch_std: np.ndarray) -> np.ndarray:
    """Apply (x - mean) / std normalization per channel."""
    return ((data - ch_mean) / ch_std).astype(np.float32)


# =============================================================================
# 3. DATASET: Sliding Forecast Windows
# =============================================================================

class SlidingForecastDataset(Dataset):
    """
    Constructs windowed (input, output) pairs for short-term forecasting.

    Each sample corresponds to:
      X_in  = [N_regions × C_channels × T_in]
      Y_out = [N_regions × C_channels × T_out]

    Labels provide the behavioral phase index for each trial.
    """

    def __init__(self,
                 data: np.ndarray,            # [Ttot, 80, 400]
                 labels: np.ndarray,          # [Ttot] phase labels
                 region_map: Dict[str, List[int]],
                 T_in=100, T_out=20, stride=20):
        super().__init__()
        self.data = data
        self.labels = labels
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride
        self.region_names = list(region_map.keys())
        self.region_chans = list(region_map.values())
        self.N = len(self.region_names)
        self.C = len(self.region_chans[0])  # assumes equal channels per region
        self.starts_per_trial = self._compute_starts(data.shape[-1], T_in, T_out, stride)
        self.num_pairs_per_trial = len(self.starts_per_trial)

    @staticmethod
    def _compute_starts(T_total, T_in, T_out, stride):
        """Compute start indices for each sliding window."""
        starts = []
        last = T_total - (T_in + T_out)
        s = 0
        while s <= last:
            starts.append(s)
            s += stride
        return starts

    def __len__(self):
        return self.data.shape[0] * self.num_pairs_per_trial

    def trial_count(self):
        return self.data.shape[0]

    def index_for(self, trial_idx: int, window_idx: int) -> int:
        """Map trial + window index → global dataset index."""
        return trial_idx * self.num_pairs_per_trial + window_idx

    def __getitem__(self, global_idx):
        trial_idx = global_idx // self.num_pairs_per_trial
        w_idx = global_idx % self.num_pairs_per_trial
        s = self.starts_per_trial[w_idx]

        phase = int(self.labels[trial_idx])
        trial = self.data[trial_idx]  # [80, T]

        X_in_regions, Y_out_regions = [], []
        for chans in self.region_chans:
            x_region = trial[chans, s:s + self.T_in]
            y_region = trial[chans, s + self.T_in:s + self.T_in + self.T_out]
            X_in_regions.append(x_region)
            Y_out_regions.append(y_region)

        X_in = np.stack(X_in_regions, axis=0)   # [N, C, T_in]
        Y_out = np.stack(Y_out_regions, axis=0) # [N, C, T_out]

        return (
            torch.from_numpy(X_in).float(),
            torch.from_numpy(Y_out).float(),
            torch.tensor(phase, dtype=torch.long)
        )


# =============================================================================
# 4. DATA LOADERS
# =============================================================================

def subset_indices_for_trials(ds: SlidingForecastDataset, trial_list: List[int]) -> List[int]:
    """Return dataset indices corresponding to specific trial IDs."""
    idxs = []
    for t in trial_list:
        for w in range(ds.num_pairs_per_trial):
            idxs.append(ds.index_for(t, w))
    return idxs


def make_loaders_from_trial_lists(ds: SlidingForecastDataset,
                                  train_trials: List[int],
                                  val_trials: List[int],
                                  test_trials: List[int],
                                  batch_size=64,
                                  device="cpu"):
    """Build DataLoaders for train/val/test splits."""
    train_idx = subset_indices_for_trials(ds, train_trials)
    val_idx = subset_indices_for_trials(ds, val_trials)
    test_idx = subset_indices_for_trials(ds, test_trials)

    pin = (device == 'cuda')
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=pin, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
                            drop_last=False, pin_memory=pin, num_workers=0)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False,
                             drop_last=False, pin_memory=pin, num_workers=0)
    return train_loader, val_loader, test_loader
