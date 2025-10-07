# config.py
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import torch

@dataclass
class Config:
    """Global configuration for the BACE framework."""

    # -----------------------------
    # Paths
    # -----------------------------
    mat_path: str = "./data/AllPhases_CleanReordered.mat"  # Placeholder (data not included)
    out_dir: Path = Path("./out")
    do_reliability: bool = False

    # -----------------------------
    # Data layout
    # -----------------------------
    num_trials_per_phase: int = 326
    window_length_raw: int = 2000
    num_channels_total: int = 80
    num_regions: int = 8
    chans_per_region: int = 10
    phases: List[str] = None

    target_len: int = 400
    trim_start: int = 3
    trim_end: int = 3

    # -----------------------------
    # Sliding windows
    # -----------------------------
    T_in: int = 100
    T_out: int = 20
    stride: int = 20

    # -----------------------------
    # Model dimensions
    # -----------------------------
    d_hidden: int = 64
    d_timectx: int = 8
    d_proj: int = 32

    # -----------------------------
    # Training
    # -----------------------------
    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_seed: int = 0
    es_patience: int = 10
    es_min_delta: float = 1e-4

    # -----------------------------
    # Regularization weights
    # -----------------------------
    lambda_1s: float = 0.0
    lambda_Sraw: float = 1e-4
    lambda_var: float = 0.0
    lambda_cont: float = 0.0
    lambda_vel: float = 1.0
    lambda_curv: float = 0.6
    use_row_gain: bool = True

    # -----------------------------
    # Reliability
    # -----------------------------
    split_half_K: int = 50
    rng_seed_reliability: int = 123

    region_to_channels: Dict[str, List[int]] = None


def finalize_config(cfg):
    """Assign default region mappings and phases, and validate structure."""
    if cfg.phases is None:
        cfg.phases = ['Wait', 'React', 'Reach', 'Return']

    if cfg.region_to_channels is None:
        cfg.region_to_channels = {
            'GPi1_L': list(range(0, 10)),   'GPi1_R': list(range(10, 20)),
            'GPi2_L': list(range(20, 30)),  'GPi2_R': list(range(30, 40)),
            'VIM_L':  list(range(40, 50)),  'VIM_R':  list(range(50, 60)),
            'STN_L':  list(range(60, 70)),  'STN_R':  list(range(70, 80)),
        }

    all_ch = sorted([c for lst in cfg.region_to_channels.values() for c in lst])
    assert len(all_ch) == cfg.num_channels_total, "Region mapping must cover all channels."
    assert all_ch == list(range(cfg.num_channels_total)), "Region mapping must be a permutation of 0..79."
    for k, v in cfg.region_to_channels.items():
        assert len(v) == cfg.chans_per_region, f"{k} must have {cfg.chans_per_region} channels."

    return cfg


cfg = finalize_config(Config())
