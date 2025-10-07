# -*- coding: utf-8 -*-
"""
config.py
----------
Central configuration file for all three pipelines:
    • real neural recordings (private dataset)
    • structured synthetic suite
    • non-Gaussian stochastic synthetic suite

Each Config class defines dataset-specific hyper-parameters while keeping
shared conventions (batch size, learning rate, device, etc.) consistent.

NOTE:
- The real neural data used in the paper cannot be shared publicly.
  The `RealDataConfig` therefore contains only placeholder paths and comments
  describing the expected structure.
- Synthetic configs reproduce the public synthetic results reported in the paper.
"""

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ==============================================================
#  Base configuration (shared knobs for all datasets)
# ==============================================================

@dataclass
class BaseConfig:
    """Shared defaults across all experiments."""
    # Generic training setup
    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model dimensions
    d_hidden: int = 64
    d_timectx: int = 8
    d_proj: int = 32

    # Temporal windowing
    T_in: int = 100
    T_out: int = 20
    stride: int = 20

    # Regularization weights (active terms only)
    lambda_Sraw: float = 1e-4
    lambda_cont: float = 0.0
    lambda_vel:  float = 1.0
    lambda_curv: float = 0.6
    use_row_gain: bool = True

    # Reproducibility
    master_seed: int = 0
    es_patience: int = 10
    es_min_delta: float = 1e-4


# ==============================================================
#  Real neural dataset (private)
# ==============================================================

@dataclass
class RealDataConfig(BaseConfig):
    """
    Configuration for real deep-brain recordings.

    Expected input (private, not distributed):
      • MATLAB .mat file containing a 4-D array:
            shape = (phase, trial, channel, time)
        where:
            phase ∈ {0: Wait, 1: React, 2: Reach, 3: Return}
            channel = 80  (10 electrodes per region × 8 regions)
            time    ≈ 400 samples at 1 kHz  (downsampled)
      • Each trial corresponds to a 400 ms segment per behavioral phase.

    Update `mat_path` to point to your local copy of the dataset.
    """

    mat_path: str = "PATH/TO/AllPhases_CleanReordered.mat"   # placeholder
    out_dir: Path = Path("./out/real")

    num_trials_per_phase: int = 326
    num_channels_total: int = 80
    num_regions: int = 8
    chans_per_region: int = 10
    phases: List[str] = ("Wait", "React", "Reach", "Return")

    # Region mapping (adjust to match your dataset)
    region_to_channels: Dict[str, List[int]] = None

    def __post_init__(self):
        if self.region_to_channels is None:
            self.region_to_channels = {
                "GPi1_L": list(range(0, 10)),
                "GPi1_R": list(range(10, 20)),
                "GPi2_L": list(range(20, 30)),
                "GPi2_R": list(range(30, 40)),
                "VIM_L":  list(range(40, 50)),
                "VIM_R":  list(range(50, 60)),
                "STN_L":  list(range(60, 70)),
                "STN_R":  list(range(70, 80)),
            }


# ==============================================================
#  Structured synthetic dataset
# ==============================================================

@dataclass
class SynthStructuredConfig(BaseConfig):
    """
    Configuration for the structured synthetic suite.

    Synthetic generator:
      x_{t+1} = A_phase * x_t + ε_t
    where A_phase is a known directed VAR(1) matrix
    (spectral radius < 1, zero diagonal), and ε_t ∼ N(0, σ²).

    The generator creates 4 behavioral phases with distinct motifs
    (hub, bilateral, hemispheric, etc.) for connectivity-recovery tasks.
    """

    out_dir: Path = Path("./out/synthetic_structured")
    num_nodes: int = 8
    num_phases: int = 4
    trials_per_phase: int = 60
    seq_len: int = 400
    noise_std: float = 0.1
    spectral_radius: float = 0.9


# ==============================================================
#  Non-Gaussian (stochastic) synthetic dataset
# ==============================================================

@dataclass
class SynthStochasticConfig(BaseConfig):
    """
    Configuration for the stochastic / non-Gaussian synthetic suite.

    Synthetic generator:
      x_{t+1} = A_phase * tanh(x_t) + ε_t,
    with ε_t drawn from a Laplace (non-Gaussian) distribution to
    induce heavy-tailed noise and causal asymmetry.

    This suite is used to evaluate graph learning under
    non-linear and non-Gaussian dynamics.
    """

    out_dir: Path = Path("./out/synthetic_stochastic")
    num_nodes: int = 8
    num_phases: int = 4
    trials_per_phase: int = 60
    seq_len: int = 400
    noise_scale: float = 0.1
    laplace_scale: float = 0.5
    spectral_radius: float = 0.9
