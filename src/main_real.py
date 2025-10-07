# main_real.py
"""
Main training pipeline for BACE (Behavior-Adaptive Connectivity Estimation)
---------------------------------------------------------------------------
This script runs the complete BACE model on real neural recordings.

Notes:
- No raw neural data are included in this repository for privacy reasons.
- To reproduce, place your preprocessed MATLAB file under ./data/
  with variable: neuralDataAllPhases_reordered ∈ R^(4 × trials × 80 × timepoints)
- This script assumes the data are already downsampled to 1 kHz.
- Statistical resampling procedures (bootstrap, reliability) are disabled here.
---------------------------------------------------------------------------

Usage:
    python src/main_real.py
"""

import os
import torch
from src.config import cfg
from src.utils import set_seed, ensure_out_dirs, save_heatmap
from src.data_utils import (
    load_mat_trials,
    split_by_trials,
    compute_train_channel_stats,
    apply_channel_norm,
    SlidingForecastDataset,
    make_loaders_from_trial_lists,
)
from src.model import PS_APM_Seq
from src.train_eval import train_full, eval_test


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and configuration
    # -------------------------------------------------------------------------
    print("\n=== BACE: Behavior-Adaptive Connectivity Estimation (Real Data) ===\n")

    set_seed(cfg.master_seed)
    cfg.out_dir = "./out_real"
    ensure_out_dirs(cfg)

    # Public-safe placeholder path (edit locally as needed)
    cfg.mat_path = "./data/AllPhases_CleanReordered.mat"

    if not os.path.exists(cfg.mat_path):
        raise FileNotFoundError(
            f"Expected .mat file not found at {cfg.mat_path}\n"
            "Please place your preprocessed data under ./data/."
        )

    # -------------------------------------------------------------------------
    # 2. Load and prepare data
    # -------------------------------------------------------------------------
    print("Loading data...")
    data, labels = load_mat_trials(cfg.mat_path)   # already 1 kHz, [Ttot, 80, T]
    print(f"Loaded {data.shape[0]} trials × {data.shape[1]} channels × {data.shape[2]} timepoints")

    # Split trials reproducibly
    train_trials, val_trials, test_trials = split_by_trials(labels, seed=cfg.master_seed)
    print(f"Split trials → train {len(train_trials)}, val {len(val_trials)}, test {len(test_trials)}")

    # Channel-wise z-score using train set
    ch_mean, ch_std = compute_train_channel_stats(data, train_trials)
    data_z = apply_channel_norm(data, ch_mean, ch_std)

    # -------------------------------------------------------------------------
    # 3. Dataset and loaders
    # -------------------------------------------------------------------------
    ds = SlidingForecastDataset(
        data_z, labels, cfg.region_to_channels,
        T_in=cfg.T_in, T_out=cfg.T_out, stride=cfg.stride
    )
    train_loader, val_loader, test_loader = make_loaders_from_trial_lists(
        ds, train_trials, val_trials, test_trials
    )

    print(f"Prepared {len(train_loader.dataset)} training windows "
          f"and {len(test_loader.dataset)} test windows.\n")

    # -------------------------------------------------------------------------
    # 4. Model initialization
    # -------------------------------------------------------------------------
    model = PS_APM_Seq(N=ds.N, C=ds.C, cfg=cfg).to(cfg.device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model initialized ({param_count:.2f}M parameters, device={cfg.device})")

    # -------------------------------------------------------------------------
    # 5. Training
    # -------------------------------------------------------------------------
    print("\n--- Training ---")
    train_full(model, train_loader, val_loader, cfg)

    # -------------------------------------------------------------------------
    # 6. Evaluation
    # -------------------------------------------------------------------------
    print("\n--- Evaluation ---")
    mse, mse_bl, gain = eval_test(model, test_loader, cfg)
    print(f"Test MSE: {mse:.4e} | Baseline: {mse_bl:.4e} | Gain: {gain:.1f}%")

    # -------------------------------------------------------------------------
    # 7. Save learned adjacency matrices (phase-specific graphs)
    # -------------------------------------------------------------------------
    A_eff = model.graphs.export_all(effective=True)
    os.makedirs(os.path.join(cfg.out_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "figs"), exist_ok=True)
    torch.save(A_eff, os.path.join(cfg.out_dir, "graphs", "A_effective.pt"))

    for p in range(4):
        save_heatmap(
            abs(A_eff[p]),
            os.path.join(cfg.out_dir, "figs", f"learned_A_phase{p}.png"),
            f"|A| (Phase = {cfg.phases[p]})",
            labels=list(cfg.region_to_channels.keys()),
            vmin=0, vmax=abs(A_eff).max()
        )

    print("\nSaved results under:", cfg.out_dir)
    print(" - metrics/: training & test losses")
    print(" - figs/: learned |A| per phase")
    print(" - graphs/: adjacency arrays\n")


if __name__ == "__main__":
    main()
