# -*- coding: utf-8 -*-
"""
main_synth_structured.py
------------------------
Run BACE on the *Structured Synthetic Suite* (Section 3.1 of the paper).

This suite consists of four regimes 𝓓₁–𝓓₄, each governed by a distinct
sparse adjacency matrix A₍φ₎ with identical row degree but shifted edge placement.
These datasets are designed to mimic the organization of the real neural data
(8 regions × 4 behavioral phases), while having known ground-truth connectivity.

Outputs:
    out/synthetic_structured/
        ├── metrics/   (training curves, recovery metrics)
        ├── figs/      (overlay trajectories, |A| heatmaps)
        └── graphs/    (A_effective.npy, A_gt.npy)
"""

import os
import numpy as np
import torch

from config import SynthStructuredConfig as Config
from model import PS_APM_Seq
from data_utils import (
    make_gt_graphs_structured,
    simulate_structured_trials,
    SlidingForecastDataset,
    make_loaders_from_trials,
)
from utils import ensure_dirs, set_seed
from train_eval import train_full, overlay_examples, save_phase_adjacency_plots


# ==============================================================
#  ----------  MAIN  ----------
# ==============================================================

def main():
    cfg = Config()
    ensure_dirs(cfg.out_dir)
    set_seed(cfg.master_seed)
    print("Device:", cfg.device)
    torch.manual_seed(cfg.master_seed)
    np.random.seed(cfg.master_seed)

    # ----------------------------------------------------------
    # 1) Generate ground-truth graphs (A_gt)
    # ----------------------------------------------------------
    print("Generating structured suite graphs (𝓓₁–𝓓₄)...")
    A_gt = make_gt_graphs_structured(cfg)
    np.save(os.path.join(cfg.out_dir, "graphs", "A_gt.npy"), A_gt)

    # ----------------------------------------------------------
    # 2) Simulate synthetic multivariate time series
    # ----------------------------------------------------------
    print("Simulating synthetic trials...")
    data, labels = simulate_structured_trials(cfg, A_gt)
    print("Data shape:", data.shape, "| Labels shape:", labels.shape)

    # ----------------------------------------------------------
    # 3) Split into train / val / test (balanced)
    # ----------------------------------------------------------
    from data_utils import split_by_trials
    train_ids, val_ids, test_ids = split_by_trials(labels, seed=cfg.master_seed)
    print(f"Trials per phase: {cfg.trials_per_phase} | Train {len(train_ids)}, Val {len(val_ids)}, Test {len(test_ids)}")

    # ----------------------------------------------------------
    # 4) Create Dataset + DataLoaders
    # ----------------------------------------------------------
    ds = SlidingForecastDataset(
        data, labels, N=cfg.num_nodes, C=1,
        T_in=cfg.T_in, T_out=cfg.T_out, stride=cfg.stride
    )
    train_loader, val_loader, test_loader = make_loaders_from_trials(
        ds, train_ids, val_ids, test_ids,
        batch_size=cfg.batch_size, device=cfg.device
    )

    # ----------------------------------------------------------
    # 5) Initialize model
    # ----------------------------------------------------------
    model = PS_APM_Seq(N=cfg.num_nodes, C=1, cfg=cfg).to(cfg.device)

    # Initialize graphs randomly (no correlation priors for synthetic)
    with torch.no_grad():
        for p in range(cfg.num_phases):
            model.graphs.S[p].uniform_(-0.1, 0.1)

    # ----------------------------------------------------------
    # 6) Train
    # ----------------------------------------------------------
    train_full(model, train_loader, val_loader, cfg)

    # ----------------------------------------------------------
    # 7) Evaluate adjacency recovery
    # ----------------------------------------------------------
    from data_utils import evaluate_graph_recovery
    A_learned = model.graphs.export_eff()
    np.save(os.path.join(cfg.out_dir, "graphs", "A_effective.npy"), A_learned)

    metrics = evaluate_graph_recovery(A_learned, A_gt, top_k=cfg.row_degree)
    print("Recovery metrics:")
    for p in range(cfg.num_phases):
        print(f"  Phase {p+1}  Corr = {metrics['corr'][p]:.3f}  F1@k_row = {metrics['f1'][p]:.3f}")
    print(f"  Mean Corr = {np.mean(metrics['corr']):.3f}  Mean F1 = {np.mean(metrics['f1']):.3f}")

    np.save(os.path.join(cfg.out_dir, "metrics", "recovery_metrics.npy"), metrics)

    # ----------------------------------------------------------
    # 8) Visualizations (as in Fig. 5B of paper)
    # ----------------------------------------------------------
    overlay_examples(model, ds, test_loader, cfg, region_idx=0)
    save_phase_adjacency_plots(model, cfg)

    print("Done. Results saved under:", cfg.out_dir)


# ==============================================================
#  Entry Point
# ==============================================================

if __name__ == "__main__":
    main()
