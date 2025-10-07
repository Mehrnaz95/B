# train_eval.py
"""
Training and evaluation utilities for BACE
------------------------------------------
Contains:
- train_full(): end-to-end model optimization
- eval_test(): compute test and baseline MSE
- save_curves(): visualize training/validation loss
------------------------------------------
This version is anonymized and data-agnostic: it uses the modular
model, config, and dataset without any patient data references.
"""

import os
import math
import csv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.utils import ensure_out_dirs
from src.model import L1_on_S

# =============================================================================
# 1. Training Loop
# =============================================================================

def train_full(model, train_loader, val_loader, cfg):
    """
    End-to-end model training with early stopping on validation loss.
    """
    ensure_out_dirs(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    patience = 0
    train_losses, val_losses = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for X_in, Y_out, phases in train_loader:
            X_in, Y_out, phases = X_in.to(cfg.device), Y_out.to(cfg.device), phases.to(cfg.device)

            # Forward pass
            Y_hat, y1 = model(X_in, phases, teacher=None, sched_p=0.0, use_neigh=True)
            
            # Main loss: weighted MSE across prediction horizon
            gamma = 3.0
            w = torch.logspace(0, math.log10(gamma), steps=cfg.T_out, device=cfg.device)
            l_mse = ((Y_hat - Y_out) ** 2 * w.view(1, 1, 1, -1)).mean()

            # Regularization terms
            loss = (
                l_mse
                + cfg.lambda_Sraw * L1_on_S(model.graphs)
                + cfg.lambda_cont * F.mse_loss(Y_hat[..., 0], X_in[..., -1])
                + cfg.lambda_vel * _velocity_loss(X_in, Y_hat, Y_out)
                + cfg.lambda_curv * _curvature_loss(X_in, Y_hat, Y_out)
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_in, Y_out, phases in val_loader:
                X_in, Y_out, phases = X_in.to(cfg.device), Y_out.to(cfg.device), phases.to(cfg.device)
                Y_hat, _ = model(X_in, phases, use_neigh=True)
                val_loss += F.mse_loss(Y_hat, Y_out, reduction='mean').item()
        val_loss /= max(1, len(val_loader))

        # Early stopping
        train_losses.append(running_loss / max(1, len(train_loader)))
        val_losses.append(val_loss)
        if val_loss < best_val - cfg.es_min_delta:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "graphs", "best.pt"))
        else:
            patience += 1
            if patience >= cfg.es_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch == 1 or epoch % 5 == 0:
            print(f"[Epoch {epoch:03d}] Train={train_losses[-1]:.4f}, Val={val_loss:.4f}")

    save_curves(train_losses, val_losses, cfg)
    return train_losses, val_losses


# =============================================================================
# 2. Evaluation
# =============================================================================

def eval_test(model, test_loader, cfg):
    """
    Evaluate test performance and compare to copy-last baseline.
    """
    model.eval()
    mse, mse_bl = 0.0, 0.0
    n = 0

    with torch.no_grad():
        for X_in, Y_out, phases in test_loader:
            X_in, Y_out, phases = X_in.to(cfg.device), Y_out.to(cfg.device), phases.to(cfg.device)
            Y_hat, _ = model(X_in, phases)
            mse += F.mse_loss(Y_hat, Y_out, reduction='sum').item()
            # copy-last baseline
            bl = X_in[:, :, :, -1:].repeat(1, 1, 1, cfg.T_out)
            mse_bl += F.mse_loss(bl, Y_out, reduction='sum').item()
            n += Y_out.numel()

    mse /= n
    mse_bl /= n
    gain = 100 * (1 - mse / mse_bl)

    # Save metrics
    os.makedirs(os.path.join(cfg.out_dir, "metrics"), exist_ok=True)
    with open(os.path.join(cfg.out_dir, "metrics", "test_mse.txt"), "w") as f:
        f.write(f"Test MSE: {mse:.6e}\n")
        f.write(f"Copy-last baseline: {mse_bl:.6e}\n")
        f.write(f"Relative gain: {gain:.2f}%\n")

    print(f"[Test] MSE={mse:.3e}, Baseline={mse_bl:.3e}, Gain={gain:.1f}%")
    return mse, mse_bl, gain


# =============================================================================
# 3. Auxiliary Losses
# =============================================================================

def _velocity_loss(X_in, Y_hat, Y_out):
    """Match initial velocity between prediction and target."""
    x_last = X_in[..., -1]
    d_pred = Y_hat[..., 0] - x_last
    d_true = Y_out[..., 0] - x_last
    return F.mse_loss(d_pred, d_true)


def _curvature_loss(X_in, Y_hat, Y_out):
    """Match second differences (curvature) across time."""
    def second_diff(x):
        return x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    return F.mse_loss(second_diff(Y_hat), second_diff(Y_out))


# =============================================================================
# 4. Visualization
# =============================================================================

def save_curves(train_losses, val_losses, cfg):
    """Save training and validation loss curves."""
    os.makedirs(os.path.join(cfg.out_dir, "figs"), exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "figs", "loss_curves.png"), dpi=150)
    plt.close()

    # Save numeric log
    with open(os.path.join(cfg.out_dir, "metrics", "losses.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train", "val"])
        for i, (a, b) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow([i, a, b])
