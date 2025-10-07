# model.py
"""
Model architecture for BACE (Behavior-Adaptive Connectivity Estimation)
-----------------------------------------------------------------------
This module defines the main neural components used in the paper:
- Per-region temporal encoder (GRU)
- Phase-specific graph learner (signed adjacency)
- Graph projection layer for spatial message passing
- Autoregressive attention-based decoder
-----------------------------------------------------------------------
All components are implemented in PyTorch and modularly combined
in the PS_APM_Seq (Phase-Specific Adaptive Projection Model).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# =============================================================================
# 1. PER-REGION TEMPORAL ENCODER
# =============================================================================

class PerRegionGRU(nn.Module):
    """
    Each brain region has its own GRU that encodes temporal dynamics
    from its local multi-channel signals.
    """
    def __init__(self, N=8, C=10, d_hidden=64):
        super().__init__()
        self.N = N
        self.grus = nn.ModuleList([
            nn.GRU(input_size=C, hidden_size=d_hidden, batch_first=False)
            for _ in range(N)
        ])
        self.ln = nn.LayerNorm(d_hidden)

    def forward(self, x):
        """
        Args:
            x: [B, N, C, T]  → batch, region, channel, time
        Returns:
            H: [B, N, T, d_hidden]
        """
        B, N, C, T = x.shape
        outs = []
        for r in range(N):
            xr = x[:, r].permute(2, 0, 1)        # [T, B, C]
            y, _ = self.grus[r](xr)              # [T, B, d_hidden]
            y = self.ln(y)
            outs.append(y.permute(1, 0, 2))      # [B, T, d_hidden]
        return torch.stack(outs, dim=1)          # [B, N, T, d_hidden]


# =============================================================================
# 2. TIME-POSITIONAL ENCODING
# =============================================================================

class TimePositional(nn.Module):
    """Simple linear time embedding (learns position encoding)."""
    def __init__(self, d_time=8, T_in=100):
        super().__init__()
        self.emb = nn.Linear(1, d_time)
        self.T_in = T_in

    def forward(self, B, T):
        t = torch.linspace(0., 1., steps=T, device='cuda' if torch.cuda.is_available() else 'cpu')
        t = t.view(1, T, 1).repeat(B, 1, 1)
        return torch.tanh(self.emb(t))           # [B, T, d_time]


# =============================================================================
# 3. PHASE-SPECIFIC GRAPH LEARNER
# =============================================================================

class PhaseGraphs(nn.Module):
    """
    Learns one signed adjacency matrix A_p per behavioral phase.
    Each row of A_p is L1-normalized (sum of abs = 1).
    Optional per-row 'gain' parameters allow flexible scaling.
    """
    def __init__(self, N=8, P=4, eps=1e-6, use_row_gain=True):
        super().__init__()
        self.N, self.P, self.eps = N, P, eps
        self.S = nn.Parameter(torch.zeros(P, N, N))  # signed adjacency
        self.register_buffer('I', torch.eye(N))
        self.use_row_gain = use_row_gain
        if use_row_gain:
            self.G = nn.Parameter(torch.zeros(P, N))  # per-phase row gain

    def _zero_diag(self, X):
        return X * (1.0 - self.I)

    def _row_norm_l1_signed(self, S):
        """Normalize each row to have unit L1 norm."""
        S = self._zero_diag(S)
        denom = S.abs().sum(-1, keepdim=True).clamp_min(self.eps)
        return S / denom

    def forward(self, phases: torch.Tensor):
        """
        Args:
            phases: [B] phase index for each batch element
        Returns:
            A_batch: [B, N, N] effective adjacency matrices
        """
        A_tilde = self._row_norm_l1_signed(self.S)
        A = A_tilde[phases]
        if self.use_row_gain:
            g_all = F.softplus(self.G) + 1e-6
            g_all = g_all * (self.N / g_all.sum(dim=-1, keepdim=True).clamp_min(self.eps))
            g = g_all[phases].unsqueeze(-1)  # [B, N, 1]
            A = A * g
        return A

    @torch.no_grad()
    def export_eff(self):
        """Return effective adjacency matrices (post-gain)."""
        A = self._row_norm_l1_signed(self.S)
        if self.use_row_gain:
            g = F.softplus(self.G) + 1e-6
            g = g * (self.N / (g.sum(dim=-1, keepdim=True) + 1e-12))
            A = A * g[..., None]
        return A.detach().cpu().numpy()


def L1_on_S(graphs: PhaseGraphs):
    """L1 regularization on off-diagonal entries."""
    S = graphs.S
    I = torch.eye(S.size(-1), device=S.device)
    return (S.abs() * (1.0 - I)).sum()


# =============================================================================
# 4. GRAPH PROJECTOR
# =============================================================================

class GraphProjector(nn.Module):
    """
    Applies spatial message passing using learned adjacency A.
    Uses spectral normalization for numerical stability.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_self = nn.Linear(d_in, d_out, bias=False)
        self.W_neigh = spectral_norm(nn.Linear(d_in, d_out, bias=False))

    def forward(self, H_seq, A_batch):
        """
        Args:
            H_seq: [B, N, T, d_in]
            A_batch: [B, N, N]
        Returns:
            Z_seq: [B, N, T, d_out]
        """
        Hs = self.W_self(H_seq)
        Hn = self.W_neigh(H_seq)
        Zn = torch.einsum('bij,bjtd->bitd', A_batch, Hn)
        return F.leaky_relu(Hs + Zn, 0.1)


# =============================================================================
# 5. AUTOREGRESSIVE DECODER
# =============================================================================

class ARHead(nn.Module):
    """
    Autoregressive decoder with temporal attention over encoder memory.
    Each output step attends over all past encoder states.
    """
    def __init__(self, d_in, C, T_out, h_dim=None, use_kv=True, attn_p=0.1):
        super().__init__()
        h_dim = d_in if h_dim is None else h_dim
        self.cell = nn.GRUCell(d_in + C, h_dim)
        self.read = nn.Linear(h_dim, C)
        self.T_out = T_out
        self.C = C
        self.use_kv = use_kv
        if use_kv:
            self.k = nn.Linear(d_in, d_in, bias=False)
            self.v = nn.Linear(d_in, d_in, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d_in))))
        self.attn_drop = nn.Dropout(p=attn_p) if attn_p and attn_p > 0 else nn.Identity()

    def _attend(self, q, K, V):
        scale = torch.exp(self.log_tau)
        scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) * scale
        alpha = torch.softmax(scores, dim=-1)
        alpha = self.attn_drop(alpha)
        ctx = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)
        return ctx, alpha

    def forward(self, Z_seq, x_last=None, teacher=None, sched_sampling_p: float = 0.0,
                A_batch=None, return_alpha: bool = False):
        B, N, T, d = Z_seq.shape
        BN = B * N
        M = Z_seq.view(BN, T, d)
        K = self.k(M) if self.use_kv else M
        V = self.v(M) if self.use_kv else M
        h = M[:, -1, :]
        y_prev = (torch.zeros(BN, self.C, device=Z_seq.device)
                  if x_last is None else x_last.reshape(BN, self.C))

        outs, alphas = [], []
        for t in range(self.T_out):
            if A_batch is not None:
                y_prev_bnc = y_prev.view(B, N, self.C)
                y_prev_bnc = y_prev_bnc + 0.3 * torch.einsum('bij,bjc->bic', A_batch, y_prev_bnc)
                y_prev = y_prev_bnc.reshape(BN, self.C)

            ctx, alpha_t = self._attend(h, K, V)
            inp = torch.cat([ctx, y_prev], dim=-1)
            h = self.cell(inp, h)
            y_t = self.read(h) + y_prev
            outs.append(y_t.view(B, N, self.C, 1))
            if return_alpha:
                alphas.append(alpha_t.view(B, N, T))

            # Scheduled sampling
            if self.training and teacher is not None and sched_sampling_p > 0.0:
                use_pred = (torch.rand(BN, 1, device=Z_seq.device) < sched_sampling_p).float()
                teach_t = teacher[..., t].reshape(BN, self.C)
                y_prev = use_pred * y_t + (1.0 - use_pred) * teach_t
            else:
                y_prev = y_t

        Y = torch.cat(outs, dim=-1)
        return (Y, torch.stack(alphas, dim=3)) if return_alpha else Y


# =============================================================================
# 6. FULL MODEL: PS_APM_Seq
# =============================================================================

class PS_APM_Seq(nn.Module):
    """
    Full Phase-Specific Adaptive Projection Model.
    Combines:
      - Per-region GRU temporal encoder
      - Time-positional embedding
      - Phase-specific graph learner
      - Graph projection layer
      - Autoregressive attention-based decoder
    """
    def __init__(self, N, C, cfg):
        super().__init__()
        self.N, self.C = N, C
        self.enc = PerRegionGRU(N, C, cfg.d_hidden)
        self.tpos = TimePositional(cfg.d_timectx, cfg.T_in)
        self.graphs = PhaseGraphs(N, P=4, use_row_gain=cfg.use_row_gain)
        self.proj = GraphProjector(cfg.d_hidden + cfg.d_timectx, cfg.d_proj)
        self.head = ARHead(cfg.d_proj, C, cfg.T_out, use_kv=True, attn_p=0.10)

    def forward(self, X_in, phases, teacher=None, sched_p: float = 0.0,
                return_alpha: bool = False, use_neigh=True):
        B, N, C, T = X_in.shape
        H = self.enc(X_in)
        u = self.tpos(B, T).unsqueeze(1).repeat(1, N, 1, 1)
        Hc = torch.cat([H, u], dim=-1)
        A = self.graphs(phases)
        Z = self.proj(Hc, A)
        x_last = X_in[:, :, :, -1]
        A_for_roll = A if use_neigh else None

        Y_delta = self.head(Z, x_last=None, teacher=teacher,
                            sched_sampling_p=sched_p, A_batch=A_for_roll,
                            return_alpha=return_alpha)

        if isinstance(Y_delta, tuple):
            Y_delta_main, alphas = Y_delta
        else:
            Y_delta_main, alphas = Y_delta, None

        Y_hat = x_last.unsqueeze(-1) + Y_delta_main
        y1 = Y_hat[:, :, :, 0:1]
        return (Y_hat, y1) if alphas is None else ((Y_hat, alphas), y1)
