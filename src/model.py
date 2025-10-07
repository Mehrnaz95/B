# -*- coding: utf-8 -*-
"""
model.py
--------
Core architecture shared across all experiments (real + synthetic).

Modules:
    • PerRegionGRU      – encodes each region’s time series separately.
    • TimePositional    – adds a temporal embedding to preserve ordering.
    • PhaseGraphs       – learns one signed, phase-specific adjacency matrix.
    • GraphProjector    – graph message-passing layer with spectral regularization.
    • ARHead            – autoregressive decoder with attention over temporal memory.
    • PS_APM_Seq        – full model combining all components.

Used by:
    - main_real.py
    - main_synth_structured.py
    - main_synth_stochastic.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ==============================================================
#  Per-region GRU encoder
# ==============================================================

class PerRegionGRU(nn.Module):
    """
    One GRU per brain region (or synthetic node).
    Encodes intra-region temporal features independently.

    Input:
        x : [B, N, C, T]
            B = batch, N = regions, C = channels per region, T = time points
    Output:
        h : [B, N, T, d_hidden]
    """
    def __init__(self, N: int, C: int, d_hidden: int):
        super().__init__()
        self.N = N
        self.grus = nn.ModuleList([
            nn.GRU(input_size=C, hidden_size=d_hidden, batch_first=False)
            for _ in range(N)
        ])
        self.ln = nn.LayerNorm(d_hidden)

    def forward(self, x):
        B, N, C, T = x.shape
        outs = []
        for r in range(N):
            # [B,C,T] → [T,B,C] for GRU
            xr = x[:, r].permute(2, 0, 1)
            y, _ = self.grus[r](xr)
            y = self.ln(y)
            outs.append(y.permute(1, 0, 2))  # [B,T,d_hidden]
        return torch.stack(outs, dim=1)       # [B,N,T,d_hidden]


# ==============================================================
#  Temporal positional embedding
# ==============================================================

class TimePositional(nn.Module):
    """Simple learnable linear time embedding."""
    def __init__(self, d_time: int, T_in: int, device=None):
        super().__init__()
        self.emb = nn.Linear(1, d_time)
        self.T_in = T_in
        self.device = device

    def forward(self, B: int, T: int):
        t = torch.linspace(0., 1., steps=T, device=self.device).view(1, T, 1)
        t = t.repeat(B, 1, 1)
        return torch.tanh(self.emb(t))  # [B,T,d_time]


# ==============================================================
#  Phase-specific graph learner
# ==============================================================

class PhaseGraphs(nn.Module):
    """
    Learns one signed adjacency per phase.
    Optionally applies per-row gain to control edge magnitudes.
    """
    def __init__(self, N: int, P: int = 4, use_row_gain: bool = True, eps: float = 1e-6):
        super().__init__()
        self.N, self.P, self.eps = N, P, eps
        self.S = nn.Parameter(torch.zeros(P, N, N))   # signed, trainable
        self.register_buffer("I", torch.eye(N))
        self.use_row_gain = use_row_gain
        if use_row_gain:
            self.G = nn.Parameter(torch.zeros(P, N))  # row gains

    # ----- helpers -----
    def _zero_diag(self, X):
        return X * (1.0 - self.I)

    def _row_norm_l1_signed(self, S):
        S = self._zero_diag(S)
        denom = S.abs().sum(-1, keepdim=True).clamp_min(self.eps)
        return S / denom

    # ----- forward -----
    def forward(self, phases: torch.Tensor):
        A = self._row_norm_l1_signed(self.S)[phases]          # [B,N,N]
        if self.use_row_gain:
            g = F.softplus(self.G) + 1e-6                     # [P,N]
            g = g * (self.N / g.sum(-1, keepdim=True).clamp_min(self.eps))
            A = A * g[phases].unsqueeze(-1)
        return A

    # ----- exports -----
    @torch.no_grad()
    def export_pattern(self):
        return self._row_norm_l1_signed(self.S).cpu().numpy()

    @torch.no_grad()
    def export_effective(self):
        A = self._row_norm_l1_signed(self.S)
        if self.use_row_gain:
            g = F.softplus(self.G) + 1e-6
            g = g * (self.N / g.sum(-1, keepdim=True).clamp_min(self.eps))
            A = A * g[..., None]
        return A.detach().cpu().numpy()

    @torch.no_grad()
    def init_from_correlation(self, C_list):
        """Initialize adjacency weights from empirical region correlations."""
        for p in range(self.P):
            M = torch.tensor(C_list[p], dtype=torch.float32)
            M.fill_diagonal_(0.0)
            self.S[p].copy_(M)
        if self.use_row_gain:
            self.G.zero_()


# ==============================================================
#  Graph projector (message passing)
# ==============================================================

class GraphProjector(nn.Module):
    """
    Applies message passing with spectral-normalized neighbor mixing.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_self = nn.Linear(d_in, d_out, bias=False)
        self.W_neigh = spectral_norm(nn.Linear(d_in, d_out, bias=False))

    def forward(self, H_seq, A_batch):
        """
        H_seq: [B,N,T,d_in]
        A_batch: [B,N,N]
        """
        Hs = self.W_self(H_seq)
        Hn = self.W_neigh(H_seq)
        Zn = torch.einsum("bij,bjtd->bitd", A_batch, Hn)
        return F.leaky_relu(Hs + Zn, 0.1)


# ==============================================================
#  Autoregressive attention decoder
# ==============================================================

class ARHead(nn.Module):
    """
    Decodes future sequence using attention over encoder memory.
    Supports optional neighbor feedback via A_batch.

    Args:
        d_in : latent dimension
        C    : output channels per node
        T_out: prediction horizon
    """
    def __init__(self, d_in, C, T_out, h_dim=None, use_kv=True, attn_p=0.1):
        super().__init__()
        h_dim = d_in if h_dim is None else h_dim
        self.cell = nn.GRUCell(d_in + C, h_dim)
        self.read = nn.Linear(h_dim, C)
        self.T_out, self.C = T_out, C
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

    def forward(self, Z_seq, x_last=None, teacher=None,
                sched_sampling_p=0.0, A_batch=None, return_alpha=False):
        B, N, T, d = Z_seq.shape
        BN = B * N
        M = Z_seq.view(BN, T, d)
        K = self.k(M) if self.use_kv else M
        V = self.v(M) if self.use_kv else M

        h = M[:, -1, :]
        y_prev = torch.zeros(BN, self.C, device=Z_seq.device) if x_last is None \
                 else x_last.reshape(BN, self.C)

        outs, alphas = [], []
        for _ in range(self.T_out):
            # optional graph feedback
            if A_batch is not None:
                y_prev_bnc = y_prev.view(B, N, self.C)
                y_prev_bnc = y_prev_bnc + 0.3 * torch.einsum("bij,bjc->bic", A_batch, y_prev_bnc)
                y_prev = y_prev_bnc.reshape(BN, self.C)

            ctx, alpha_t = self._attend(h, K, V)
            inp = torch.cat([ctx, y_prev], dim=-1)
            h = self.cell(inp, h)
            y_t = self.read(h)
            y_t = y_t + y_prev  # residual decoding

            outs.append(y_t.view(B, N, self.C, 1))
            if return_alpha:
                alphas.append(alpha_t.view(B, N, T))

            # scheduled sampling (optional)
            if self.training and teacher is not None and sched_sampling_p > 0.0:
                use_pred = (torch.rand(BN, 1, device=Z_seq.device) < sched_sampling_p).float()
                teach_t = teacher[..., _].reshape(BN, self.C)
                y_prev = use_pred * y_t + (1.0 - use_pred) * teach_t
            else:
                y_prev = y_t

        Y = torch.cat(outs, dim=-1)
        if return_alpha:
            return Y, torch.stack(alphas, dim=3)
        return Y


# ==============================================================
#  Complete model
# ==============================================================

class PS_APM_Seq(nn.Module):
    """
    Phase-Specific Adaptive Projection Model (BACE variant).

    Pipeline:
        1. PerRegionGRU → temporal encoding per node
        2. TimePositional → add explicit time context
        3. PhaseGraphs → get phase-specific adjacency A
        4. GraphProjector → combine temporal & spatial info
        5. ARHead → autoregressively decode next T_out samples
    """
    def __init__(self, N, C, cfg):
        super().__init__()
        self.N, self.C = N, C
        self.enc = PerRegionGRU(N, C, cfg.d_hidden)
        self.tpos = TimePositional(cfg.d_timectx, cfg.T_in, device=cfg.device)
        self.graphs = PhaseGraphs(N, P=4, use_row_gain=cfg.use_row_gain)
        self.proj = GraphProjector(cfg.d_hidden + cfg.d_timectx, cfg.d_proj)
        self.head = ARHead(cfg.d_proj, C, cfg.T_out, use_kv=True, attn_p=0.1)

    def forward(self, X_in, phases, teacher=None, sched_p=0.0,
                return_alpha=False, use_neigh=True):
        B, N, C, T = X_in.shape
        H = self.enc(X_in)
        u = self.tpos(B, T).unsqueeze(1).repeat(1, N, 1, 1)
        Hc = torch.cat([H, u], dim=-1)

        A = self.graphs(phases)
        Z = self.proj(Hc, A)
        x_last = X_in[..., -1]

        Y_delta = self.head(Z, x_last=None, teacher=teacher,
                            sched_sampling_p=sched_p,
                            A_batch=A if use_neigh else None,
                            return_alpha=return_alpha)

        if isinstance(Y_delta, tuple):
            Y_delta_main, alphas = Y_delta
        else:
            Y_delta_main, alphas = Y_delta, None

        Y_hat = x_last.unsqueeze(-1) + Y_delta_main
        y1 = Y_hat[..., 0:1]
        return (Y_hat, y1) if alphas is None else ((Y_hat, alphas), y1)


# ==============================================================
#  Regularization helper
# ==============================================================

def L1_on_S(graphs: PhaseGraphs):
    """L1 penalty on off-diagonal elements of signed adjacency."""
    S = graphs.S
    I = torch.eye(S.size(-1), device=S.device)
    return (S.abs() * (1.0 - I)).sum()
