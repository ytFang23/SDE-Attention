"""
SeqLink Cross-Sample Pyramidal Attention
Based on: "SeqLink: A Robust Neural-ODE Architecture for Modelling Partially Observed Time Series"
Abushaqra et al., Transactions on Machine Learning Research (07/2024)

=== What matches the paper exactly ===
  - Eq. 6-7  : two separate embedding networks phi_x and phi_u for different signals
  - Eq. 8-9  : concat -> score -> softmax importance weights across samples
  - Algorithm 1: iterative mean-split (MeanV = sum(rates)/K) pyramidal sorting
  - Eq. 10   : level weights increase toward the apex (most important = highest weight)
  - Eq. 11   : cross-sample representation fused into each sample's hidden state

=== Intentional departures (unavoidable given end-to-end ODE-RNN setup) ===
  1. Cross-sample scope: paper uses full dataset K (pre-computed by a separate ODE
     auto-encoder). Here we use the current mini-batch B. Effect: less diversity per
     forward pass; mitigated by shuffled batching across epochs.
  2. Integration point: paper's Link-ODE injects p into the RNNCell at every timestep.
     Here we apply a single post-encoder fusion on the full latent_ys trajectory.
  3. phi_x input: paper feeds raw observation x; here we use latent_ys[:, 0, :]
     (first latent state, closest proxy to 'what was observed'). phi_u gets mean latent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqLinkAttention(nn.Module):
    """
    Cross-sample pyramidal attention for ODE-RNN.
    """

    def __init__(self, latent_dim: int, num_levels: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_levels = num_levels

        # phi_x: embeds "observation-side" signal (Eq. 6) — first latent state as proxy
        self.embed_x = nn.Linear(latent_dim, hidden_dim)
        # phi_u: embeds "learned representation" signal (Eq. 7) — temporal mean of latent
        self.embed_u = nn.Linear(latent_dim, hidden_dim)

        # Scoring: (e_x concat e_u) . theta  (Eq. 8)
        self.score_net = nn.Linear(hidden_dim * 2, 1)

        # Level weights w: fixed, linearly increasing from base to apex (Eq. 10).
        # Registered as buffer (not learnable) to match the paper.
        w = torch.linspace(1.0, float(num_levels), num_levels)
        w = w / w.sum()
        self.register_buffer("level_weights", w)   # (num_levels,)

        # Fusion (Eq. 11)
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def _attention_scores(self, x_repr, u_repr):
        """Eq. 8-9: returns (B, B) softmax importance matrix."""
        B = x_repr.shape[0]
        ex = self.embed_x(x_repr)                           # (B, H)
        eu = self.embed_u(u_repr)                           # (B, H)
        ex_exp = ex.unsqueeze(1).expand(B, B, -1)
        eu_exp = eu.unsqueeze(0).expand(B, B, -1)
        S = self.score_net(torch.cat([ex_exp, eu_exp], dim=-1)).squeeze(-1)  # (B, B)
        diag_mask = torch.eye(B, dtype=torch.bool, device=S.device)
        S = S.masked_fill(diag_mask, float("-inf"))
        return F.softmax(S, dim=-1)                         # (B, B)

    def _pyramidal_sort(self, representations, weights):
        """
        Algorithm 1 from SeqLink (faithful translation):
          MeanV = sum(rates) / K   [paper line 7 — mean, not median]
          lower half -> current level, upper half -> next level
        Returns (B, num_levels, D)
        """
        B, D = representations.shape
        device = representations.device
        level_reprs = torch.zeros(B, self.num_levels, D, device=device)

        for b in range(B):
            other_mask = torch.ones(B, dtype=torch.bool, device=device)
            other_mask[b] = False
            alpha = weights[b][other_mask].clone()    # (B-1,)
            reprs = representations[other_mask]       # (B-1, D)

            if alpha.shape[0] == 0:
                continue

            active = torch.ones(alpha.shape[0], dtype=torch.bool, device=device)

            for l in range(self.num_levels - 1):
                if not active.any():
                    continue
                # Algorithm 1 line 7: MeanV = sum(rates) / K
                mean_val = alpha[active].sum() / active.float().sum()
                lower = active & (alpha <= mean_val)
                if lower.any():
                    level_reprs[b, l] = reprs[lower].mean(0)
                active = active & ~lower    # upper half moves to next level

            # Apex: remaining (highest importance)
            if active.any():
                level_reprs[b, self.num_levels - 1] = reprs[active].mean(0)
            else:
                top_idx = weights[b][other_mask].argmax()
                level_reprs[b, self.num_levels - 1] = reprs[top_idx]

        return level_reprs    # (B, num_levels, D)

    def forward(self, latent_ys: torch.Tensor) -> torch.Tensor:
        """
        Args:  latent_ys (B, T, D)
        Returns: enhanced (B, T, D)
        """
        B, T, D = latent_ys.shape
        if B < 2:
            return latent_ys

        x_repr = latent_ys[:, 0, :]           # first latent state ~ "x" (Eq. 6)
        u_repr = latent_ys.mean(dim=1)         # temporal mean ~ "u" (Eq. 7)

        weights = self._attention_scores(x_repr, u_repr)          # (B, B)  Eq.8-9
        level_reprs = self._pyramidal_sort(u_repr, weights)       # (B, L, D)  Alg.1
        cross_repr = (level_reprs * self.level_weights.view(1, -1, 1)).sum(1)  # (B,D) Eq.10
        cross_expanded = cross_repr.unsqueeze(1).expand(B, T, D)
        fused = self.fusion(torch.cat([latent_ys, cross_expanded], dim=-1))    # Eq.11
        return self.norm(fused + latent_ys)