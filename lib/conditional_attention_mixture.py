"""
Conditional Attention Mixture - ENHANCED VERSION (FIXED)

Key improvements to fix underperformance:
1. ✅ Rich feature extraction (17 statistics instead of 7)
2. ✅ Direct routing from hidden states (no information bottleneck)
3. ✅ Expert diversity loss
4. ✅ Better normalization
5. ✅ Two-stage training support

FIXED: Dimension mismatch errors in feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from lib.attention_mechanisms import ChannelAttention, TimeVaryingFeatureAttention
from lib.pyramidal_attention import PyramidalAttention


class EnhancedStatistics(nn.Module):
    """
    Compute rich time series features for routing.

    Features include:
    - Temporal: autocorrelation, trend
    - Frequency: dominant freq, spectral entropy
    - Statistical: skewness, kurtosis, quantiles
    - Shape: peak count, zero crossings, change rate
    - Missing data: per-sample missing rate, density variance
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T, D) optional

        Returns:
            stats: (B, num_features) - rich feature vector
        """
        B, T, D = x.shape
        features = []

        # 1. Temporal features
        features.append(self._temporal_features(x))  # 2 features

        # 2. Frequency features
        features.append(self._frequency_features(x))  # 2 features

        # 3. Statistical features
        features.append(self._statistical_features(x))  # 5 features

        # 4. Shape features
        features.append(self._shape_features(x))  # 3 features

        # 5. Missing data features (if mask provided)
        if mask is not None:
            features.append(self._missing_features(x, mask))  # 3 features
        else:
            features.append(torch.zeros(B, 3, device=x.device))

        # 6. Channel features (if multivariate)
        if D > 1:
            features.append(self._channel_features(x))  # 2 features
        else:
            features.append(torch.zeros(B, 2, device=x.device))

        # Total: 2+2+5+3+3+2 = 17 features
        stats = torch.cat(features, dim=1)

        return stats

    def _temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Autocorrelation and trend"""
        B, T, D = x.shape

        # Autocorrelation at lag 1
        if T > 1:
            x_t = x[:, :-1, :]
            x_t1 = x[:, 1:, :]
            autocorr = (x_t * x_t1).mean(dim=(1, 2), keepdim=True).squeeze(-1)  # ✅ FIXED: (B, 1)
        else:
            autocorr = torch.zeros(B, 1, device=x.device)

        # Linear trend (slope)
        t = torch.arange(T, dtype=x.dtype, device=x.device)
        t_mean = t.mean()
        x_mean_t = x.mean(dim=2)  # (B, T)
        x_mean_overall = x_mean_t.mean(dim=1, keepdim=True)

        numerator = ((t - t_mean) * (x_mean_t - x_mean_overall)).sum(dim=1, keepdim=True)
        denominator = ((t - t_mean) ** 2).sum() + self.eps
        trend = numerator / denominator  # (B, 1)

        return torch.cat([autocorr, trend], dim=1)  # (B, 2)

    def _frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """Dominant frequency and spectral entropy"""
        B, T, D = x.shape

        # Average across channels
        x_mean = x.mean(dim=2)  # (B, T)

        # FFT
        if T > 2:
            fft = torch.fft.rfft(x_mean, dim=1)
            power = torch.abs(fft) ** 2

            # Dominant frequency (normalized)
            dom_freq_idx = power.argmax(dim=1).float() / (T / 2)
            dom_freq_idx = dom_freq_idx.unsqueeze(1)  # (B, 1)

            # Spectral entropy
            power_norm = power / (power.sum(dim=1, keepdim=True) + self.eps)
            spec_entropy = -(power_norm * torch.log(power_norm + self.eps)).sum(dim=1, keepdim=True)  # (B, 1)
        else:
            dom_freq_idx = torch.zeros(B, 1, device=x.device)
            spec_entropy = torch.zeros(B, 1, device=x.device)

        return torch.cat([dom_freq_idx, spec_entropy], dim=1)  # (B, 2)

    def _statistical_features(self, x: torch.Tensor) -> torch.Tensor:
        """Higher-order moments and quantiles"""
        B, T, D = x.shape

        # Centered data
        x_centered = x - x.mean(dim=(1, 2), keepdim=True)

        # Moments
        m2 = (x_centered ** 2).mean(dim=(1, 2))  # (B,)
        m3 = (x_centered ** 3).mean(dim=(1, 2))  # (B,)
        m4 = (x_centered ** 4).mean(dim=(1, 2))  # (B,)

        # Skewness
        skewness = m3 / (m2 ** 1.5 + self.eps)  # (B,)

        # Kurtosis
        kurtosis = m4 / (m2 ** 2 + self.eps)  # (B,)

        # Quantiles
        x_flat = x.reshape(B, -1)
        q25 = torch.quantile(x_flat, 0.25, dim=1)  # (B,)
        q50 = torch.quantile(x_flat, 0.50, dim=1)  # (B,)
        q75 = torch.quantile(x_flat, 0.75, dim=1)  # (B,)

        return torch.stack([skewness, kurtosis, q25, q50, q75], dim=1)  # (B, 5)

    def _shape_features(self, x: torch.Tensor) -> torch.Tensor:
        """Peak count, zero crossings, rate of change"""
        B, T, D = x.shape

        x_mean = x.mean(dim=2)  # (B, T)

        # Peak count (local maxima/minima)
        if T > 2:
            diff = x_mean[:, 1:] - x_mean[:, :-1]
            sign_change = (diff[:, 1:] * diff[:, :-1]) < 0
            peak_count = sign_change.float().sum(dim=1, keepdim=True) / (T - 2)  # (B, 1)
        else:
            peak_count = torch.zeros(B, 1, device=x.device)

        # Zero crossing rate
        if T > 1:
            x_centered = x_mean - x_mean.mean(dim=1, keepdim=True)
            zero_cross = ((x_centered[:, 1:] * x_centered[:, :-1]) < 0).float().sum(dim=1, keepdim=True) / (T - 1)  # (B, 1)
        else:
            zero_cross = torch.zeros(B, 1, device=x.device)

        # Rate of change (mean absolute difference)
        if T > 1:
            roc = torch.abs(x_mean[:, 1:] - x_mean[:, :-1]).mean(dim=1, keepdim=True)  # (B, 1)
        else:
            roc = torch.zeros(B, 1, device=x.device)

        return torch.cat([peak_count, zero_cross, roc], dim=1)  # (B, 3)

    def _missing_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Missing data patterns"""
        B, T, D = x.shape

        # Per-sample missing rate
        per_sample_observed = mask.reshape(B, -1).sum(dim=1, keepdim=True)
        per_sample_total = T * D
        missing_rate = 1.0 - (per_sample_observed / (per_sample_total + self.eps))  # (B, 1)

        # Observation density variance (temporal)
        obs_per_time = mask.sum(dim=2).float()  # (B, T)
        obs_density = obs_per_time / (D + self.eps)
        density_var = obs_density.var(dim=1, keepdim=True)  # (B, 1)

        # Coverage (span of observations)
        time_mask = mask.sum(dim=2) > 0  # (B, T)
        first_obs = torch.zeros(B, 1, device=x.device)

        for b in range(B):
            obs_indices = torch.where(time_mask[b])[0]
            if len(obs_indices) > 1:
                first_obs[b] = obs_indices[0].float() / T
            elif len(obs_indices) == 1:
                first_obs[b] = obs_indices[0].float() / T

        return torch.cat([missing_rate, density_var, first_obs], dim=1)  # (B, 3)

    def _channel_features(self, x: torch.Tensor) -> torch.Tensor:
        """Channel correlations"""
        B, T, D = x.shape

        if D > 1:
            # Mean correlation between channels
            x_centered = x - x.mean(dim=1, keepdim=True)
            cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (T - 1 + self.eps)
            std = x.std(dim=1, keepdim=True).transpose(1, 2)
            corr = cov / (std * std.transpose(1, 2) + self.eps)

            # Off-diagonal correlations
            mask_eye = 1 - torch.eye(D, device=x.device)
            mean_corr = (corr * mask_eye).sum(dim=(1, 2)) / (D * (D - 1) + self.eps)
            mean_corr = mean_corr.unsqueeze(1)  # (B, 1)

            # Variance explained by first PC
            try:
                _, S, _ = torch.svd(x_centered)
                var_explained = (S[:, 0] ** 2) / (S ** 2).sum(dim=1)
                var_explained = var_explained.unsqueeze(1)  # (B, 1)
            except:
                # SVD might fail, use fallback
                var_explained = torch.ones(B, 1, device=x.device)
        else:
            mean_corr = torch.zeros(B, 1, device=x.device)
            var_explained = torch.ones(B, 1, device=x.device)

        return torch.cat([mean_corr, var_explained], dim=1)  # (B, 2)


class DirectHiddenRouting(nn.Module):
    """
    Route directly from hidden states using attention pooling.

    This avoids the information bottleneck of hand-crafted statistics.
    """

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 32,
                 dropout: float = 0.2):
        super().__init__()

        self.num_experts = num_experts

        # Attention pooling: (B, T, D) → (B, D)
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Routing network
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

        self.temperature = nn.Parameter(torch.ones(1))

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, T, D) - hidden states

        Returns:
            alpha: (B, M) - routing weights
            h_pooled: (B, D) - pooled hidden state
        """
        B, T, D = h.shape

        # Attention pooling
        attn_logits = self.attention_pool(h)  # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_pooled = (h * attn_weights).sum(dim=1)  # (B, D)

        # Route
        logits = self.router(h_pooled)
        temp = self.temperature.clamp(min=0.5, max=2.0)  # Tighter range
        alpha = F.softmax(logits / temp, dim=-1)

        return alpha, h_pooled


class ConditionalAttentionMixture(nn.Module):
    """
    Conditional Attention Mixture with three fixes:

    1. Entropy regularisation sign corrected  → maximises entropy (soft routing)
    2. Diversity loss replaced by lightweight balance loss  → prevents collapse
       without suppressing TVF's natural advantage
    3. Dynamic expert gradient pruning  → when one expert consistently dominates
       (EMA weight > dominance_threshold), the other two experts are detached
       from the gradient graph, saving compute and focusing learning
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_attention_levels: int = 3,
        tvf_method: str = 'lstm',
        lambda_entropy: float = 0.01,
        lambda_balance: float = 0.001,      # replaces lambda_diversity; kept small
        use_enhanced_stats: bool = True,
        use_direct_routing: bool = True,
        # --- dynamic pruning ---
        dominance_threshold: float = 0.75,  # EMA weight above this → cut others
        ema_momentum: float = 0.99,         # slow EMA to avoid jitter
        dominance_warmup: int = 300,        # forward steps; set to 30*n_train_batches in run_models.py
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()

        self.input_dim = input_dim
        self.device = device
        self.lambda_entropy = lambda_entropy
        self.lambda_balance = lambda_balance
        self.use_enhanced_stats = use_enhanced_stats
        self.use_direct_routing = use_direct_routing

        # --- dynamic pruning state ---
        self.dominance_threshold = dominance_threshold
        self.ema_momentum = ema_momentum
        self.dominance_warmup = dominance_warmup
        # alpha_ema: running mean of routing weights (non-parameter buffer)
        self.register_buffer(
            'alpha_ema',
            torch.ones(3) / 3  # start uniform
        )
        self.register_buffer(
            '_fwd_count',
            torch.tensor(0, dtype=torch.long)
        )

        # Attention Experts
        self.experts = nn.ModuleList([
            ChannelAttention(input_dim=input_dim, hidden_dim=hidden_dim).to(device),
            TimeVaryingFeatureAttention(input_dim=input_dim, hidden_dim=hidden_dim,
                                       method=tvf_method).to(device),
            PyramidalAttention(input_dim=input_dim, num_levels=num_attention_levels,
                              hidden_dim=hidden_dim).to(device),
        ])
        self.num_experts = len(self.experts)

        # stats_computer is ALWAYS instantiated so statistics are available
        # for post-hoc correlation analysis even when use_enhanced_stats=False.
        # When disabled it runs inside torch.no_grad() and never touches gradients.
        self.stats_computer = EnhancedStatistics()

        # Routing networks
        if use_enhanced_stats and use_direct_routing:
            self.stats_router = nn.Sequential(
                nn.Linear(17, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, self.num_experts)
            ).to(device)
            self.hidden_router = DirectHiddenRouting(
                input_dim=input_dim,
                num_experts=self.num_experts,
                hidden_dim=hidden_dim
            ).to(device)
            self.alpha_combine = nn.Parameter(torch.tensor(0.5))

        elif use_direct_routing:
            # Stream 1 disabled: only latent-state routing drives alpha.
            # stats_computer still runs under no_grad to populate the info dict.
            self.hidden_router = DirectHiddenRouting(
                input_dim=input_dim,
                num_experts=self.num_experts,
                hidden_dim=hidden_dim
            ).to(device)

        elif use_enhanced_stats:
            self.stats_router = nn.Sequential(
                nn.Linear(17, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, self.num_experts)
            ).to(device)
            self.temperature = nn.Parameter(torch.ones(1))

        print(f"[ConditionalAttentionMixture]")
        print(f"  - Enhanced stats: {use_enhanced_stats}")
        print(f"  - Direct routing: {use_direct_routing}")
        print(f"  - λ_entropy={lambda_entropy}  λ_balance={lambda_balance}")
        print(f"  - Dominance pruning: threshold={dominance_threshold}, warmup={dominance_warmup} steps")

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _compute_alpha(
        self,
        h_stats: torch.Tensor,
        x_input: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute routing weights α ∈ (0,1)^M.

        Statistical features are ALWAYS computed and returned for logging /
        post-hoc correlation analysis.  They only influence α when
        use_enhanced_stats=True.  When disabled the computation runs inside
        torch.no_grad() so it never affects gradients or training.

        Returns:
            alpha: (B, M) routing weights
            stats: (B, 17) statistical features  ← always present now
        """
        # ── compute stats (always, gradient only when use_enhanced_stats=True) ──
        src = x_input if x_input is not None else h_stats
        if self.use_enhanced_stats:
            stats = self.stats_computer(src, mask)          # in graph
        else:
            with torch.no_grad():
                stats = self.stats_computer(src, mask)      # observation only, no grad

        # ── compute alpha ──────────────────────────────────────────────────────
        if self.use_enhanced_stats and self.use_direct_routing:
            stats_alpha = F.softmax(self.stats_router(stats), dim=-1)
            hidden_alpha, _ = self.hidden_router(h_stats)
            w = torch.sigmoid(self.alpha_combine)
            alpha = w * hidden_alpha + (1 - w) * stats_alpha

        elif self.use_direct_routing:
            # Stream 1 disabled: pure latent routing
            alpha, _ = self.hidden_router(h_stats)

        else:  # stats only
            logits = self.stats_router(stats)
            temp = self.temperature.clamp(min=0.5, max=2.0)
            alpha = F.softmax(logits / temp, dim=-1)

        return alpha, stats

    def _update_ema(self, alpha: torch.Tensor) -> None:
        """Update the running EMA of batch-mean routing weights (no grad)."""
        with torch.no_grad():
            batch_mean = alpha.mean(dim=0).to(self.alpha_ema.device)
            self.alpha_ema.mul_(self.ema_momentum).add_(
                batch_mean * (1.0 - self.ema_momentum)
            )
            self._fwd_count.add_(1)

    def _grad_mask(self) -> List[bool]:
        """
        Return a per-expert bool list: True → keep gradient, False → detach.

        Pruning only activates after warmup AND when one expert's EMA weight
        exceeds the dominance threshold.  During eval the mask is always all-True
        so inference is unaffected.
        """
        if not self.training:
            return [True] * self.num_experts

        if self._fwd_count.item() < self.dominance_warmup:
            return [True] * self.num_experts

        dominant = int(self.alpha_ema.argmax().item())
        if self.alpha_ema[dominant].item() >= self.dominance_threshold:
            mask = [False] * self.num_experts
            mask[dominant] = True
            return mask

        return [True] * self.num_experts

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        x_input: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_routing_loss: bool = True
    ) -> Tuple[torch.Tensor, Dict]:

        # --- shape handling ---
        if h.dim() == 4:
            S, B, T, D = h.shape
            h_stats = h[0]
        elif h.dim() == 3:
            B, T, D = h.shape
            S = 1
            h_stats = h
        else:
            raise ValueError(f"Expected h to be 3D or 4D, got {h.dim()}D")

        # --- routing ---
        alpha, stats = self._compute_alpha(h_stats, x_input, mask)

        # update EMA during training (no grad needed)
        if self.training:
            self._update_ema(alpha)

        # --- dynamic gradient mask ---
        grad_mask = self._grad_mask()
        active_experts = [i for i, keep in enumerate(grad_mask) if keep]
        if len(active_experts) < self.num_experts and self.training:
            expert_names = self.get_expert_names()
            dominant_name = expert_names[active_experts[0]]
            # log once every 200 steps to avoid spam
            if self._fwd_count.item() % 200 == 0:
                print(f"[CAM] Pruning inactive experts – dominant: {dominant_name} "
                      f"(EMA={self.alpha_ema[active_experts[0]]:.3f})")

        # --- apply experts with selective gradient ---
        if h.dim() == 4:
            h_reshaped = h.reshape(S * B, T, D)
            expert_outputs = []
            expert_weights_list = []
            for i, expert in enumerate(self.experts):
                h_i_flat, weights = expert(h_reshaped)
                h_i = h_i_flat.reshape(S, B, T, D)
                if not grad_mask[i]:
                    h_i = h_i.detach()  # cut gradient for non-dominant experts
                expert_outputs.append(h_i)
                expert_weights_list.append(weights)
        else:
            expert_outputs = []
            expert_weights_list = []
            for i, expert in enumerate(self.experts):
                h_i, weights = expert(h)
                if not grad_mask[i]:
                    h_i = h_i.detach()
                expert_outputs.append(h_i)
                expert_weights_list.append(weights)

        # --- weighted mixture ---
        if h.dim() == 4:
            expert_stack = torch.stack(expert_outputs, dim=0).permute(1, 2, 3, 4, 0)
            alpha_expanded = alpha.view(1, B, 1, 1, self.num_experts)
        else:
            expert_stack = torch.stack(expert_outputs, dim=0).permute(1, 2, 3, 0)
            alpha_expanded = alpha.view(B, 1, 1, self.num_experts)
        h_att = (expert_stack * alpha_expanded).sum(dim=-1)

        # --- info dict ---
        info = {
            'mixture_weights': alpha,           # (B, M)  routing weights per sample
            'stats': stats,                     # (B, 17) or None – raw statistical features
            'stats_names': [                    # column names for correlation analysis
                # temporal (2)
                'autocorr_lag1', 'linear_trend',
                # frequency (2)
                'dominant_freq', 'spectral_entropy',
                # distributional (5)
                'skewness', 'kurtosis', 'q25', 'q50', 'q75',
                # shape (3)
                'peak_count', 'zero_crossing_rate', 'rate_of_change',
                # missingness (3)
                'missing_rate', 'obs_density_var', 'first_obs_time',
                # channel dependence (2)
                'mean_channel_corr', 'var_explained_pc1',
            ],
            'expert_outputs': expert_outputs,
            'expert_attention_weights': expert_weights_list,
            'grad_mask': grad_mask,
            'alpha_ema': self.alpha_ema.detach().clone(),
        }

        # --- auxiliary losses ---
        if return_routing_loss:
            entropy_loss = self.compute_entropy_loss(alpha)
            balance_loss = self.compute_balance_loss(alpha)
            total_routing_loss = entropy_loss + balance_loss
            info['routing_loss'] = total_routing_loss
            info['entropy_loss'] = entropy_loss
            info['balance_loss'] = balance_loss

        return h_att, info

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def compute_entropy_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Maximise routing entropy → encourage soft allocation early in training.

        FIX: original code minimised entropy (sign was wrong).
        H(α) = -∑ α_i log α_i  (always ≥ 0)
        We want to maximise H, so the loss contribution is -λ * H.
        """
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()  # H ≥ 0
        return -self.lambda_entropy * entropy   # add to total loss → maximises H

    def compute_balance_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Penalise batch-level expert imbalance.

        L_balance = ∑_i (ᾱ_i - 1/M)²
        Kept very small (lambda_balance=0.001) so it only acts as a
        last-resort safeguard against collapse, not as a hard constraint.
        """
        alpha_bar = alpha.mean(dim=0)               # (M,) batch-mean weights
        uniform = torch.full_like(alpha_bar, 1.0 / self.num_experts)
        return self.lambda_balance * ((alpha_bar - uniform) ** 2).sum()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_expert_names(self) -> List[str]:
        return ["Channel Attention", "TVF Attention", "Pyramidal Attention"]

    def analyze_routing(self, alpha: torch.Tensor) -> Dict:
        expert_names = self.get_expert_names()
        avg_weights = alpha.mean(dim=0).cpu().numpy()
        most_selected = alpha.argmax(dim=1)
        selection_counts = torch.bincount(most_selected, minlength=self.num_experts)
        selection_freq = selection_counts.float() / alpha.size(0)
        entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean().item()

        analysis = {
            'average_weights': {expert_names[i]: float(avg_weights[i])
                                for i in range(self.num_experts)},
            'selection_frequency': {expert_names[i]: float(selection_freq[i])
                                    for i in range(self.num_experts)},
            'routing_entropy': entropy,
            'alpha_ema': {expert_names[i]: float(self.alpha_ema[i].item())
                          for i in range(self.num_experts)},
            'dominant_expert': expert_names[int(self.alpha_ema.argmax().item())],
            'pruning_active': any(not keep for keep in self._grad_mask()),
        }
        if hasattr(self, 'hidden_router'):
            analysis['temperature'] = float(self.hidden_router.temperature.item())
        if hasattr(self, 'alpha_combine'):
            analysis['hybrid_weight'] = float(torch.sigmoid(self.alpha_combine).item())
        return analysis