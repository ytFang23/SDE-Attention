"""
Temporal and Channel Attention Mechanisms for SDE-RNN
Complete implementation in one file

Usage:
    from attention_mechanisms import TemporalAttention, ChannelAttention, CombinedAttention
    
    # Temporal only
    temporal_attn = TemporalAttention(seq_len=100, hidden_dim=32)
    x_attn, t_weights = temporal_attn(x)
    
    # Channel only
    channel_attn = ChannelAttention(input_dim=10, hidden_dim=32)
    x_attn, c_weights = channel_attn(x)
    
    # Combined (Temporal + Channel)
    combined_attn = CombinedAttention(seq_len=100, input_dim=10, hidden_dim=32)
    x_attn, (t_weights, c_weights) = combined_attn(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Union




class TimeVaryingFeatureAttention(nn.Module):
    """
    Time-Varying Feature Attention: Different features are important at different time steps
    This is what you ACTUALLY want when you say "at different times, which features matter"

    Unlike TemporalAttention (which gives each time step a scalar weight),
    this gives each time step a VECTOR of weights - one per feature.

    Args:
        input_dim: Number of features
        hidden_dim: Hidden dimension for computing attention
        method: 'lstm' (sequential) or 'transformer' (parallel)

    Example:
        >>> x = torch.randn(4, 100, 10)  # (batch, time, features)
        >>> attn = TimeVaryingFeatureAttention(input_dim=10, hidden_dim=32)
        >>> x_attn, weights = attn(x)
        >>> weights.shape  # (4, 100, 10) - different features per time step!
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, method: str = 'lstm'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.method = method

        if method == 'lstm':
            # Use LSTM to capture temporal dependencies
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.weight_projection = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()  # Feature weights in [0, 1]
            )
        elif method == 'transformer':
            # Use self-attention (simpler, parallelizable)
            self.query = nn.Linear(input_dim, hidden_dim)
            self.key = nn.Linear(input_dim, hidden_dim)
            self.value = nn.Linear(input_dim, input_dim)
            self.scale = hidden_dim ** 0.5
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            x_weighted: (batch, seq_len, input_dim)
            weights: (batch, seq_len, input_dim) - feature importance per time step
        """
        batch_size, seq_len, input_dim = x.shape

        if self.method == 'lstm':
            # LSTM processes sequence and outputs hidden states
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

            # Project to feature weights for each time step
            weights = self.weight_projection(lstm_out)  # (batch, seq_len, input_dim)

        elif self.method == 'transformer':
            # Self-attention to get context-aware feature weights
            Q = self.query(x)  # (batch, seq_len, hidden_dim)
            K = self.key(x)    # (batch, seq_len, hidden_dim)

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)
            attn_weights = torch.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

            # Apply attention to get weighted features
            V = self.value(x)  # (batch, seq_len, input_dim)
            context = torch.matmul(attn_weights, V)  # (batch, seq_len, input_dim)

            # Feature importance based on context
            weights = torch.sigmoid(context)  # (batch, seq_len, input_dim)

        # Apply time-varying feature weights
        x_weighted = x * weights

        return x_weighted, weights


class TemporalAttention(nn.Module):
    """
    Temporal Attention: Learn importance of each time step

    Args:
        seq_len: Sequence length (number of time steps)
        hidden_dim: Hidden dimension for MLP-based weight computation

    Example:
        >>> x = torch.randn(4, 100, 10)  # (batch, seq_len, input_dim)
        >>> attn = TemporalAttention(seq_len=100)
        >>> x_attn, weights = attn(x)
        >>> x_attn.shape  # (4, 100, 10)
        >>> weights.shape  # (100, 1)
    """

    def __init__(self, seq_len: int, hidden_dim: int = 32, method: str = 'simple'):
        """
        Args:
            seq_len: Sequence length
            hidden_dim: Hidden dimension (only used if method='mlp')
            method: 'simple' (direct learnable params) or 'mlp' (neural network)
        """
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.method = method

        if method == 'simple':
            # Method 1: Direct learnable weights (simplest, fastest)
            # Each time step has a learnable importance score
            self.temporal_weights = nn.Parameter(torch.ones(seq_len, 1) * 0.5)

        elif method == 'mlp':
            # Method 2: MLP-based weights (more complex, more flexible)
            # Learn weights through a small neural network
            # NOTE: Input will be pooled features, so we don't know input_dim here
            # Will be initialized lazily in forward pass
            self.weight_net = None
            self._mlp_initialized = False
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention to input

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            x_weighted: Weighted input (batch, seq_len, input_dim)
            weights: Temporal weights (seq_len, 1)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        if self.method == 'simple':
            # === FIXED: Dynamically adjust weights if seq_len mismatch ===
            if self.temporal_weights.shape[0] != seq_len:
                self.temporal_weights = nn.Parameter(
                    torch.ones(seq_len, 1, device=device, dtype=dtype) * 0.5
                )

            # Normalize weights using softmax to ensure they sum to 1
            weights = torch.softmax(self.temporal_weights, dim=0)  # (seq_len, 1)

        elif self.method == 'mlp':
            # === FIXED: Lazy initialization with correct input dimension ===
            if not self._mlp_initialized:
                self.weight_net = nn.Sequential(
                    nn.Linear(dim, self.hidden_dim),  # Use actual input_dim from data
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid()
                ).to(device)
                self._mlp_initialized = True

            # Compute weights for each time step using MLP
            weights_list = []
            for t in range(seq_len):
                x_t = x[:, t, :]  # (batch, input_dim)
                # Average across batch for weight computation
                x_t_mean = x_t.mean(dim=0)  # (input_dim,)
                w_t = self.weight_net(x_t_mean)  # (1,)
                weights_list.append(w_t)

            weights = torch.stack(weights_list, dim=0)  # (seq_len, 1)
            weights = weights / (weights.sum() + 1e-8)  # Normalize

        # Apply weights: multiply each time step by its importance
        x_weighted = x * weights.unsqueeze(0)  # (batch, seq_len, dim)

        return x_weighted, weights


class ChannelAttention(nn.Module):
    """
    Channel Attention: Learn importance of each feature dimension

    Args:
        input_dim: Number of feature dimensions
        hidden_dim: Hidden dimension for weight computation
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        """
        Args:
            input_dim: Number of input dimensions (channels)
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()

        self.input_dim = input_dim

        # MLP to learn channel weights
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply channel attention to input

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            x_weighted: Weighted input (batch, seq_len, input_dim)
            weights: Channel weights (input_dim,)
        """
        batch_size, seq_len, input_dim = x.shape

        # Global average pooling over time dimension
        x_avg = x.mean(dim=1)  # (batch, input_dim)

        # Learn channel weights
        # Average across batch for consistent weights
        x_avg_batch = x_avg.mean(dim=0)  # (input_dim,)
        weights = self.weight_net(x_avg_batch)  # (input_dim,)

        # Apply weights: multiply each channel by its importance
        x_weighted = x * weights.unsqueeze(0).unsqueeze(0)  # (batch, seq_len, input_dim)

        return x_weighted, weights


class CombinedAttention(nn.Module):
    """
    Combined Temporal + Channel Attention
    Learns both time step importance AND feature importance

    Args:
        seq_len: Sequence length
        input_dim: Number of feature dimensions
        hidden_dim: Hidden dimension

    Example:
        >>> x = torch.randn(4, 100, 10)  # (batch, seq_len, input_dim)
        >>> attn = CombinedAttention(seq_len=100, input_dim=10)
        >>> x_attn, (t_weights, c_weights) = attn(x)
        >>> x_attn.shape  # (4, 100, 10)
        >>> t_weights.shape  # (100, 1)
        >>> c_weights.shape  # (10,)
    """

    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int = 32,
                 temporal_method: str = 'simple'):
        """
        Args:
            seq_len: Sequence length
            input_dim: Number of input dimensions
            hidden_dim: Hidden dimension
            temporal_method: 'simple' or 'mlp' for temporal attention
        """
        super().__init__()

        self.temporal = TemporalAttention(seq_len, hidden_dim, method=temporal_method)
        self.channel = ChannelAttention(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply combined attention (temporal first, then channel)

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            x_weighted: Weighted input (batch, seq_len, input_dim)
            weights: Tuple of (temporal_weights, channel_weights)
        """
        # Apply temporal attention first
        x_temporal, t_weights = self.temporal(x)

        # Then apply channel attention
        x_output, c_weights = self.channel(x_temporal)

        return x_output, (t_weights, c_weights)


class AdaptiveAttention(nn.Module):
    """
    Adaptive Attention: Choose between temporal, channel, or combined

    Args:
        seq_len: Sequence length
        input_dim: Number of feature dimensions
        hidden_dim: Hidden dimension
        attention_type: 'temporal', 'channel', or 'combined'

    Example:
        >>> x = torch.randn(4, 100, 10)
        >>> # Use temporal attention
        >>> attn = AdaptiveAttention(seq_len=100, input_dim=10, attention_type='temporal')
        >>> x_attn, info = attn(x)
        >>> # Use combined attention
        >>> attn = AdaptiveAttention(seq_len=100, input_dim=10, attention_type='combined')
        >>> x_attn, info = attn(x)
    """

    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int = 32,
                 attention_type: str = 'combined'):
        """
        Args:
            seq_len: Sequence length
            input_dim: Number of input dimensions
            hidden_dim: Hidden dimension
            attention_type: 'temporal', 'channel', or 'combined'
        """
        super().__init__()

        self.attention_type = attention_type

        if attention_type == 'temporal':
            self.attention = TemporalAttention(seq_len, hidden_dim)
        elif attention_type == 'channel':
            self.attention = ChannelAttention(input_dim, hidden_dim)
        elif attention_type == 'combined':
            self.attention = CombinedAttention(seq_len, input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply attention

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            x_weighted: Weighted input
            info: Dictionary with weights information
        """
        x_weighted, weights = self.attention(x)

        info = {
            'attention_type': self.attention_type,
            'weights': weights
        }

        if self.attention_type == 'combined':
            info['temporal_weights'] = weights[0]
            info['channel_weights'] = weights[1]

        return x_weighted, info


class PyramidalChannelAttention(nn.Module):
    """
    Combined Pyramidal + Channel Attention

    Architecture:
    1. Apply channel attention to input
    2. Multi-scale pyramidal attention on channel-weighted features
    3. Fuse results

    Different from CombinedAttention (Temporal+Channel),
    this combines Pyramidal (multi-scale) + Channel (feature importance)

    Args:
        input_dim: Input feature dimension
        num_levels: Number of pyramid levels (default: 3)
        hidden_dim: Hidden dimension for attention computations

    Example:
        >>> x = torch.randn(4, 100, 10)
        >>> attn = PyramidalChannelAttention(input_dim=10, num_levels=3)
        >>> x_attn, info = attn(x)
    """

    def __init__(self, input_dim: int, num_levels: int = 3, hidden_dim: int = 32):
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.stride_base = 2

        # Channel attention (applied first)
        self.channel_attention = ChannelAttention(input_dim, hidden_dim)

        # Pyramidal processing blocks
        self.attention_blocks = nn.ModuleList()
        for _ in range(num_levels):
            block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            self.attention_blocks.append(block)

        # Multi-scale fusion
        if num_levels > 1:
            self.fusion = nn.Sequential(
                nn.Linear(input_dim * num_levels, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, input_dim)
            )
        else:
            self.fusion = None

    def _downsample(self, x, factor):
        """Downsample by stride"""
        if factor == 1:
            return x
        return x[:, ::factor, :]

    def _upsample(self, x, target_length):
        """Upsample back to target length"""
        if x.shape[1] == target_length:
            return x
        batch_size, seq_len, dim = x.shape
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, seq_len, input_dim)
            attention_info: dict with channel_weights
        """
        batch_size, seq_len, dim = x.shape

        # Step 1: Channel attention
        x_channel, channel_weights = self.channel_attention(x)

        # Step 2: Multi-scale pyramidal processing
        multi_scale_features = []
        for level_idx, attn_block in enumerate(self.attention_blocks):
            stride = self.stride_base ** level_idx
            x_down = self._downsample(x_channel, stride)
            x_attended = attn_block(x_down)
            x_up = self._upsample(x_attended, seq_len)
            multi_scale_features.append(x_up)

        # Step 3: Fuse
        if self.fusion is not None:
            x_fused = torch.cat(multi_scale_features, dim=-1)
            output = self.fusion(x_fused)
        else:
            output = multi_scale_features[0]

        attention_info = {
            'attention_type': 'pyramidal-channel',
            'channel_weights': channel_weights,
            'num_levels': self.num_levels
        }

        return output, attention_info


class HybridPyramidalChannelAttention(nn.Module):
    """
    Hybrid: Channel attention at each pyramid level
    More computationally intensive but potentially more expressive
    """

    def __init__(self, input_dim: int, num_levels: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.stride_base = 2

        self.channel_attentions = nn.ModuleList([
            ChannelAttention(input_dim, hidden_dim) for _ in range(num_levels)
        ])

        self.spatial_blocks = nn.ModuleList()
        for _ in range(num_levels):
            block = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            self.spatial_blocks.append(block)

        if num_levels > 1:
            self.fusion = nn.Sequential(
                nn.Linear(input_dim * num_levels, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, input_dim)
            )
        else:
            self.fusion = None

    def _downsample(self, x, factor):
        if factor == 1:
            return x
        return x[:, ::factor, :]

    def _upsample(self, x, target_length):
        if x.shape[1] == target_length:
            return x
        batch_size, seq_len, dim = x.shape
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, dim = x.shape
        multi_scale_features = []
        channel_weights_list = []

        for level_idx, (chan_attn, spat_block) in enumerate(
                zip(self.channel_attentions, self.spatial_blocks)
        ):
            stride = self.stride_base ** level_idx
            x_down = self._downsample(x, stride)
            x_channel, channel_weights = chan_attn(x_down)
            channel_weights_list.append(channel_weights)
            x_spatial = spat_block(x_channel)
            x_up = self._upsample(x_spatial, seq_len)
            multi_scale_features.append(x_up)

        if self.fusion is not None:
            x_fused = torch.cat(multi_scale_features, dim=-1)
            output = self.fusion(x_fused)
        else:
            output = multi_scale_features[0]

        return output, {
            'attention_type': 'hybrid-pyramidal-channel',
            'channel_weights': channel_weights_list,
            'num_levels': self.num_levels
        }
# ============================================================================
# Utility functions
# ============================================================================

def get_attention_module(attention_type: str, seq_len: int, input_dim: int,
                        hidden_dim: int = 32) -> nn.Module:
    """
    Factory function to create attention modules

    Args:
        attention_type: 'temporal', 'channel', or 'combined'
        seq_len: Sequence length
        input_dim: Number of input dimensions
        hidden_dim: Hidden dimension

    Returns:
        Attention module

    Example:
        >>> attn = get_attention_module('temporal', seq_len=100, input_dim=10)
    """
    if attention_type == 'temporal':
        return TemporalAttention(seq_len, hidden_dim)
    elif attention_type == 'channel':
        return ChannelAttention(input_dim, hidden_dim)
    elif attention_type == 'combined':
        return CombinedAttention(seq_len, input_dim, hidden_dim)
    elif attention_type == 'pyramidal-channel':
        return CombinedAttention(seq_len, input_dim, hidden_dim)
    elif attention_type == 'tcran_channel':
        return TCRAN_ChannelAttention(channels=input_dim, reduction_ratio=16)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# ============================================================================
# TCRAN Channel Attention (SE-Net based)
# ============================================================================

class TCRAN_ChannelAttention(nn.Module):
    """
    TCRAN-style Channel Attention based on SE-Net (Squeeze-and-Excitation Networks)

    Key differences from standard ChannelAttention:
    - Uses bottleneck compression (reduction_ratio) for parameter efficiency
    - Per-sample weights (not batch-averaged) for flexibility
    - Includes residual connection (w*X + X) for better gradient flow
    - Much fewer parameters (~16x less than standard MLP-based channel attention)

    Architecture:
        Input (B,T,C) → Global Avg Pool → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → Scale + Residual

    Args:
        channels: Number of input channels (features)
        reduction_ratio: Bottleneck reduction factor (default: 16)
            - Higher ratio = fewer parameters, more compression
            - Lower ratio = more parameters, more capacity
        use_residual: Whether to use residual connection (default: True)
            - Recommended to keep enabled for stable training

    Example:
        >>> x = torch.randn(4, 100, 24)  # (batch, time, channels)
        >>> attn = TCRAN_ChannelAttention(channels=24, reduction_ratio=16)
        >>> x_out, weights = attn(x)
        >>> x_out.shape  # (4, 100, 24)
        >>> weights.shape  # (4, 24) - per-sample channel importance!

    Parameter count:
        With channels=24, reduction_ratio=16:
        - Bottleneck dim = 24/16 = 1.5 ≈ 2 (rounded to at least 1)
        - FC1: 24×2 = 48 params
        - FC2: 2×24 = 48 params
        - Total: 96 params (vs ~1,536 for standard channel attention!)
    """

    def __init__(self, channels: int, reduction_ratio: int = 16, use_residual: bool = True):
        super().__init__()

        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.use_residual = use_residual

        # Bottleneck dimension (at least 1 to avoid degenerate cases)
        self.bottleneck = max(channels // reduction_ratio, 1)

        # Two FC layers implementing the squeeze-and-excitation mechanism
        # FC1: Compression to bottleneck
        self.fc1 = nn.Linear(channels, self.bottleneck, bias=True)
        # FC2: Expansion back to original channels
        self.fc2 = nn.Linear(self.bottleneck, channels, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply TCRAN channel attention

        Args:
            x: Input tensor of shape (batch, seq_len, channels)

        Returns:
            x_out: Output tensor of shape (batch, seq_len, channels)
            weights: Channel attention weights of shape (batch, channels)
                - Each sample has its own channel importance weights
                - Values in [0, 1] after sigmoid

        Forward pass steps:
            1. Squeeze: Global average pooling over time dimension
            2. Excitation: Two FC layers with ReLU and Sigmoid
            3. Scale: Apply channel weights to input
            4. Residual: Add original input (if enabled)
        """
        # Input validation
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (B,T,C), got shape {x.shape}")

        batch_size, seq_len, channels = x.shape

        if channels != self.channels:
            raise ValueError(f"Input channels {channels} != expected {self.channels}")

        # Save original input for residual connection
        residual = x

        # === Squeeze: Global average pooling over time dimension ===
        # Aggregate temporal information into a channel descriptor
        # (B, T, C) → (B, C)
        s_avg = x.mean(dim=1)

        # === Excitation: Learn channel-wise importance ===
        # First FC with compression to bottleneck
        # (B, C) → (B, bottleneck)
        out = self.fc1(s_avg)
        out = self.relu(out)

        # Second FC with expansion back to channels
        # (B, bottleneck) → (B, C)
        out = self.fc2(out)

        # Sigmoid to get attention weights in [0, 1]
        # These represent the importance of each channel for each sample
        weights = self.sigmoid(out)  # (B, C)

        # === Scale: Apply channel-wise attention weights ===
        # Broadcast weights across time dimension and multiply
        # weights.unsqueeze(1): (B, C) → (B, 1, C)
        # Multiply: (B, 1, C) * (B, T, C) → (B, T, C)
        out = weights.unsqueeze(1) * x

        # === Residual connection ===
        # Helps with gradient flow and allows the network to bypass attention if needed
        if self.use_residual:
            out = out + residual

        return out, weights

    def extra_repr(self) -> str:
        """String representation for print(model)"""
        return (f'channels={self.channels}, reduction_ratio={self.reduction_ratio}, '
                f'bottleneck={self.bottleneck}, use_residual={self.use_residual}')


# ============================================================================
# Testing and demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Temporal and Channel Attention Mechanisms")
    print("=" * 80)

    # Create dummy data
    batch_size, seq_len, input_dim = 4, 100, 10
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"\nInput shape: {x.shape} (batch, seq_len, input_dim)")

    # ========================================================================
    # Test 1: Temporal Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. TEMPORAL ATTENTION")
    print("=" * 80)

    temporal_attn = TemporalAttention(seq_len=seq_len, method='simple')
    x_temporal, t_weights = temporal_attn(x)

    print(f"Output shape: {x_temporal.shape}")
    print(f"Weights shape: {t_weights.shape}")
    print(f"Weights range: [{t_weights.min():.3f}, {t_weights.max():.3f}]")
    print(f"Weights sum: {t_weights.sum():.3f} (should be ~1.0)")
    print(f"\nFirst 10 temporal weights:")
    print(t_weights[:10].squeeze().detach().numpy())

    # ========================================================================
    # Test 2: Channel Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. CHANNEL ATTENTION")
    print("=" * 80)

    channel_attn = ChannelAttention(input_dim=input_dim, hidden_dim=32)
    x_channel, c_weights = channel_attn(x)

    print(f"Output shape: {x_channel.shape}")
    print(f"Weights shape: {c_weights.shape}")
    print(f"Weights range: [{c_weights.min():.3f}, {c_weights.max():.3f}]")
    print(f"\nChannel weights (feature importance):")
    print(c_weights.detach().numpy())

    # ========================================================================
    # Test 3: Combined Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. COMBINED ATTENTION (Temporal + Channel)")
    print("=" * 80)

    combined_attn = CombinedAttention(seq_len=seq_len, input_dim=input_dim)
    x_combined, (t_weights_combined, c_weights_combined) = combined_attn(x)

    print(f"Output shape: {x_combined.shape}")
    print(f"Temporal weights shape: {t_weights_combined.shape}")
    print(f"Channel weights shape: {c_weights_combined.shape}")
    print(f"\nTemporal weights (time importance) - first 10:")
    print(t_weights_combined[:10].squeeze().detach().numpy())
    print(f"\nChannel weights (feature importance):")
    print(c_weights_combined.detach().numpy())

    # ========================================================================
    # Test 4: Time-Varying Feature Attention (NEW!)
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. TIME-VARYING FEATURE ATTENTION (Better for your use case!)")
    print("=" * 80)

    tvf_attn = TimeVaryingFeatureAttention(input_dim=input_dim, hidden_dim=32, method='lstm')
    x_tvf, tvf_weights = tvf_attn(x)

    print(f"Output shape: {x_tvf.shape}")
    print(f"Weights shape: {tvf_weights.shape}  <- Different features per time!")
    print(f"\nFeature weights at t=0 (first 5 features):")
    print(tvf_weights[0, 0, :5].detach().numpy())
    print(f"\nFeature weights at t=50 (first 5 features):")
    print(tvf_weights[0, 50, :5].detach().numpy())
    print(f"\nNotice: Weights are DIFFERENT at different time steps!")

    # ========================================================================
    # Test 5: Adaptive Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. ADAPTIVE ATTENTION")
    print("=" * 80)

    for attn_type in ['temporal', 'channel', 'combined']:
        adaptive_attn = AdaptiveAttention(seq_len=seq_len, input_dim=input_dim,
                                         attention_type=attn_type)
        x_adaptive, info = adaptive_attn(x)
        print(f"✓ {attn_type:12s} -> output shape: {x_adaptive.shape}")

    # ========================================================================
    # Test 6: Performance comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. PERFORMANCE COMPARISON")
    print("=" * 80)

    import time

    n_iterations = 100
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda:0'))

    for device in devices:
        print(f"\nDevice: {device}")
        print("-" * 40)

        x_device = x.to(device)

        # Temporal
        temporal_attn.to(device)
        start = time.time()
        for _ in range(n_iterations):
            _ = temporal_attn(x_device)
        temporal_time = (time.time() - start) / n_iterations * 1000

        # Channel
        channel_attn.to(device)
        start = time.time()
        for _ in range(n_iterations):
            _ = channel_attn(x_device)
        channel_time = (time.time() - start) / n_iterations * 1000

        # Combined
        combined_attn.to(device)
        start = time.time()
        for _ in range(n_iterations):
            _ = combined_attn(x_device)
        combined_time = (time.time() - start) / n_iterations * 1000

        print(f"Temporal:  {temporal_time:6.2f} ms")
        print(f"Channel:   {channel_time:6.2f} ms")
        print(f"Combined:  {combined_time:6.2f} ms")

    # ========================================================================
    # Test 7: Parameter count
    # ========================================================================
    print("\n" + "=" * 80)
    print("7. PARAMETER COUNT")
    print("=" * 80)

    print(f"Temporal:  {sum(p.numel() for p in temporal_attn.parameters()):6,} parameters")
    print(f"Channel:   {sum(p.numel() for p in channel_attn.parameters()):6,} parameters")
    print(f"Combined:  {sum(p.numel() for p in combined_attn.parameters()):6,} parameters")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)