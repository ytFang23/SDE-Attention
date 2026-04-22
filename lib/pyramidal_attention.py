"""
Clean PyTorch Pyramidal Attention - Drop-in Replacement
Replaces the broken original pyramidal_attention.py

Key differences:
- Pure PyTorch (no Keras)
- Fully differentiable
- Integrates with SDE-RNN seamlessly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class ScaledDotProductAttention(nn.Module):
    """Standard Transformer attention mechanism"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = math.sqrt(hidden_dim)

    def forward(self, query, key, value):
        """
        Args:
            query: (batch, seq_len, hidden_dim)
            key: (batch, seq_len, hidden_dim)
            value: (batch, seq_len, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
            weights: (batch, seq_len, seq_len)
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights


class AttentionBlock(nn.Module):
    """Single attention block with residual connections and FFN"""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()

        # Q, K, V projections
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Attention
        self.attention = ScaledDotProductAttention(hidden_dim)

        # Normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, x):
        # Self-attention with residual
        residual = x
        x_norm = self.norm1(x)

        Q = self.query_proj(x_norm)
        K = self.key_proj(x_norm)
        V = self.value_proj(x_norm)

        attended, weights = self.attention(Q, K, V)
        attended = self.output_proj(attended)
        x = residual + attended

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, weights


class PyramidalAttention(nn.Module):
    """
    Multi-scale pyramidal attention

    Processes at different resolutions and fuses results:
    - Level 0: stride 1 (original)
    - Level 1: stride 2
    - Level 2: stride 4
    - ...
    Then fuses all back to original resolution
    """

    def __init__(self, input_dim: int, num_levels: int = 3, hidden_dim: int = 32):
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.stride_base = 2

        # Attention block for each level
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(input_dim, hidden_dim) for _ in range(num_levels)
        ])

        # Fusion layer
        if num_levels > 1:
            self.fusion = nn.Linear(input_dim * num_levels, input_dim)
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
        x = x.permute(0, 2, 1)  # (batch, dim, seq_len)
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # (batch, seq_len, dim)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, seq_len, input_dim)
            weights_list: List of attention weights from each level
        """
        batch_size, seq_len, dim = x.shape

        multi_scale_features = []
        attention_weights_list = []

        # Process each pyramid level
        for level_idx, attn_block in enumerate(self.attention_blocks):
            # Downsample
            stride = self.stride_base ** level_idx
            x_down = self._downsample(x, stride)

            # Apply attention
            x_attended, weights = attn_block(x_down)
            attention_weights_list.append(weights)

            # Upsample
            x_up = self._upsample(x_attended, seq_len)
            multi_scale_features.append(x_up)

        # Fuse multi-scale features
        if self.fusion is not None:
            x_fused = torch.cat(multi_scale_features, dim=-1)  # (batch, seq_len, dim*num_levels)
            output = self.fusion(x_fused)  # (batch, seq_len, dim)
        else:
            output = multi_scale_features[0]

        return output, attention_weights_list