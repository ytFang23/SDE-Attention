"""
Attention Mechanisms for Latent-Level Use
"""

import torch
import torch.nn as nn
from typing import Tuple


class TimeVaryingFeatureAttention(nn.Module):
    """
    Time-Varying Feature Attention: Different features are important at different time steps
    Unlike scalar temporal attention, this gives each time step a VECTOR of weights - one per feature.

    Args:
        input_dim: Number of features
        hidden_dim: Hidden dimension for computing attention
        method: 'lstm' (sequential) or 'transformer' (parallel)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, method: str = 'lstm'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.method = method

        if method == 'lstm':
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.weight_projection = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )
        elif method == 'transformer':
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
            weights: (batch, seq_len, input_dim)
        """
        if self.method == 'lstm':
            lstm_out, _ = self.lstm(x)
            weights = self.weight_projection(lstm_out)
        elif self.method == 'transformer':
            Q = self.query(x)
            K = self.key(x)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            V = self.value(x)
            context = torch.matmul(attn_weights, V)
            weights = torch.sigmoid(context)

        x_weighted = x * weights
        return x_weighted, weights


class ChannelAttention(nn.Module):
    """
    Channel Attention: Learn importance of each feature dimension.

    Args:
        input_dim: Number of feature dimensions
        hidden_dim: Hidden dimension for weight computation
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()

        self.input_dim = input_dim
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            x_weighted: (batch, seq_len, input_dim)
            weights: (input_dim,)
        """
        x_avg = x.mean(dim=1)
        x_avg_batch = x_avg.mean(dim=0)
        weights = self.weight_net(x_avg_batch)
        x_weighted = x * weights.unsqueeze(0).unsqueeze(0)
        return x_weighted, weights
