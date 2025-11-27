"""
Modified lib/sde_rnn.py with Latent-Level Attention Support
Supports:
- Channel Latent Attention (existing)
- Pyramidal Latent Attention (NEW)
- TVF Latent Attention (NEW)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import Encoder_z0_SDE_RNN
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from lib.base_models import Baseline

# Import attention modules
from lib.pyramidal_attention import PyramidalAttention
from lib.attention_mechanisms import (
    ChannelAttention,
    TimeVaryingFeatureAttention,
)


class SDE_RNN(Baseline):
    def __init__(self, input_dim, latent_dim, device=torch.device("cpu"),
                 z0_diffeq_solver=None, n_gru_units=100, n_units=100,
                 concat_mask=False, obsrv_std=0.1,
                 use_binary_classif=False, classif_per_tp=False,
                 n_labels=1, train_classif_w_reconstr=False,
                 # Latent-level attention parameters
                 use_channel_latent_attention=False,
                 use_pyramidal_latent_attention=False,
                 use_tvf_latent_attention=False,
                 num_attention_levels=3,
                 attention_hidden_dim=32,
                 tvf_method='lstm'):
        """
        SDE-RNN with latent-level attention mechanisms

        Args:
            use_channel_latent_attention: Apply channel attention in latent space
            use_pyramidal_latent_attention: Apply pyramidal attention in latent space
            use_tvf_latent_attention: Apply TVF attention in latent space
            num_attention_levels: Number of levels for pyramidal attention
            attention_hidden_dim: Hidden dimension for attention modules
            tvf_method: Method for TVF attention ('lstm' or 'transformer')
        """

        super().__init__(input_dim, latent_dim, device,
                         obsrv_std, use_binary_classif,
                         classif_per_tp, use_poisson_proc=False,
                         linear_classifier=False, n_labels=n_labels,
                         train_classif_w_reconstr=train_classif_w_reconstr)

        # === Initialize Latent-level Attention Modules ===
        channel_latent_attention = None
        pyramidal_latent_attention = None
        tvf_latent_attention = None

        if use_channel_latent_attention:
            channel_latent_attention = ChannelAttention(
                input_dim=latent_dim,
                hidden_dim=attention_hidden_dim
            ).to(device)
            print(f"[SDE-RNN] Channel Latent Attention enabled (latent_dim={latent_dim})")

        if use_pyramidal_latent_attention:
            pyramidal_latent_attention = PyramidalAttention(
                input_dim=latent_dim,
                num_levels=num_attention_levels,
                hidden_dim=attention_hidden_dim
            ).to(device)
            print(f"[SDE-RNN] Pyramidal Latent Attention enabled (levels={num_attention_levels}, latent_dim={latent_dim})")

        if use_tvf_latent_attention:
            tvf_latent_attention = TimeVaryingFeatureAttention(
                input_dim=latent_dim,
                hidden_dim=attention_hidden_dim,
                method=tvf_method
            ).to(device)
            print(f"[SDE-RNN] TVF Latent Attention enabled (method={tvf_method}, latent_dim={latent_dim})")

        # === Encoder ===
        self.sde_gru = Encoder_z0_SDE_RNN(
            latent_dim=latent_dim,
            input_dim=input_dim * 2,  # data + mask
            z0_diffeq_solver=z0_diffeq_solver,
            n_gru_units=n_gru_units,
            device=device,
            use_channel_latent_attention=use_channel_latent_attention,
            channel_latent_attention=channel_latent_attention,
            use_pyramidal_latent_attention=use_pyramidal_latent_attention,
            pyramidal_latent_attention=pyramidal_latent_attention,
            use_tvf_latent_attention=use_tvf_latent_attention,
            tvf_latent_attention=tvf_latent_attention
        ).to(device)

        # === Decoder ===
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim)
        )
        utils.init_network_weights(self.decoder)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.use_kld = False
        self._eps = 1e-6

        # Store attention parameters
        self.use_channel_latent_attention = use_channel_latent_attention
        self.use_pyramidal_latent_attention = use_pyramidal_latent_attention
        self.use_tvf_latent_attention = use_tvf_latent_attention

    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps,
                           mask=None, n_traj_samples=None, mode=None):
        """
        Main forward pass with latent-level attention support
        """

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (
                torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for SDE-RNN")

        assert len(truth_time_steps) == len(time_steps_to_predict)
        assert mask is not None

        # === Concatenate data and mask ===
        data_and_mask = torch.cat([data, mask], dim=-1)

        # === Run SDE-GRU encoder (with latent-level attention applied inside) ===
        last_y, last_std, latent_ys, extra_info_enc = self.sde_gru.run_sdernn(
            data_and_mask, truth_time_steps, run_backwards=False
        )

        # Permute to (n_traj_samples, n_traj, n_tp, latent_dim)
        latent_ys = latent_ys.permute(0, 2, 1, 3)
        last_hidden = latent_ys[:, :, -1, :]

        # === Decode ===
        outputs = self.decoder(latent_ys)
        first_point = data[:, 0, :]
        outputs = utils.shift_outputs(outputs, first_point)

        # === Prepare info dict ===
        info = {
            'first_point': (last_y, last_std, extra_info_enc),
            'latent_traj': latent_ys
        }

        # Add z0 statistics for KL loss
        z0_mu = last_y.squeeze(0)
        z0_std = (last_std.squeeze(0) if torch.is_tensor(last_std) else None)
        if z0_std is None:
            z0_std = torch.full_like(z0_mu, fill_value=0.1)
        z0_logvar = torch.log(z0_std.clamp_min(self._eps) ** 2)

        info["z0_mu"] = z0_mu
        info["z0_logvar"] = z0_logvar

        # === Classification ===
        if self.use_binary_classif:
            if self.classif_per_tp:
                # For per-timepoint classification, keep 4D shape [n_traj_samples, n_traj, n_tp, n_labels]
                n_traj_samples, n_traj, n_tp, latent_dim = latent_ys.shape
                # Reshape to [n_traj_samples * n_traj * n_tp, latent_dim] for classifier
                latent_flat = latent_ys.reshape(-1, latent_dim)
                logits_flat = self.classifier(latent_flat)
                # Reshape back to 4D
                logits = logits_flat.reshape(n_traj_samples, n_traj, n_tp, -1)
                info["label_predictions"] = logits
            else:
                logits = self.classifier(last_hidden.squeeze(0))
                info["label_predictions"] = logits.squeeze(-1)

        return outputs, info