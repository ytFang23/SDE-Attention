###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *
from lib.seqlink_attention import SeqLinkAttention

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter
from lib.base_models import Baseline


class ODE_RNN(Baseline):
    def __init__(self, input_dim, latent_dim, device=torch.device("cpu"),
                 z0_diffeq_solver=None, n_gru_units=100, n_units=100,
                 concat_mask=False, obsrv_std=0.1, use_binary_classif=False,
                 classif_per_tp=False, n_labels=1, train_classif_w_reconstr=False,
                 # Latent-level attention parameters
                 use_channel_latent_attention=False,
                 use_pyramidal_latent_attention=False,
                 use_tvf_latent_attention=False,
                 use_seqlink=False,
                 num_attention_levels=3,
                 attention_hidden_dim=32,
                 tvf_method='lstm'):
        """
        ODE-RNN with latent-level attention mechanisms

        Args:
            use_channel_latent_attention: Apply channel attention in latent space
            use_pyramidal_latent_attention: Apply pyramidal attention in latent space
            use_tvf_latent_attention: Apply TVF attention in latent space
            use_seqlink: Use SeqLink cross-sample pyramidal attention (Abushaqra et al. 2024)
            num_attention_levels: Number of levels for pyramidal attention
            attention_hidden_dim: Hidden dimension for attention modules
            tvf_method: Method for TVF attention ('lstm' or 'transformer')
        """

        Baseline.__init__(self, input_dim, latent_dim, device=device,
                          obsrv_std=obsrv_std, use_binary_classif=use_binary_classif,
                          classif_per_tp=classif_per_tp,
                          n_labels=n_labels,
                          train_classif_w_reconstr=train_classif_w_reconstr)

        ode_rnn_encoder_dim = latent_dim

        # === Initialize Latent-level Attention Modules ===
        channel_latent_attention = None
        pyramidal_latent_attention = None
        tvf_latent_attention = None

        # SeqLink cross-sample attention (independent; applied post-encoder)
        self.use_seqlink = use_seqlink
        if use_seqlink:
            self.seqlink = SeqLinkAttention(
                latent_dim=latent_dim,
                num_levels=num_attention_levels,
                hidden_dim=attention_hidden_dim,
            ).to(device)
            print(f"[ODE-RNN] SeqLink cross-sample attention enabled "
                  f"(levels={num_attention_levels}, hidden_dim={attention_hidden_dim})")
        else:
            self.seqlink = None

        if use_channel_latent_attention:
            from lib.attention_mechanisms import ChannelAttention
            channel_latent_attention = ChannelAttention(
                input_dim=latent_dim,
                hidden_dim=attention_hidden_dim
            ).to(device)
            print(f"[ODE-RNN] Channel Latent Attention enabled (latent_dim={latent_dim})")

        if use_pyramidal_latent_attention:
            from lib.pyramidal_attention import PyramidalAttention
            pyramidal_latent_attention = PyramidalAttention(
                input_dim=latent_dim,
                num_levels=num_attention_levels,
                hidden_dim=attention_hidden_dim
            ).to(device)
            print(f"[ODE-RNN] Pyramidal Latent Attention enabled (levels={num_attention_levels}, latent_dim={latent_dim})")

        if use_tvf_latent_attention:
            from lib.attention_mechanisms import TimeVaryingFeatureAttention
            tvf_latent_attention = TimeVaryingFeatureAttention(
                input_dim=latent_dim,
                hidden_dim=attention_hidden_dim,
                method=tvf_method
            ).to(device)
            print(f"[ODE-RNN] TVF Latent Attention enabled (method={tvf_method}, latent_dim={latent_dim})")

        # === Encoder with attention ===
        self.ode_gru = Encoder_z0_ODE_RNN(
            latent_dim=ode_rnn_encoder_dim,
            input_dim=(input_dim) * 2,  # input and the mask
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

        self.z0_diffeq_solver = z0_diffeq_solver

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim), )

        utils.init_network_weights(self.decoder)

        # Store attention parameters
        self.use_channel_latent_attention = use_channel_latent_attention
        self.use_pyramidal_latent_attention = use_pyramidal_latent_attention
        self.use_tvf_latent_attention = use_tvf_latent_attention

    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps,
                           mask=None, n_traj_samples=None, mode=None):

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (
                torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for ODE-RNN")

        # time_steps_to_predict and truth_time_steps should be the same
        assert (len(truth_time_steps) == len(time_steps_to_predict))
        assert (mask is not None)

        data_and_mask = data
        if mask is not None:
            data_and_mask = torch.cat([data, mask], -1)

        _, _, latent_ys, _ = self.ode_gru.run_odernn(
            data_and_mask, truth_time_steps, run_backwards=False)

        latent_ys = latent_ys.permute(0, 2, 1, 3)
        last_hidden = latent_ys[:, :, -1, :]

        # === SeqLink cross-sample attention (post-encoder, pre-decoder) ===
        if self.use_seqlink and self.seqlink is not None:
            n_samp, B, T, D = latent_ys.shape
            latent_ys_sq = latent_ys.view(n_samp * B, T, D)
            latent_ys_sq = self.seqlink(latent_ys_sq)
            latent_ys = latent_ys_sq.view(n_samp, B, T, D)
            last_hidden = latent_ys[:, :, -1, :]

        # assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        outputs = self.decoder(latent_ys)
        # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
        first_point = data[:, 0, :]
        outputs = utils.shift_outputs(outputs, first_point)

        extra_info = {"first_point": (latent_ys[:, :, -1, :], 0.0, latent_ys[:, :, -1, :])}

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(latent_ys)
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

        # outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
        return outputs, extra_info