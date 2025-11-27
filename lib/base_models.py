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

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter

def kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-sample KL( N(mu,diag(exp(logvar))) || N(0,I) ), returns shape (B,)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=1)


def create_classifier(z0_dim, n_labels):
    return nn.Sequential(
            nn.Linear(z0_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_labels),)


class Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim, device,
        obsrv_std = 0.01, use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):
        super(Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
        self.device = device

        self.use_binary_classif = use_binary_classif
        self.classif_per_tp = classif_per_tp
        self.use_poisson_proc = use_poisson_proc
        self.linear_classifier = linear_classifier
        self.train_classif_w_reconstr = train_classif_w_reconstr

        z0_dim = latent_dim
        if use_poisson_proc:
            z0_dim += latent_dim

        if use_binary_classif:
            if linear_classifier:
                self.classifier = nn.Sequential(
                    nn.Linear(z0_dim, n_labels))
            else:
                self.classifier = create_classifier(z0_dim, n_labels)
            utils.init_network_weights(self.classifier)


    def get_gaussian_likelihood(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = masked_gaussian_log_density(pred_y, truth,
            obsrv_std = self.obsrv_std, mask = mask)
        log_density_data = log_density_data.permute(1,0)

        # Compute the total density
        # Take mean over n_traj_samples
        log_density = torch.mean(log_density_data, 0)

        # shape: [n_traj]
        return log_density


    def get_mse(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth, mask = mask)

        return torch.mean(log_density_data)

    def compute_all_losses(self, batch_dict,
                           n_tp_to_sample=None, n_traj_samples=1, kl_coef=1.):

        pred_x, info = self.get_reconstruction(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            n_traj_samples=n_traj_samples,
            mode=batch_dict["mode"]
        )

        likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_x,
            mask=batch_dict["mask_predicted_data"]
        )

        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_x,
            mask=batch_dict["mask_predicted_data"]
        )
        device = get_device(batch_dict["data_to_predict"])
        ce_loss = torch.tensor([0.0], device=device)

        if (batch_dict["labels"] is not None) and self.use_binary_classif:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                # Binary classification (e.g., PhysioNet mortality)
                ce_loss = compute_binary_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"]
                )
            else:
                # Multiclass classification (e.g., Activity / UEA)
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"],
                    mask=batch_dict["mask_predicted_data"]
                )

            if torch.isnan(ce_loss):
                print("label pred:", info["label_predictions"])
                print("labels:", batch_dict["labels"])
                raise Exception("CE loss is Nan!")

        pois_log_likelihood = torch.tensor([0.0], device=get_device(batch_dict["data_to_predict"]))
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_x,
                info, mask=batch_dict["mask_predicted_data"]
            )
            # Average over MC trajectories
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        loss = -torch.mean(likelihood)

        # Add Poisson term (if enabled)
        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood

        # Add/replace with classification loss (if enabled)
        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                # Joint training: reconstruction + 100 * CE
                loss = loss + ce_loss * 100.0
            else:
                # Classification-only regime: replace by CE
                loss = ce_loss

        kl_loss = torch.tensor(0.0, device=device)
        kl_applied = False

        def _kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            """Per-sample KL( N(mu,diag(exp(logvar))) || N(0,I) ), returns shape (B,)."""
            return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=1)

        z0_mu = info.get("z0_mu", None) if isinstance(info, dict) else None
        z0_lv = info.get("z0_logvar", None) if isinstance(info, dict) else None

        if (z0_mu is not None) and (z0_lv is not None) and (kl_coef is not None) and (float(kl_coef) > 0.0):
            # Mean per-batch KL
            kl_per = _kl_normal_standard(z0_mu, z0_lv)  # (B,)
            kl_loss = kl_per.mean()  # scalar
            # Optional dynamic weight (e.g., for warmup/anneal), default 1.0
            kl_w = float(getattr(self, "kl_weight", 1.0))
            loss = loss + (float(kl_coef) * kl_w) * kl_loss
            kl_applied = True

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()

        # Report KL stats (0.0 if not applied)
        results["kl"] = kl_loss.detach() if kl_applied else torch.tensor(0.0, device=device)
        results["kl_first_p"] = torch.tensor(0.0, device=device)  # kept for compatibility with downstream code
        results["std_first_p"] = torch.tensor(0.0, device=device)
        results["kl_weight"] = torch.tensor(
            (float(kl_coef) * float(getattr(self, "kl_weight", 1.0))) if kl_applied else 0.0,
            device=device
        )

        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results


class VAE_Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim,
        z0_prior, device,
        obsrv_std = 0.01,
        use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):

        super(VAE_Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.n_labels = n_labels

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

        self.z0_prior = z0_prior
        self.use_binary_classif = use_binary_classif
        self.classif_per_tp = classif_per_tp
        self.use_poisson_proc = use_poisson_proc
        self.linear_classifier = linear_classifier
        self.train_classif_w_reconstr = train_classif_w_reconstr

        z0_dim = latent_dim
        if use_poisson_proc:
            z0_dim += latent_dim

        if use_binary_classif:
            if linear_classifier:
                self.classifier = nn.Sequential(
                    nn.Linear(z0_dim, n_labels))
            else:
                self.classifier = create_classifier(z0_dim, n_labels)
            utils.init_network_weights(self.classifier)


    def get_gaussian_likelihood(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(pred_y, truth_repeated,
            obsrv_std = self.obsrv_std, mask = mask)
        log_density_data = log_density_data.permute(1,0)
        log_density = torch.mean(log_density_data, 1)

        # shape: [n_traj_samples]
        return log_density


    def get_mse(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)
        # shape: [1]
        return torch.mean(log_density_data)


    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):
        # Condition on subsampled points
        # Make predictions for all the points
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"],
            batch_dict["observed_data"], batch_dict["observed_tp"],
            mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        #print("get_reconstruction done -- computing likelihood")
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_distr = Normal(fp_mu, fp_std)

        assert(torch.sum(fp_std < 0) == 0.)

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [n_traj_samples]
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))

        # Compute likelihood of all the points
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y,
                info, mask = batch_dict["mask_predicted_data"])
            # Take mean over n_traj
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ################################
        # Compute CE loss for binary classification on Physionet
        device = get_device(batch_dict["data_to_predict"])
        ce_loss = torch.Tensor([0.]).to(device)

        if (batch_dict["labels"] is not None) and self.use_binary_classif:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"])
            else:
                use_focal = getattr(self, 'use_focal_loss', False)
                focal_gamma = getattr(self, 'focal_gamma', 2.0)

                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"],
                    mask=batch_dict["mask_predicted_data"],
                    use_focal=use_focal,
                    focal_gamma=focal_gamma)

        loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood

        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                loss = loss +  ce_loss * 100
            else:
                loss =  ce_loss

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()

        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results