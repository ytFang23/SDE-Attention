###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import gc
import numpy as np
import sklearn as sk
import numpy as np
# import gc
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
import torch
import torch.nn as nn
from lib.utils import get_device

def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None):
    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
    return log_prob


def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
    # masked_log_lambdas and masked_data
    n_data_points = masked_data.size()[-1]

    if n_data_points > 0:
        log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
    # log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(get_device(masked_data)).squeeze()
    return log_prob


def compute_binary_CE_loss(label_predictions, mortality_label):
    # print("Computing binary classification loss: compute_CE_loss")

    mortality_label = mortality_label.reshape(-1)

    if len(label_predictions.size()) == 1:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj_samples = label_predictions.size(0)
    label_predictions = label_predictions.reshape(n_traj_samples, -1)

    idx_not_nan = ~torch.isnan(mortality_label)
    if len(idx_not_nan) == 0.:
        print("All are labels are NaNs!")
        ce_loss = torch.Tensor(0.).to(get_device(mortality_label))

    label_predictions = label_predictions[:, idx_not_nan]
    mortality_label = mortality_label[idx_not_nan]

    if torch.sum(mortality_label == 0.) == 0 or torch.sum(mortality_label == 1.) == 0:
        print("Warning: all examples in a batch belong to the same class -- please increase the batch size.")

    assert (not torch.isnan(label_predictions).any())
    assert (not torch.isnan(mortality_label).any())

    # For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
    mortality_label = mortality_label.repeat(n_traj_samples, 1)
    ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)

    # divide by number of patients in a batch
    ce_loss = ce_loss / n_traj_samples
    return ce_loss




"""
修复后的compute_multiclass_CE_loss函数

直接替换likelihood_eval.py中的同名函数（第83-177行）

修复内容：
1. 3D分支：不再只使用最后一个时间点，而是报错提示维度不匹配
2. 4D分支：使用label mask而不是observation mask
3. 简化逻辑，提高可读性
"""


def compute_multiclass_CE_loss(label_predictions, true_label, mask=None,
                                use_focal=False, focal_gamma=2.0):
    """
    Compute multiclass cross-entropy loss for classification tasks.

    ⚠️ FIXED VERSION - addresses bugs in original code

    Args:
        label_predictions: Logits from classifier
            - Sequence-level: [n_traj_samples, n_traj, n_labels] (3D)
            - Per-timepoint: [n_traj_samples, n_traj, n_tp, n_labels] (4D)
        true_label: Ground truth labels (one-hot or indices)
            - Sequence-level: [n_traj, n_labels] (2D)
            - Per-timepoint: [n_traj, n_tp, n_labels] (3D)
        mask: Observation mask [n_traj, n_tp, n_dims] (not used for label filtering)
        use_focal: Whether to use Focal Loss instead of CE
        focal_gamma: Focal Loss gamma parameter

    Returns:
        ce_loss: Scalar tensor
    """
    # Ensure at least 3D
    if len(label_predictions.size()) == 2:
        label_predictions = label_predictions.unsqueeze(0)

    # ========================================================================
    # Case 1: Sequence-level classification (3D predictions)
    # ========================================================================
    if len(label_predictions.size()) == 3:
        n_traj_samples, n_traj, n_dims = label_predictions.size()

        # Check true_label dimensions
        if len(true_label.size()) == 2:
            # ✅ Correct: Sequence-level labels [n_traj, n_labels]
            # Repeat for each Monte Carlo sample
            true_label = true_label.repeat(n_traj_samples, 1, 1)

        elif len(true_label.size()) == 3:
            # ❌ BUG DETECTED: Dimension mismatch!
            # Predictions is 3D (sequence-level) but labels is 3D (per-timepoint)
            # This should NOT happen!

            # OLD BUGGY CODE that caused the issue:
            # true_label = true_label[:, -1, :]  # Only use last timepoint! ❌
            # true_label = true_label.repeat(n_traj_samples, 1, 1)

            # NEW CODE: Raise error to force fixing the root cause
            raise ValueError(
                f"\n{'='*70}\n"
                f"DIMENSION MISMATCH BUG DETECTED!\n"
                f"{'='*70}\n"
                f"label_predictions shape: {label_predictions.shape} (3D - sequence-level)\n"
                f"true_label shape: {true_label.shape} (3D - per-timepoint)\n"
                f"\nThis indicates one of two issues:\n"
                f"  1. For per-timepoint tasks, predictions should be 4D:\n"
                f"     [n_traj_samples, n_traj, n_tp, n_labels]\n"
                f"     -> Check if n_traj_samples dimension was squeezed incorrectly\n"
                f"     -> Look at sde_rnn.py get_reconstruction() method\n"
                f"  2. Or true_label should be 2D for sequence-level tasks\n"
                f"\nPrevious buggy code only used the LAST timepoint of labels,\n"
                f"discarding all other labeled timepoints! This caused:\n"
                f"  - Abnormally small CE loss\n"
                f"  - Accuracy jumping from 0 to 1\n"
                f"  - Model only learning the last timepoint pattern\n"
                f"{'='*70}\n"
            )

        # Flatten
        label_predictions = label_predictions.reshape(n_traj_samples * n_traj, n_dims)
        true_label = true_label.reshape(n_traj_samples * n_traj, n_dims)

        # Convert one-hot to indices if needed
        if true_label.size(-1) > 1:
            true_label = true_label.argmax(dim=-1)

        # Compute loss
        if use_focal:
            focal_loss_fn = FocalLoss(gamma=focal_gamma, reduction='mean')
            ce_loss = focal_loss_fn(label_predictions, true_label.long())
        else:
            ce_loss = nn.CrossEntropyLoss()(label_predictions, true_label.long())

        return ce_loss

    # ========================================================================
    # Case 2: Per-timepoint classification (4D predictions)
    # ========================================================================
    elif len(label_predictions.size()) == 4:
        n_traj_samples, n_traj, n_tp, n_dims = label_predictions.size()

        # Expand true_label to match n_traj_samples dimension
        # [n_traj, n_tp, n_labels] -> [n_traj_samples, n_traj, n_tp, n_labels]
        true_label = true_label.unsqueeze(0).repeat(n_traj_samples, 1, 1, 1)

        # ====================================================================
        # ✅ FIX #1: Use LABEL mask, not observation mask
        # ====================================================================
        # OLD BUGGY CODE:
        # mask = torch.sum(mask, -1) > 0  # Uses observation mask ❌
        # This caused: Including timepoints without labels in CE calculation

        # NEW CODE: Compute label mask from labels themselves
        label_mask = (true_label.sum(dim=-1) > 0)  # [n_traj_samples, n_traj, n_tp]
        # True where sum > 0, meaning at least one class is active (has label)

        # Flatten everything
        label_predictions = label_predictions.reshape(-1, n_dims)  # [N, n_dims]
        true_label = true_label.reshape(-1, n_dims)  # [N, n_dims]
        label_mask = label_mask.reshape(-1)  # [N]

        # Select only labeled timepoints
        if label_mask.sum() == 0:
            # No labeled timepoints in this batch
            print("WARNING: No labeled timepoints found in batch!")
            return torch.tensor(0.0, device=get_device(label_predictions))

        label_predictions = label_predictions[label_mask]  # [M, n_dims] where M = num labeled
        true_label = true_label[label_mask]  # [M, n_dims]

        # Debug info (optional, comment out in production)
        # print(f"[CE Loss] Total timepoints: {label_mask.numel()}, "
        #       f"Labeled: {label_mask.sum().item()} ({label_mask.float().mean()*100:.1f}%)")

        # Convert one-hot to indices if needed
        if true_label.size(-1) > 1:
            true_label = true_label.argmax(dim=-1)  # [M]

        # Compute loss
        if use_focal:
            focal_loss_fn = FocalLoss(gamma=focal_gamma, reduction='mean')
            ce_loss = focal_loss_fn(label_predictions, true_label.long())
        else:
            ce_loss = nn.CrossEntropyLoss()(label_predictions, true_label.long())

        return ce_loss

    else:
        raise ValueError(
            f"Unexpected label_predictions shape: {label_predictions.size()}\n"
            f"Expected 3D (sequence-level) or 4D (per-timepoint)"
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (optional)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma focuses more on hard examples
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices (0 to C-1)
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    print("Fixed compute_multiclass_CE_loss function")
    print("="*70)
    print("\n替换方法:")
    print("1. 打开 lib/likelihood_eval.py")
    print("2. 找到第83-177行的 compute_multiclass_CE_loss 函数")
    print("3. 替换为本文件中的版本")
    print("4. 确保导入 FocalLoss 类（如果使用focal loss）")

    print("\n修复内容:")
    print("✅ 3D分支: 不再只使用最后一个时间点")
    print("✅ 4D分支: 使用label mask而不是observation mask")
    print("✅ 添加详细的错误提示")
    print("✅ 代码注释更清晰")

    print("\n测试:")
    device = 'cpu'

    # Test case 1: Sequence-level (3D)
    print("\n[Test 1] Sequence-level classification (3D):")
    preds_3d = torch.randn(2, 4, 7)  # [n_samples=2, n_traj=4, n_labels=7]
    labels_2d = torch.zeros(4, 7)  # [n_traj=4, n_labels=7]
    labels_2d[torch.arange(4), torch.randint(0, 7, (4,))] = 1  # One-hot

    loss_3d = compute_multiclass_CE_loss(preds_3d, labels_2d)
    print(f"  CE loss: {loss_3d.item():.4f}")
    print(f"  ✅ Should be around 1.946 (log(7)) for random predictions")

    # Test case 2: Per-timepoint (4D)
    print("\n[Test 2] Per-timepoint classification (4D):")
    preds_4d = torch.randn(1, 3, 10, 7)  # [n_samples=1, n_traj=3, n_tp=10, n_labels=7]
    labels_3d = torch.zeros(3, 10, 7)  # [n_traj=3, n_tp=10, n_labels=7]
    # Randomly label some timepoints
    for i in range(3):
        for t in range(10):
            if torch.rand(1) > 0.3:  # 70% have labels
                labels_3d[i, t, torch.randint(0, 7, (1,))] = 1

    loss_4d = compute_multiclass_CE_loss(preds_4d, labels_3d)
    print(f"  CE loss: {loss_4d.item():.4f}")
    print(f"  Labeled timepoints: {(labels_3d.sum(dim=-1) > 0).sum().item()} / {3*10}")
    print(f"  ✅ Should be around 1.946 for random predictions")

    # Test case 3: Dimension mismatch (should raise error)
    print("\n[Test 3] Dimension mismatch (should raise error):")
    try:
        preds_3d_bad = torch.randn(1, 3, 7)  # 3D predictions
        labels_3d_bad = torch.zeros(3, 10, 7)  # 3D labels (per-timepoint)
        loss_bad = compute_multiclass_CE_loss(preds_3d_bad, labels_3d_bad)
        print(f"  ❌ Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError:")
        print(f"     {str(e)[:100]}...")

    print("\n" + "="*70)
    print("Tests completed!")

def compute_masked_likelihood(mu, data, mask, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj_samples, n_traj, n_timepoints, n_dims = data.size()

    res = []
    for i in range(n_traj_samples):
        for k in range(n_traj):
            for j in range(n_dims):
                data_masked = torch.masked_select(data[i, k, :, j], mask[i, k, :, j].bool())

                # assert(torch.sum(data_masked == 0.) < 10)

                mu_masked = torch.masked_select(mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = likelihood_func(mu_masked, data_masked, indices=(i, k, j))
                res.append(log_prob)
    # shape: [n_traj*n_traj_samples, 1]

    res = torch.stack(res, 0).to(get_device(data))
    res = res.reshape((n_traj_samples, n_traj, n_dims))
    # Take mean over the number of dimensions
    res = torch.mean(res, -1)  # !!!!!!!!!!! changed from sum to mean
    res = res.transpose(0, 1)
    return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    # these cases are for plotting through plot_estim_density
    if (len(mu.size()) == 3):
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if (len(data.size()) == 2):
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    assert (data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)

        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std=obsrv_std, indices=indices)
        res = compute_masked_likelihood(mu, data, mask, func)
    return res


def mse(mu, data, indices=None):
    n_data_points = mu.size()[-1]

    if n_data_points > 0:
        mse = nn.MSELoss()(mu, data)
    else:
        mse = torch.zeros([1]).to(get_device(data)).squeeze()
    return mse


def compute_mse(mu, data, mask=None):
    # these cases are for plotting through plot_estim_density
    if (len(mu.size()) == 3):
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if (len(data.size()) == 2):
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    assert (data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        res = mse(mu_flat, data_flat)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        res = compute_masked_likelihood(mu, data, mask, mse)
    return res


def compute_poisson_proc_likelihood(truth, pred_y, info, mask=None):
    # Compute Poisson likelihood
    # https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
    # Sum log lambdas across all time points
    if mask is None:
        poisson_log_l = torch.sum(info["log_lambda_y"], 2) - info["int_lambda"]
        # Sum over data dims
        poisson_log_l = torch.mean(poisson_log_l, -1)
    else:
        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask_repeated = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
        int_lambda = info["int_lambda"]
        f = lambda log_lam, data, indices: poisson_log_likelihood(log_lam, data, indices, int_lambda)
        poisson_log_l = compute_masked_likelihood(info["log_lambda_y"], truth_repeated, mask_repeated, f)
        poisson_log_l = poisson_log_l.permute(1, 0)
    # Take mean over n_traj
    # poisson_log_l = torch.mean(poisson_log_l, 1)

    # poisson_log_l shape: [n_traj_samples, n_traj]
    return poisson_log_l


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()