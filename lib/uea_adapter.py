# lib/uea_adapter.py
import os
import pathlib
import shutil
import zipfile
import urllib.request
from collections import OrderedDict
from typing import Tuple, Dict, List
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import load_from_tsfile_to_dataframe

from . import utils

here = pathlib.Path(__file__).resolve().parent


def _data_root() -> pathlib.Path:
    """
    Where to store/read UEA datasets.
    - Preferred: <project_root>/data/UEA
    - Override with env UEA_DATA_ROOT
    """
    env = os.environ.get("UEA_DATA_ROOT")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return (here.parent / "data" / "UEA").resolve()


def _old_data_root() -> pathlib.Path:
    """The previous location (inside lib/) for auto-migration."""
    return (here / "data" / "UEA").resolve()


def _uea_base_dir() -> pathlib.Path | None:
    base = _data_root()
    c1 = base / "Multivariate2018_ts"
    c2 = base / "Multivariate_ts"
    if c1.exists(): return c1
    if c2.exists(): return c2
    return None


def _migrate_if_needed():
    """
    If data exists in the old lib/data/UEA location, move it to the new
    <project_root>/data/UEA once. Safe if already migrated.
    """
    new = _data_root()
    old = _old_data_root()
    if not old.exists():
        return
    new.mkdir(parents=True, exist_ok=True)
    for p in old.iterdir():
        target = new / p.name
        if target.exists():
            continue
        if p.is_dir():
            shutil.move(str(p), str(target))
        else:
            shutil.move(str(p), str(target))
    try:
        old.rmdir()
    except OSError:
        pass


def _ensure_uea_data():
    """
    Ensure the UEA archive is present and extracted under _data_root().
    """
    _migrate_if_needed()
    base = _data_root()
    base.mkdir(parents=True, exist_ok=True)

    if _uea_base_dir() is not None:
        return

    url = "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip"
    zip_path = base / "Multivariate2018_ts.zip"
    print(f"[UEA] Downloading: {url} -> {zip_path}")
    urllib.request.urlretrieve(url, str(zip_path))
    print(f"[UEA] Extracting: {zip_path} -> {base}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(base))
    if _uea_base_dir() is None:
        raise FileNotFoundError(
            "[UEA] extracted, but dataset folder not found."
        )


def _load_uea_raw(name: str):
    _ensure_uea_data()
    base_dir = _uea_base_dir()
    assert base_dir is not None
    base = base_dir / name / name
    tr_path = str(base) + "_TRAIN.ts"
    te_path = str(base) + "_TEST.ts"
    Xtr_df, ytr = load_from_tsfile_to_dataframe(tr_path)
    Xte_df, yte = load_from_tsfile_to_dataframe(te_path)
    return (Xtr_df.to_numpy(), np.array(ytr)), (Xte_df.to_numpy(), np.array(yte))


def _detect_variable_length_dataset(X_list: np.ndarray) -> Tuple[bool, Dict]:
    """Detect if dataset has variable-length sequences"""
    lengths = []

    for sample in X_list:
        D = len(sample)
        for d in range(D):
            lengths.append(len(sample[d]))

    lengths = np.array(lengths)
    mean_len = lengths.mean()
    std_len = lengths.std()
    cv = std_len / mean_len if mean_len > 0 else 0

    is_variable = cv > 0.1

    stats = {
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(mean_len),
        "median_length": float(np.median(lengths)),
        "std_length": float(std_len),
        "cv": float(cv),
    }

    return is_variable, stats


def _resample_channel(channel: np.ndarray, target_length: int) -> np.ndarray:
    """Resample channel to target length using linear interpolation"""
    if len(channel) == target_length:
        return channel.astype(np.float32)

    indices = np.linspace(0, len(channel) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(channel)), channel)
    return resampled.astype(np.float32)


def _pad_stack(
        X_list: np.ndarray,
        strategy: str = "pad_to_max",
        target_length: int = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Stack and pad variable-length time series.
    If target_length provided, use it (for test set consistency).
    """
    B = len(X_list)
    D = len(X_list[0])

    is_variable, var_stats = _detect_variable_length_dataset(X_list)

    # If target_length not provided, determine from strategy
    if target_length is None:
        if is_variable:
            if strategy == "resample_to_mean":
                target_length = int(np.round(var_stats['mean_length']))
            elif strategy == "resample_to_median":
                target_length = int(var_stats['median_length'])
            elif strategy == "truncate_to_median":
                target_length = int(var_stats['median_length'])
            else:
                target_length = int(var_stats['max_length'])
        else:
            target_length = int(var_stats['max_length'])

    # Process data with target_length
    if is_variable and strategy != "pad_to_max":
        X_list_processed = []

        for sample in X_list:
            sample_processed = []
            for d in range(D):
                channel = sample[d]

                if strategy in ["resample_to_mean", "resample_to_median"]:
                    channel_new = _resample_channel(channel, target_length)
                elif strategy == "truncate_to_median":
                    if len(channel) >= target_length:
                        channel_new = channel[:target_length].astype(np.float32)
                    else:
                        channel_new = np.pad(
                            channel.astype(np.float32),
                            (0, target_length - len(channel)),
                            mode='constant',
                            constant_values=0
                        )

                sample_processed.append(channel_new)

            X_list_processed.append(np.array(sample_processed, dtype=np.float32))

        X_list = X_list_processed

    T_max = target_length

    X = torch.zeros(B, T_max, D, dtype=torch.float32)
    M = torch.zeros(B, T_max, D, dtype=torch.float32)

    for i, sample in enumerate(X_list):
        for d in range(D):
            try:
                channel = sample[d]
                if isinstance(channel, np.ndarray):
                    channel = channel.astype(np.float32)

                ch = torch.as_tensor(channel, dtype=torch.float32)
                T_d = len(ch)
                X[i, :T_d, d] = ch
                M[i, :T_d, d] = 1.0
            except Exception as e:
                print(f"[UEA] Warning: sample {i}, dimension {d} failed: {e}")
                continue

    stats = {
        "is_variable": is_variable,
        "variable_stats": var_stats,
        "final_shape": (B, T_max, D),
        "target_length": T_max,
        "sparsity": float((M.sum() / M.numel()).item()),
    }

    return X, M, stats


def _map_labels_to_int(y: np.ndarray) -> Tuple[torch.Tensor, OrderedDict]:
    table = OrderedDict()
    idxs = []
    for lab in y:
        if lab not in table:
            table[lab] = len(table)
        idxs.append(table[lab])
    return torch.tensor(idxs, dtype=torch.long), table


class UeaDataset(Dataset):
    def __init__(self, X: torch.Tensor, M: torch.Tensor, y: torch.Tensor):
        """
        X: (B,T,D), M: (B,T,D), y: (B,)
        """
        self.X = X
        self.M = M
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.M[i], self.y[i]


def _make_collate_fn(time_steps: torch.Tensor, n_classes: int, args, device, data_type: str):
    """Convert labels to one-hot encoding"""

    def collate(batch):
        X, M, Y = zip(*batch)
        X = torch.stack(X, 0)
        M = torch.stack(M, 0)
        Y = torch.stack(Y, 0)

        Y_onehot = torch.zeros(Y.size(0), n_classes, dtype=torch.float32)
        Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)

        data_dict = {
            "data": X.to(device),
            "time_steps": time_steps.to(device),
            "mask": M.to(device),
            "labels": Y_onehot.to(device),
        }

        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
        return data_dict

    return collate


def build_uea_dataloaders(
        name: str,
        batch_size: int = 64,
        missing_rate: float = 0.0,
        missing_scheme: str = "per-value",
        device: torch.device = torch.device("cpu"),
        num_workers: int = 0,
        args=None,
        variable_length_strategy: str = "auto",
):
    """Build UEA dataloaders with variable-length handling"""

    (Xtr_np, ytr_np), (Xte_np, yte_np) = _load_uea_raw(name)

    print(f"\n[UEA] Processing dataset: {name}")
    print(f"[UEA] Train set: {len(Xtr_np)} samples")
    print(f"[UEA] Test set: {len(Xte_np)} samples")

    # Detect strategy from train set
    is_variable, var_stats = _detect_variable_length_dataset(Xtr_np)

    if is_variable:
        print(f"[UEA] Detected variable-length dataset:")
        print(f"  - Length range: {var_stats['min_length']} ~ {var_stats['max_length']}")
        print(f"  - Mean length: {var_stats['mean_length']:.1f}")
        print(f"  - Coefficient of variation: {var_stats['cv']:.3f}")

        if variable_length_strategy == "auto":
            if var_stats['cv'] > 0.3:
                strategy = "resample_to_median"
            else:
                strategy = "resample_to_mean"
            print(f"  - Auto-selected strategy: {strategy}")
        else:
            strategy = variable_length_strategy
    else:
        strategy = "pad_to_max"
        print(f"[UEA] Consistent length dataset (CV={var_stats['cv']:.3f})")

    # Compute target length from train set
    if strategy == "resample_to_mean":
        target_length = int(np.round(var_stats['mean_length']))
    elif strategy == "resample_to_median":
        target_length = int(var_stats['median_length'])
    elif strategy == "truncate_to_median":
        target_length = int(var_stats['median_length'])
    else:
        target_length = int(var_stats['max_length'])

    print(f"[UEA] Using target length: {target_length}")

    # Process train and test with SAME target_length
    print(f"[UEA] Processing train set...")
    Xtr, Mtr, tr_stats = _pad_stack(Xtr_np, strategy=strategy, target_length=target_length)
    print(f"[UEA] Train set shape: {Xtr.shape}, sparsity: {tr_stats['sparsity']:.1%}")

    print(f"[UEA] Processing test set...")
    Xte, Mte, te_stats = _pad_stack(Xte_np, strategy=strategy, target_length=target_length)
    print(f"[UEA] Test set shape: {Xte.shape}, sparsity: {te_stats['sparsity']:.1%}")

    T = Xtr.shape[1]
    time_steps = torch.linspace(0, T - 1, T, dtype=torch.float32)

    if missing_rate > 0.0:
        print(f"\n[UEA] Applying missing data...")
        print(f"[UEA] Missing rate: {missing_rate:.2%} (scheme: {missing_scheme})")

        original_obs_train = Mtr.sum().item()
        original_obs_test = Mte.sum().item()

        if missing_scheme == "per-value":
            keep_prob = 1.0 - missing_rate
            drop_mask_tr = (torch.rand_like(Mtr) < keep_prob).float()
            drop_mask_te = (torch.rand_like(Mte) < keep_prob).float()
            Mtr = Mtr * drop_mask_tr
            Mte = Mte * drop_mask_te

        elif missing_scheme == "per-time":
            B_tr, T_tr, D_tr = Mtr.shape
            B_te, T_te, D_te = Mte.shape
            keep_prob = 1.0 - missing_rate

            drop_mask_tr = (torch.rand(B_tr, T_tr, 1) < keep_prob).float()
            drop_mask_te = (torch.rand(B_te, T_te, 1) < keep_prob).float()

            drop_mask_tr = drop_mask_tr.expand(B_tr, T_tr, D_tr)
            drop_mask_te = drop_mask_te.expand(B_te, T_te, D_te)

            Mtr = Mtr * drop_mask_tr
            Mte = Mte * drop_mask_te

        elif missing_scheme == "per-dim":
            B_tr, T_tr, D_tr = Mtr.shape
            B_te, T_te, D_te = Mte.shape
            keep_prob = 1.0 - missing_rate

            drop_mask_tr = (torch.rand(B_tr, 1, D_tr) < keep_prob).float()
            drop_mask_te = (torch.rand(B_te, 1, D_te) < keep_prob).float()

            drop_mask_tr = drop_mask_tr.expand(B_tr, T_tr, D_tr)
            drop_mask_te = drop_mask_te.expand(B_te, T_te, D_te)

            Mtr = Mtr * drop_mask_tr
            Mte = Mte * drop_mask_te

        Xtr = Xtr * Mtr
        Xte = Xte * Mte

        new_obs_train = Mtr.sum().item()
        new_obs_test = Mte.sum().item()

        actual_missing_train = 1.0 - (new_obs_train / original_obs_train)
        actual_missing_test = 1.0 - (new_obs_test / original_obs_test)

        print(
            f"[UEA] Train: {original_obs_train:.0f} → {new_obs_train:.0f} observations ({actual_missing_train:.2%} missing)")
        print(
            f"[UEA] Test:  {original_obs_test:.0f} → {new_obs_test:.0f} observations ({actual_missing_test:.2%} missing)")

    ytr, table = _map_labels_to_int(ytr_np)
    yte = torch.tensor([table.get(lab, -1) for lab in yte_np], dtype=torch.long)
    assert (yte >= 0).all(), "Test set contains unseen labels."
    n_classes = len(table)

    train_ds = UeaDataset(Xtr, Mtr, ytr)
    test_ds = UeaDataset(Xte, Mte, yte)

    collate_train = _make_collate_fn(time_steps, n_classes, args, device, data_type="train")
    collate_test = _make_collate_fn(time_steps, n_classes, args, device, data_type="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_train,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_test,
    )

    data_objects = {
        "train_dataloader": utils.inf_generator(train_loader),
        "test_dataloader": utils.inf_generator(test_loader),
        "input_dim": Xtr.shape[-1],
        "n_train_batches": len(train_loader),
        "n_test_batches": len(test_loader),
        "classif_per_tp": False,
        "n_labels": n_classes,
    }
    return data_objects