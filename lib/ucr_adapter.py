# lib/ucr_adapter.py
"""
UCR Time Series Classification Archive Adapter

Handles univariate time series datasets from:
http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip

Similar architecture to uea_adapter.py but for single-dimension time series.
"""

import os
import pathlib
import shutil
import zipfile
import urllib.request
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import load_from_tsfile_to_dataframe

from . import utils

here = pathlib.Path(__file__).resolve().parent


def _data_root() -> pathlib.Path:
    """
    Where to store/read UCR datasets.
    - Preferred: <project_root>/data/UCR
    - Override with env UCR_DATA_ROOT
    """
    env = os.environ.get("UCR_DATA_ROOT")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return (here.parent / "data" / "UCR").resolve()


def _old_data_root() -> pathlib.Path:
    """The previous location (inside lib/) for auto-migration."""
    return (here / "data" / "UCR").resolve()


def _ucr_base_dir() -> pathlib.Path | None:
    """Find the extracted UCR archive directory."""
    base = _data_root()
    # Check for common folder names after extraction
    candidates = [
        base / "Univariate2018_ts",
        base / "Univariate_ts",
        base / "UCRArchive_2018",
        base,  # Also check if datasets are directly under base
    ]
    for c in candidates:
        if c.exists():
            # Verify it has dataset folders
            subdirs = [x for x in c.iterdir() if x.is_dir()]
            if subdirs:
                # Check if any subdir has TRAIN.ts files
                for subdir in subdirs:
                    train_files = list(subdir.glob("*_TRAIN.ts"))
                    if train_files:
                        return c
    return None


def _migrate_if_needed():
    """
    If data exists in the old lib/data/UCR location, move it to the new
    <project_root>/data/UCR once. Safe if already migrated.
    """
    new = _data_root()
    old = _old_data_root()
    if not old.exists():
        return
    new.mkdir(parents=True, exist_ok=True)
    # Move everything from old -> new (files & folders)
    for p in old.iterdir():
        target = new / p.name
        if target.exists():
            continue
        if p.is_dir():
            shutil.move(str(p), str(target))
        else:
            shutil.move(str(p), str(target))
    # Try to remove the now-empty old path
    try:
        old.rmdir()
    except OSError:
        pass


def _ensure_ucr_data():
    """
    Ensure the UCR archive is present and extracted under _data_root().
    Also runs a one-time migration from the old location.
    """
    _migrate_if_needed()
    base = _data_root()
    base.mkdir(parents=True, exist_ok=True)

    if _ucr_base_dir() is not None:
        return

    url = "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip"
    zip_path = base / "Univariate2018_ts.zip"
    print(f"[UCR] Downloading: {url} -> {zip_path}")
    try:
        urllib.request.urlretrieve(url, str(zip_path))
        print(f"[UCR] Extracting: {zip_path} -> {base}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(base))
    except Exception as e:
        raise FileNotFoundError(
            f"[UCR] Failed to download/extract data: {e}\n"
            f"Please manually download from {url} and extract to {base}"
        )

    if _ucr_base_dir() is None:
        raise FileNotFoundError(
            "[UCR] Data extracted but dataset folder not found. "
            "Expected structure: data/UCR/Univariate2018_ts/<dataset_name>/"
        )


def _load_ucr_raw(name: str):
    """Load train and test splits for a UCR dataset."""
    _ensure_ucr_data()
    base_dir = _ucr_base_dir()
    assert base_dir is not None

    # Try to find the dataset directory
    dataset_dir = base_dir / name
    if not dataset_dir.exists():
        # Maybe datasets are directly under base_dir
        possible_dirs = [
            base_dir / name,
            base_dir / "Univariate2018_ts" / name,
            base_dir / "Univariate_ts" / name,
        ]
        for d in possible_dirs:
            if d.exists():
                dataset_dir = d
                break
        else:
            raise FileNotFoundError(
                f"[UCR] Dataset '{name}' not found. Checked: {possible_dirs}"
            )

    base = dataset_dir / name
    tr_path = str(base) + "_TRAIN.ts"
    te_path = str(base) + "_TEST.ts"

    if not pathlib.Path(tr_path).exists():
        raise FileNotFoundError(f"[UCR] Train file not found: {tr_path}")
    if not pathlib.Path(te_path).exists():
        raise FileNotFoundError(f"[UCR] Test file not found: {te_path}")

    Xtr_df, ytr = load_from_tsfile_to_dataframe(tr_path)
    Xte_df, yte = load_from_tsfile_to_dataframe(te_path)
    return (Xtr_df.to_numpy(), np.array(ytr)), (Xte_df.to_numpy(), np.array(yte))


def _pad_stack_univariate(X_list: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad and stack univariate time series.

    Args:
        X_list: Array of samples, each sample contains a single Series (univariate)

    Returns:
        X: (B, T, 1) - padded time series data (univariate expanded to dim 1)
        M: (B, T, 1) - mask indicating valid time points
    """
    B = len(X_list)

    # Extract lengths from the univariate series (dimension 0)
    lengths = [len(sample[0]) for sample in X_list]
    T_max = int(max(lengths))

    # UCR datasets are univariate, so D=1
    D = 1

    X = torch.zeros(B, T_max, D, dtype=torch.float32)
    M = torch.zeros(B, T_max, D, dtype=torch.float32)

    for i, sample in enumerate(X_list):
        T_i = len(sample[0])
        # UCR datasets have only one dimension
        ch = torch.as_tensor(sample[0], dtype=torch.float32)  # (T_i,)

        # Pad if necessary
        if T_i < T_max:
            pad_tail = ch[-1].expand(T_max - T_i)
            ch = torch.cat([ch, pad_tail], dim=0)

        X[i, :, 0] = ch
        M[i, :T_i, 0] = 1.0

    return X, M


def _map_labels_to_int(y: np.ndarray) -> Tuple[torch.Tensor, OrderedDict]:
    """Map string/numeric labels to integer indices."""
    table = OrderedDict()
    idxs = []
    for lab in y:
        if lab not in table:
            table[lab] = len(table)
        idxs.append(table[lab])
    return torch.tensor(idxs, dtype=torch.long), table


class UcrDataset(Dataset):
    """PyTorch Dataset wrapper for UCR time series."""

    def __init__(self, X: torch.Tensor, M: torch.Tensor, y: torch.Tensor):
        """
        Args:
            X: (B, T, 1) - time series data
            M: (B, T, 1) - mask
            y: (B,) - labels
        """
        self.X = X
        self.M = M
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.M[i], self.y[i]


def _make_collate_fn(time_steps: torch.Tensor, n_classes: int, args, device, data_type: str):
    """
    Create collate function that converts integer labels to one-hot encoding.
    """

    def collate(batch):
        X, M, Y = zip(*batch)
        X = torch.stack(X, 0)  # (B, T, 1)
        M = torch.stack(M, 0)  # (B, T, 1)
        Y = torch.stack(Y, 0)  # (B,) - integer labels

        # Convert to one-hot encoding
        Y_onehot = torch.zeros(Y.size(0), n_classes, dtype=torch.float32)
        Y_onehot.scatter_(1, Y.unsqueeze(1), 1.0)  # (B, n_classes)

        data_dict = {
            "data": X.to(device),
            "time_steps": time_steps.to(device),
            "mask": M.to(device),
            "labels": Y_onehot.to(device),
        }

        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
        return data_dict

    return collate


def build_ucr_dataloaders(
        name: str,
        batch_size: int = 64,
        missing_rate: float = 0.0,
        missing_scheme: str = "per-value",
        device: torch.device = torch.device("cpu"),
        num_workers: int = 0,
        args=None,
):
    """
    Build UCR dataloaders with optional missing data simulation.

    Args:
        name: UCR dataset name (e.g., 'GunPoint', 'Coffee', 'ECG200')
        batch_size: Batch size
        missing_rate: Fraction of values to drop (0.0 = no missing, 0.3 = 30% missing)
        missing_scheme: How to create missingness:
            - "per-value": Drop individual values randomly (MCAR)
            - "per-time": Drop entire time steps
            - "per-dim": Drop entire features (N/A for univariate, treated as per-value)
        device: torch device
        num_workers: Number of workers for dataloader
        args: Additional arguments

    Returns:
        Dictionary containing dataloaders and metadata
    """
    # Load raw data
    (Xtr_np, ytr_np), (Xte_np, yte_np) = _load_ucr_raw(name)

    # Pad and stack
    Xtr, Mtr = _pad_stack_univariate(Xtr_np)  # (B, T, 1)
    Xte, Mte = _pad_stack_univariate(Xte_np)

    T = Xtr.shape[1]
    time_steps = torch.linspace(0, T - 1, T, dtype=torch.float32)

    # Apply missing data if requested
    if missing_rate > 0.0:
        print(f"[UCR] Applying missing rate: {missing_rate:.2%} (scheme: {missing_scheme})")

        original_obs_train = Mtr.sum().item()
        original_obs_test = Mte.sum().item()

        if missing_scheme == "per-value":
            # MCAR: Missing Completely At Random - drop individual values
            keep_prob = 1.0 - missing_rate

            # Only drop from originally observed values
            drop_mask_tr = (torch.rand_like(Mtr) < keep_prob).float()
            drop_mask_te = (torch.rand_like(Mte) < keep_prob).float()

            Mtr = Mtr * drop_mask_tr
            Mte = Mte * drop_mask_te

        elif missing_scheme == "per-time":
            # Drop entire time steps
            B_tr, T_tr, D_tr = Mtr.shape
            B_te, T_te, D_te = Mte.shape

            keep_prob = 1.0 - missing_rate

            # Generate time-level dropout: (B, T, 1)
            drop_mask_tr = (torch.rand(B_tr, T_tr, 1) < keep_prob).float()
            drop_mask_te = (torch.rand(B_te, T_te, 1) < keep_prob).float()

            # For univariate, this already covers the single dimension
            Mtr = Mtr * drop_mask_tr
            Mte = Mte * drop_mask_te

        elif missing_scheme == "per-dim":
            # For univariate data, per-dim is equivalent to dropping entire sequences
            # We'll treat it as per-value instead
            print(f"[UCR] Warning: 'per-dim' not meaningful for univariate data, using 'per-value' instead")
            keep_prob = 1.0 - missing_rate
            drop_mask_tr = (torch.rand_like(Mtr) < keep_prob).float()
            drop_mask_te = (torch.rand_like(Mte) < keep_prob).float()
            Mtr = Mtr * drop_mask_tr
            Mte = Mte * drop_mask_te

        else:
            raise ValueError(f"Unknown missing_scheme: {missing_scheme}")

        # Zero out dropped values
        Xtr = Xtr * Mtr
        Xte = Xte * Mte

        # Report statistics
        new_obs_train = Mtr.sum().item()
        new_obs_test = Mte.sum().item()

        actual_missing_train = 1.0 - (new_obs_train / original_obs_train)
        actual_missing_test = 1.0 - (new_obs_test / original_obs_test)

        print(f"[UCR] Train: {original_obs_train:.0f} → {new_obs_train:.0f} obs "
              f"({actual_missing_train:.2%} missing)")
        print(f"[UCR] Test:  {original_obs_test:.0f} → {new_obs_test:.0f} obs "
              f"({actual_missing_test:.2%} missing)")

    # Map labels to integers
    ytr, table = _map_labels_to_int(ytr_np)
    yte = torch.tensor([table.get(lab, -1) for lab in yte_np], dtype=torch.long)
    assert (yte >= 0).all(), "Test set contains unseen labels from training set."
    n_classes = len(table)

    # Create datasets
    train_ds = UcrDataset(Xtr, Mtr, ytr)
    test_ds = UcrDataset(Xte, Mte, yte)

    # Create collate functions
    collate_train = _make_collate_fn(time_steps, n_classes, args, device, data_type="train")
    collate_test = _make_collate_fn(time_steps, n_classes, args, device, data_type="test")

    # Create dataloaders
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
        "input_dim": 1,  # UCR datasets are univariate
        "n_train_batches": len(train_loader),
        "n_test_batches": len(test_loader),
        "classif_per_tp": False,  # Sequence-level classification
        "n_labels": n_classes,
    }
    return data_objects


# Popular UCR dataset names for reference
POPULAR_UCR_DATASETS = [
    # Small datasets (good for quick testing)
    "GunPoint", "Coffee", "ItalyPowerDemand", "SonyAIBORobotSurface1",

    # Medium datasets
    "ECG200", "FaceFour", "Lightning2", "Lightning7", "Trace",
    "TwoLeadECG", "SyntheticControl", "DiatomSizeReduction",

    # Larger datasets
    "Adiac", "Beef", "CBF", "ChlorineConcentration", "CinCECGTorso",
    "ECGFiveDays", "FaceAll", "FacesUCR", "FiftyWords", "FISH",
    "Haptics", "InlineSkate", "MedicalImages", "MoteStrain",
    "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
    "OliveOil", "OSULeaf", "SwedishLeaf", "Symbols", "Wafer",
    "WordSynonyms", "Yoga",
]


def list_available_ucr_datasets():
    """
    List all available UCR datasets in the downloaded archive.

    Returns:
        List of dataset names, or empty list if archive not downloaded.
    """
    base_dir = _ucr_base_dir()
    if base_dir is None:
        return []

    datasets = []
    for item in base_dir.iterdir():
        if item.is_dir():
            # Check if it has the expected _TRAIN.ts file
            train_file = item / f"{item.name}_TRAIN.ts"
            if train_file.exists():
                datasets.append(item.name)

    return sorted(datasets)