###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from person_activity import PersonActivity, variable_time_collate_fn_activity
from lib.uea_adapter import build_uea_dataloaders
from sklearn import model_selection
import random
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ALL_UEA_DATASETS = [
    # High-dimensional datasets (>=6 dimensions)
    "NATOPS", "Heartbeat", "FaceDetection", "SpokenArabicDigits",
    "JapaneseVowels", "ArticularyWordRecognition", "BasicMotions",
    "RacketSports", "SelfRegulationSCP1", "Cricket",

    # Other UEA datasets
    "AtrialFibrillation", "CharacterTrajectories", "EigenWorms", "Epilepsy",
    "ERing", "EthanolConcentration", "FingerMovements", "HandMovementDirection",
    "Handwriting", "InsectWingbeat", "Libras", "LSST",
    "MotorImagery", "PEMS-SF", "PenDigits", "PhonemeSpectra",
    "SelfRegulationSCP2", "StandWalkJump", "UWaveGestureLibrary"
]
#####################################################################################################
def parse_datasets(args, device):
    def basic_collate_fn(batch, time_steps, args=args, device=device, data_type="train"):
        # (B, T, D)
        batch = torch.stack(batch).to(device)
        B, T, D = batch.shape

        # --- build a mask (all ones by default) ---
        mask = torch.ones_like(batch)

        # Only apply synthetic missingness for the periodic dataset and miss_rate < 1
        if getattr(args, "dataset", "") == "periodic" and float(getattr(args, "miss_rate", 1.0)) < 1.0:
            keep = float(args.miss_rate)

            if args.miss_scheme == "per-value":
                # iid drop each entry
                mask = (torch.rand(B, T, D, device=device) < keep).float()

            elif args.miss_scheme == "per-time":
                # drop (or keep) entire timesteps for all dims
                mt = (torch.rand(B, T, 1, device=device) < keep).float()
                mask = mt.expand(B, T, D)

            elif args.miss_scheme == "per-dim":
                # drop (or keep) entire variables across all time
                md = (torch.rand(B, 1, D, device=device) < keep).float()
                mask = md.expand(B, T, D)

            # Optional: zero-out the unobserved entries so encoders that don’t read mask won’t see them
            batch = batch * mask

        data_dict = {
            "data": batch,
            "time_steps": time_steps.to(device).float(),
            "mask": mask,
        }
        # This will create observed_* and *_to_predict parts respecting the mask
        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
        return data_dict

    dataset_name = args.dataset

    n_total_tp = args.timepoints + args.extrap
    max_t_extrap = args.max_t / args.timepoints * n_total_tp

    ##################################################################
    # MuJoCo dataset
    if dataset_name == "hopper":
        dataset_obj = HopperPhysics(root='data', download=True, generate=False, device=device)
        dataset = dataset_obj.get_dataset()[:args.n]
        dataset = dataset.to(device)

        n_tp_data = dataset[:].shape[1]

        # Time steps that are used later on for exrapolation
        time_steps = torch.arange(start=0, end=n_tp_data, step=1).float().to(device)
        time_steps = time_steps / len(time_steps)

        dataset = dataset.to(device)
        time_steps = time_steps.to(device)

    elif args.dataset == "mujoco":
        import mujoco as mujoco_module

        times, train_dl, val_dl, test_dl = mujoco_module.get_mujoco_data(
            batch_size=args.batch_size,
            missing_rate=getattr(args, 'mujoco_missing_rate', 0.0),
            intensity=getattr(args, 'mujoco_intensity', 0.5),
            time_seq=getattr(args, 'mujoco_time_seq', 100),
            y_seq=getattr(args, 'mujoco_y_seq', 50),
            manual_seed=args.random_seed
        )

        input_dim = getattr(args, 'mujoco_intensity', 0.5) + 14

        return {
            'train_dataloader': utils.inf_generator(train_dl),
            'test_dataloader': utils.inf_generator(test_dl),
            'val_dataloader': utils.inf_generator(val_dl),
            'input_dim': input_dim,
            'n_train_batches': len(train_dl),
            'n_test_batches': len(test_dl),
        }
        if not args.extrap:
            # Creating dataset for interpolation
            # sample time points from different parts of the timeline,
            # so that the model learns from different parts of hopper trajectory
            n_traj = len(dataset)
            n_tp_data = dataset.shape[1]
            n_reduced_tp = args.timepoints

            # sample time points from different parts of the timeline,
            # so that the model learns from different parts of hopper trajectory
            start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp + 1, size=n_traj)
            end_ind = start_ind + n_reduced_tp
            sliced = []
            for i in range(n_traj):
                sliced.append(dataset[i, start_ind[i]: end_ind[i], :])
            dataset = torch.stack(sliced).to(device)
            time_steps = time_steps[:n_reduced_tp]

        # Split into train and test by the time sequences
        train_y, test_y = utils.split_train_test(dataset, train_fraq=0.8)

        n_samples = len(dataset)
        input_dim = dataset.size(-1)

        batch_size = min(args.batch_size, args.n)
        train_dataloader = DataLoader(train_y, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type="train"))
        test_dataloader = DataLoader(test_y, batch_size=n_samples, shuffle=False,
                                     collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type="test"))

        data_objects = {"dataset_obj": dataset_obj,
                        "train_dataloader": utils.inf_generator(train_dataloader),
                        "test_dataloader": utils.inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
        return data_objects

    ##################################################################
    # Physionet dataset

    if dataset_name == "physionet":
        train_dataset_obj = PhysioNet('data/physionet', train=True,
                                      quantization=args.quantization,
                                      download=True, n_samples=min(10000, args.n),
                                      device=device)
        # Use custom collate_fn to combine samples with arbitrary time observations.
        # Returns the dataset along with mask and time steps
        test_dataset_obj = PhysioNet('data/physionet', train=False,
                                     quantization=args.quantization,
                                     download=True, n_samples=min(10000, args.n),
                                     device=device)

        # Combine and shuffle samples from physionet Train and physionet Test
        total_dataset = train_dataset_obj[:len(train_dataset_obj)]

        if not args.classif:
            # Concatenate samples from original Train and Test sets
            # Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
            total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                                 random_state=42, shuffle=True)

        record_id, tt, vals, mask, labels = train_data[0]

        n_samples = len(total_dataset)
        input_dim = vals.size(-1)

        batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
        data_min, data_max = get_data_min_max(total_dataset)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: variable_time_collate_fn(batch, args, device,
                                                                                        data_type="train",
                                                                                        data_min=data_min,
                                                                                        data_max=data_max))
        test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False,
                                     collate_fn=lambda batch: variable_time_collate_fn(batch, args, device,
                                                                                       data_type="test",
                                                                                       data_min=data_min,
                                                                                       data_max=data_max))

        attr_names = train_dataset_obj.params
        data_objects = {"dataset_obj": train_dataset_obj,
                        "train_dataloader": utils.inf_generator(train_dataloader),
                        "test_dataloader": utils.inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader),
                        "attr": attr_names,  # optional
                        "classif_per_tp": False,  # optional
                        "n_labels": 1}  # optional
        return data_objects

    ##################################################################
    # Human activity dataset

    if dataset_name == "activity":
        n_samples = min(10000, args.n)
        dataset_obj = PersonActivity('data/PersonActivity',
                                     download=True, n_samples=n_samples, device=device)
        print(dataset_obj)
        # Use custom collate_fn to combine samples with arbitrary time observations.
        # Returns the dataset along with mask and time steps

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8,
                                                                 random_state=42, shuffle=True)

        train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
        test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

        record_id, tt, vals, mask, labels = train_data[0]
        input_dim = vals.size(-1)

        batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: variable_time_collate_fn_activity(batch, args, device,
                                                                                                 data_type="train"))
        test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False,
                                     collate_fn=lambda batch: variable_time_collate_fn_activity(batch, args, device,
                                                                                                data_type="test"))

        data_objects = {"dataset_obj": dataset_obj,
                        "train_dataloader": utils.inf_generator(train_dataloader),
                        "test_dataloader": utils.inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader),
                        "classif_per_tp": True,  # optional
                        "n_labels": labels.size(-1)}

        return data_objects

    if dataset_name == "uea":
        uea_dataset_name = getattr(args, "uea_dataset", "BasicMotions")

        print(f"[INFO] Loading UEA dataset: {uea_dataset_name}")

        from lib.uea_adapter import build_uea_dataloaders

        # Get missing rate parameters
        missing_rate = float(getattr(args, "uea_missing_rate", 0.0))
        missing_scheme = getattr(args, "uea_missing_scheme", "per-value")

        # Validate missing_rate
        if missing_rate < 0.0 or missing_rate >= 1.0:
            raise ValueError(f"uea_missing_rate must be in [0.0, 1.0), got {missing_rate}")

        # Validate missing_scheme
        valid_schemes = ["per-value", "per-time", "per-dim"]
        if missing_scheme not in valid_schemes:
            raise ValueError(f"uea_missing_scheme must be one of {valid_schemes}, got {missing_scheme}")

        # Build dataloaders with missing data
        data_objects = build_uea_dataloaders(
            name=uea_dataset_name,
            batch_size=args.batch_size,
            missing_rate=missing_rate,
            missing_scheme=missing_scheme,
            device=device,
            num_workers=0,
            args=args
        )

        print(f"[INFO] UEA dataset loaded successfully")
        print(f"[INFO] - Input dimension: {data_objects['input_dim']}")
        print(f"[INFO] - Number of classes: {data_objects['n_labels']}")
        print(f"[INFO] - Train batches: {data_objects['n_train_batches']}")
        print(f"[INFO] - Test batches: {data_objects['n_test_batches']}")
        if missing_rate > 0:
            print(f"[INFO] - Missing rate: {missing_rate:.2%} ({missing_scheme})")

        return data_objects

    ##################################################################
    # UCR dataset (univariate time series classification)

    if dataset_name == "ucr":
        ucr_dataset_name = getattr(args, "ucr_dataset", "GunPoint")

        print(f"[INFO] Loading UCR dataset: {ucr_dataset_name}")

        from lib.ucr_adapter import build_ucr_dataloaders

        # Get missing rate parameters
        missing_rate = float(getattr(args, "ucr_missing_rate", 0.0))
        missing_scheme = getattr(args, "ucr_missing_scheme", "per-value")

        # Validate parameters
        if missing_rate < 0.0 or missing_rate >= 1.0:
            raise ValueError(f"ucr_missing_rate must be in [0.0, 1.0), got {missing_rate}")

        valid_schemes = ["per-value", "per-time", "per-dim"]
        if missing_scheme not in valid_schemes:
            raise ValueError(f"ucr_missing_scheme must be one of {valid_schemes}, got {missing_scheme}")

        # Build dataloaders
        data_objects = build_ucr_dataloaders(
            name=ucr_dataset_name,
            batch_size=args.batch_size,
            missing_rate=missing_rate,
            missing_scheme=missing_scheme,
            device=device,
            num_workers=0,
            args=args
        )

        print(f"[INFO] UCR dataset loaded successfully")
        print(f"[INFO] - Input dimension: {data_objects['input_dim']}")
        print(f"[INFO] - Number of classes: {data_objects['n_labels']}")
        print(f"[INFO] - Train batches: {data_objects['n_train_batches']}")
        print(f"[INFO] - Test batches: {data_objects['n_test_batches']}")
        if missing_rate > 0:
            print(f"[INFO] - Missing rate: {missing_rate:.2%} ({missing_scheme})")

        return data_objects
    ########### 1d datasets ###########

    # Sampling args.timepoints time points in the interval [0, args.max_t]
    # Sample points for both training sequence and explapolation (test)
    distribution = uniform.Uniform(torch.Tensor([0.0]), torch.Tensor([max_t_extrap]))
    time_steps_extrap = distribution.sample(torch.Size([n_total_tp - 1]))[:, 0]
    time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
    time_steps_extrap = torch.sort(time_steps_extrap)[0]

    dataset_obj = None
    ##################################################################
    # Sample a periodic function
    if dataset_name == "periodic":
        dataset_obj = Periodic_1d(
            init_freq=None, init_amplitude=1.,
            final_amplitude=1., final_freq=None,
            z0=1.)

    ##################################################################
    if dataset_name in ALL_UEA_DATASETS:
        from lib.uea_adapter import build_uea_dataloaders

        print(f"[INFO] Loading UEA dataset: {dataset_name}")

        data_objects = build_uea_dataloaders(
            name=dataset_name,
            batch_size=args.batch_size,
            missing_rate=getattr(args, 'uea_missing_rate', 0.0),
            missing_scheme=getattr(args, 'uea_missing_scheme', 'per-value'),
            device=device,
            args=args
        )

        return data_objects
    if dataset_obj is None:
        raise Exception("Unknown dataset: {}".format(dataset_name))

    # --- Periodic dataset sampling (supports either i.i.d. or SDE noise) ---
    if dataset_name == "periodic":
        # Periodic_1d expects either numpy or torch time steps; numpy is fine
        ts_np = time_steps_extrap.cpu().numpy()

        # Be consistent with argparse flag names:
        #   --use-sde-noise
        #   --sde-noise-type {bm,ou}
        #   --sde-noise-sigma FLOAT
        #   --sde-ou-theta FLOAT
        #   --sde-ou-mu FLOAT
        #   --noise-weight FLOAT (for i.i.d. noise)
        use_sde = bool(getattr(args, "use_sde_noise", False))

        sde_type = getattr(args, "sde_noise_type", None)  # "bm" or "ou"
        sde_sigma = float(getattr(args, "sde_noise_sigma", 0.15))  # default if not given
        ou_theta = float(getattr(args, "sde_ou_theta", 1.5))
        ou_mu = float(getattr(args, "sde_ou_mu", 0.0))

        noise_weight = float(getattr(args, "noise_weight", 1.0))  # i.i.d. noise strength

        dataset = dataset_obj.sample_traj(
            ts_np,
            n_samples=args.n,
            # exactly one of these paths will be taken inside sample_traj:
            use_sde_noise=use_sde,
            sde_noise_type=sde_type,
            sde_noise_sigma=sde_sigma,
            sde_ou_theta=ou_theta,
            sde_ou_mu=ou_mu,
            noise_weight=noise_weight,
        )

    # Process small datasets
    dataset = dataset.to(device)
    time_steps_extrap = time_steps_extrap.to(device)

    train_y, test_y = utils.split_train_test(dataset, train_fraq=0.8)

    n_samples = len(dataset)
    input_dim = dataset.size(-1)

    batch_size = min(args.batch_size, args.n)
    train_dataloader = DataLoader(train_y, batch_size=batch_size, shuffle=False,
                                  collate_fn=lambda batch: basic_collate_fn(batch, time_steps_extrap,
                                                                            data_type="train"))
    test_dataloader = DataLoader(test_y, batch_size=args.n, shuffle=False,
                                 collate_fn=lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type="test"))

    data_objects = {  # "dataset_obj": dataset_obj,
        "train_dataloader": utils.inf_generator(train_dataloader),
        "test_dataloader": utils.inf_generator(test_dataloader),
        "input_dim": input_dim,
        "n_train_batches": len(train_dataloader),
        "n_test_batches": len(test_dataloader)}

    return data_objects


"""
Extension for parse_datasets.py to support IEEE 37 Bus dataset
Add this to your parse_datasets.py or use as standalone module
"""

# CRITICAL: Define DataRecord for pickle compatibility
class DataRecord:
    """Compatible with IEEE37 generator"""

    def __init__(self, measurement_type=None, values=None, times=None, mask=None):
        self.measurement_type = measurement_type
        self.values = values
        self.times = times
        self.mask = mask


def parse_ieee37_dataset(args, device):
    """Parse IEEE 37 bus dataset in format compatible with run_models.py"""

    # Import utils
    import lib.utils as utils

    # Dataset preparation
    class IEEE37Dataset(Dataset):
        def __init__(self, data_dict):
            self.samples = []
            for node_id, records in data_dict.items():
                max_time = 0
                for record in records:
                    if hasattr(record, 'times') and len(record.times) > 0:
                        max_time = max(max_time, int(record.times[-1]))

                time_points = max_time + 1
                data = torch.zeros(time_points, 3)
                mask = torch.zeros(time_points, 3)

                for record in records:
                    dim_map = {'P': 0, 'Q': 1, 'V': 2}
                    dim_idx = dim_map[record.measurement_type]

                    for time, value, avail in zip(record.times, record.values, record.mask):
                        time_int = int(time)
                        if avail == 1:
                            data[time_int, dim_idx] = value
                            mask[time_int, dim_idx] = 1.0

                self.samples.append({
                    'data': data,
                    'mask': mask,
                    'time': torch.arange(time_points, dtype=torch.float32) / 60.0
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    # Collate function
    def collate_ieee37(batch, data_type="train"):
        data_list = [item['data'] for item in batch]
        mask_list = [item['mask'] for item in batch]

        max_len = max([d.shape[0] for d in data_list])

        # Pad sequences
        padded_data = []
        padded_mask = []
        for data, mask in zip(data_list, mask_list):
            pad_len = max_len - data.shape[0]
            if pad_len > 0:
                data = torch.cat([data, torch.zeros(pad_len, 3)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len, 3)], dim=0)
            padded_data.append(data)
            padded_mask.append(mask)

        batch_data = torch.stack(padded_data).to(device)
        batch_mask = torch.stack(padded_mask).to(device)
        time_steps = torch.arange(max_len, dtype=torch.float32).to(device) / max_len

        # Apply additional missing rate
        if hasattr(args, 'ieee37_missing_rate'):
            rate = float(args.ieee37_missing_rate)
            if data_type == "test":
                rate = float(getattr(args, 'ieee37_test_missing_rate', rate))

            if rate > 0:
                B, T, D = batch_data.shape
                keep_mask = (torch.rand(B, T, D, device=device) > rate).float()
                batch_mask = batch_mask * keep_mask
                batch_data = batch_data * batch_mask

        data_dict = {
            "data": batch_data,
            "time_steps": time_steps,
            "mask": batch_mask
        }

        # Use split_and_subsample_batch
        return utils.split_and_subsample_batch(data_dict, args, data_type=data_type)

    # Load data
    data_path = os.path.join(getattr(args, 'data_root', 'data'), 'ieee_37bus')

    print(f"Loading IEEE 37 bus data from {data_path}")

    # Register DataRecord for pickle
    import sys
    sys.modules['__main__'].DataRecord = DataRecord

    try:
        with open(os.path.join(data_path, 'ieee37_train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(data_path, 'ieee37_test.pkl'), 'rb') as f:
            test_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading IEEE37 data: {e}")
        print("Please run ieee_37bus_dataset_generator.py first")
        raise

    # Create datasets
    train_dataset = IEEE37Dataset(train_data)
    test_dataset = IEEE37Dataset(test_data)

    batch_size = min(args.batch_size, len(train_dataset))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_ieee37(b, "train")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        collate_fn=lambda b: collate_ieee37(b, "test")
    )

    # Return in expected format
    data_objects = {
        "dataset_obj": None,
        "train_dataloader": utils.inf_generator(train_loader),
        "test_dataloader": utils.inf_generator(test_loader),
        "input_dim": 3,
        "n_train_batches": len(train_loader),
        "n_test_batches": len(test_loader),
        "n_labels": 1,
        "classif_per_tp": False
    }

    print(f"IEEE37 loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")

    return data_objects
