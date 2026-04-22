# Latent ODEs for Irregularly-Sampled Time Series

An extended implementation of Latent ODE models for irregularly-sampled time series, built on top of the original work by Rubanova et al. (2019). This repository adds **SDE-RNN**, **latent-level attention mechanisms**, and benchmark support for UCR/UEA time series classification archives.

> **Base paper**: Yulia Rubanova, Ricky Chen, David Duvenaud. *"Latent ODEs for Irregularly-Sampled Time Series"* (NeurIPS 2019). [[arXiv]](https://arxiv.org/abs/1907.03907)

<p align="center">
<img align="middle" src="./assets/viz.gif" width="800" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Extensions](#key-extensions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Datasets](#datasets)
- [Attention Mechanisms](#attention-mechanisms)
- [CLI Reference](#cli-reference)
- [Citation](#citation)

---

## Overview

This codebase supports training and evaluating continuous-time sequence models on both **reconstruction** and **classification** tasks with irregularly-sampled or partially observed time series. Missing data is handled natively via observation masks.

**Core idea**: Encode a time series into a latent initial state using a recognition network (ODE-RNN or RNN), then decode the trajectory by solving an ODE/SDE forward in time — enabling interpolation, extrapolation, and generation at arbitrary time points.

---

## Key Extensions

Beyond the original Latent ODE paper, this repository includes:

| Feature | Description |
|---|---|
| **SDE-RNN** | Stochastic Differential Equation RNN with latent noise for uncertainty modeling |
| **Channel Attention** | Learns global importance weights per feature dimension on latent states |
| **Pyramidal Attention** | Multi-scale transformer attention blocks over latent trajectories |
| **Time-Varying Feature (TVF) Attention** | Per-time-step feature weights via LSTM or Transformer |
| **Conditional Attention Mixture (CAM)** | Data-driven routing of sequences to the best-matching attention expert |
| **UCR Benchmark** | Support for 128+ UCR univariate time series classification datasets |
| **UEA Benchmark** | Support for UEA multivariate time series classification datasets |
| **Missing Data Simulation** | MCAR, per-time, and per-dimension missing data schemes |
| **W&B Logging** | Integrated Weights & Biases experiment tracking |

---

## Installation

**Requirements**: Python 3.8+, PyTorch 2.x

```bash
git clone https://github.com/<your-username>/latent_ode.git
cd latent_ode
pip install -r requirements.txt
```

Key dependencies: `torch >= 2.0`, `torchdiffeq >= 0.2.5`, `torchsde >= 0.2.6`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

---

## Quick Start

**Toy dataset** (1D periodic functions):
```bash
python run_models.py --niters 500 -n 1000 -s 50 -l 10 \
    --dataset periodic --latent-ode --noise-weight 0.01
```

**UCR classification**:
```bash
python run_models.py \
    --dataset ucr --ucr-dataset GunPoint \
    --sde-rnn --classif \
    --niters 200 --batch-size 32 --lr 0.001
```

---

## Models

| Flag | Model | Description |
|---|---|---|
| `--latent-ode` | Latent ODE | VAE with ODE-RNN encoder (Rubanova et al., 2019) |
| `--latent-ode --z0-encoder rnn` | Latent ODE (RNN enc.) | VAE with standard RNN encoder (Chen et al., 2018) |
| `--latent-ode --poisson` | Latent ODE + Poisson | Adds Poisson process likelihood for observation times |
| `--ode-rnn` | ODE-RNN | RNN with ODE transitions between observations |
| `--sde-rnn` | SDE-RNN | RNN with stochastic ODE transitions (this work) |
| `--rnn-vae` | RNN-VAE | VAE with standard RNN encoder/decoder |
| `--classic-rnn` | Classic RNN | Baseline GRU/RNN |
| `--classic-rnn --input-decay --rnn-cell expdecay` | GRU-D | GRU with exponential decay for missing data |

---

## Datasets

### Original Datasets

Raw data is downloaded automatically on first run.

**MuJoCo (Hopper)**:
```bash
python run_models.py --niters 300 -n 10000 -l 15 \
    --dataset hopper --latent-ode \
    --rec-dims 30 --gru-units 100 --units 300 \
    --gen-layers 3 --rec-layers 3
```

**PhysioNet**:
```bash
python run_models.py --niters 100 -n 8000 -l 20 \
    --dataset physionet --latent-ode \
    --rec-dims 40 --rec-layers 3 --gen-layers 3 \
    --units 50 --gru-units 50 --quantization 0.016 --classif
```

**Human Activity**:
```bash
python run_models.py --niters 200 -n 10000 -l 15 \
    --dataset activity --latent-ode \
    --rec-dims 100 --rec-layers 4 --gen-layers 2 \
    --units 500 --gru-units 50 --classif --linear-classif
```

### UCR Archive (128+ univariate datasets)

Data is auto-downloaded (~500 MB) on first use.

```bash
# Basic training
python run_models.py \
    --dataset ucr --ucr-dataset GunPoint \
    --sde-rnn --classif --niters 200

# With 30% MCAR missing data
python run_models.py \
    --dataset ucr --ucr-dataset GunPoint \
    --sde-rnn --classif \
    --ucr-missing-rate 0.3 --ucr-missing-scheme per-value
```

**Missing data schemes** (`--ucr-missing-scheme`):
- `per-value` — MCAR: each observation independently dropped
- `per-time` — Entire time steps dropped
- `per-dim` — Per dimension (equivalent to per-value for univariate data)

### UEA Archive (multivariate datasets)

```bash
python run_models.py \
    --dataset uea --uea-dataset BasicMotions \
    --sde-rnn --classif --niters 200
```

---

## Attention Mechanisms

All attention modules operate on the **latent trajectory** inside SDE-RNN.

| Flag | Mechanism | Description |
|---|---|---|
| `--use-channel-latent-attention` | Channel Attention | Global importance weight per feature dimension |
| `--use-pyramidal-latent-attention` | Pyramidal Attention | Multi-scale transformer attention over latent states |
| `--use-tvf-latent-attention` | Time-Varying Feature | Per-time-step feature weights (LSTM or Transformer) |
| `--use-conditional-attention-mixture` | CAM | Routes each sequence to the best-matching attention expert |

**TVF method** can be selected with `--tvf-method lstm` (default) or `--tvf-method transformer`.

### Conditional Attention Mixture (CAM)

CAM is a mixture-of-experts module. A gating network extracts rich time series statistics (temporal, frequency, statistical, shape, and missingness features) to dynamically route each input to one of three expert attention modules. It includes an expert diversity loss and a dominance warm-up phase (`--cam-dominance-warmup`).

---

## CLI Reference

### Core arguments

| Argument | Description |
|---|---|
| `-n INT` | Number of training samples |
| `-s INT` | Number of time steps (toy dataset) |
| `-l / --latents INT` | Latent dimension size |
| `--niters INT` | Training iterations |
| `-b / --batch-size INT` | Batch size |
| `--lr FLOAT` | Learning rate |
| `--rec-dims INT` | Recognition model dimension |
| `--rec-layers INT` | Recognition model depth |
| `--gen-layers INT` | Generative model depth |
| `--gru-units INT` | GRU hidden units |
| `--units INT` | ODE/SDE network hidden units |
| `--classif` | Enable classification head |
| `--linear-classif` | Use linear classifier (vs. MLP) |
| `--noise-weight FLOAT` | Observation noise weight |
| `--random-seed INT` | Random seed |

### KL regularization

| Argument | Description |
|---|---|
| `--use-kld` | Enable KL divergence regularization |
| `--kl-base FLOAT` | Base KL coefficient (default: 1.0) |
| `--kl-warmup-epochs INT` | Epochs with zero KL weight |
| `--kl-anneal-epochs INT` | Epochs to anneal KL weight |

### W&B logging

```bash
python run_models.py ... \
    --wandb --wandb-project my-project --wandb-entity my-team
```

### Visualization

```bash
python run_models.py --niters 100 -n 5000 -b 100 -l 3 \
    --dataset periodic --latent-ode --noise-weight 0.5 \
    --lr 0.01 --viz --rec-layers 2 --gen-layers 2 -u 100 -c 30
```

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{rubanova2019latent,
  title     = {Latent Ordinary Differential Equations for Irregularly-Sampled Time Series},
  author    = {Rubanova, Yulia and Chen, Ricky T. Q. and Duvenaud, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}
```

Original implementation: [YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode)
