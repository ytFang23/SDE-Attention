# SDE-RNN with Latent-Level Attention Mechanisms

A PyTorch implementation of Stochastic Differential Equation Recurrent Neural Networks (SDE-RNN) enhanced with multiple attention mechanisms for time series classification and imputation with missing data.

## Overview

This project extends the original Latent ODE framework with **latent-level attention mechanisms** specifically designed for irregular time series with missing observations. Unlike input-level attention, our mechanisms operate in the learned latent space, capturing temporal dynamics more effectively.

### Key Contributions
- ✅ **Three novel latent-level attention mechanisms**: Pyramidal, TVF-LSTM, TVF-Transformer
- ✅ **Comprehensive missing data simulation**: Per-value, per-time, per-dimension schemes
- ✅ **Large-scale evaluation**: 20 UCR datasets × 4 attention variants × 4 missing rates × 4 seeds
- ✅ **Production-ready codebase**: Automated batch processing, result collection, statistical analysis

## Features

### Attention Mechanisms (Latent-Level)
- **Channel Attention**: Learn cross-channel dependencies in latent representations
- **Pyramidal Attention**: Multi-scale temporal patterns with 3-level hierarchy
- **TVF-LSTM Attention**: Sequential temporal feature weighting
- **TVF-Transformer Attention**: Self-attention based temporal modeling

### Supported Models
- **SDE-RNN**: Our main model with SDE-based dynamics
- **ODE-RNN**: Deterministic variant
- **Latent ODE**: VAE with ODE/SDE solvers
- **RNN Baselines**: GRU, LSTM with decay

### Datasets
- **UCR Archive**: 128 univariate classification datasets (auto-downloaded)
- **UEA Archive**: 30 multivariate classification datasets (auto-downloaded)
- **Synthetic**: Configurable periodic time series generator

### Missing Data Handling
- **MCAR (Per-value)**: Missing Completely At Random
- **Per-time**: Missing entire time steps (temporal gaps)
- **Per-dimension**: Missing entire features (sensor failures)
- **Configurable rates**: 0%, 30%, 60%, 90%

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/sde-rnn-attention
cd sde-rnn-attention

# Install dependencies
pip install torch numpy scipy scikit-learn sktime torchsde torchdiffeq

# Datasets will auto-download on first use
```

**Requirements:**
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

## Quick Start

### 1. Single Experiment

**Baseline (no attention)**
```bash
python run_models.py \
    --dataset ucr \
    --ucr-dataset Coffee \
    --sde-rnn \
    --classif \
    --niters 100 \
    --random-seed 42
```

**With Pyramidal Latent Attention**
```bash
python run_models.py \
    --dataset ucr \
    --ucr-dataset Coffee \
    --sde-rnn \
    --classif \
    --pyramidal-latent-attention \
    --attention-levels 3 \
    --attention-hidden-dim 32 \
    --niters 100
```

**With Missing Data**
```bash
python run_models.py \
    --dataset ucr \
    --ucr-dataset Coffee \
    --sde-rnn \
    --classif \
    --pyramidal-latent-attention \
    --ucr-missing-rate 0.3 \
    --ucr-missing-scheme per-value \
    --niters 100
```

### 2. Batch Experiments (Reproducibility)

**Run all 20 datasets with 4 seeds**
```bash
# Baseline
python run_all_ucr.py --variant baseline --multi-seed --seeds 4

# Pyramidal attention
python run_all_ucr.py --variant pyramidal_latent --multi-seed --seeds 4

# TVF-LSTM attention
python run_all_ucr.py --variant tvf_lstm_latent --multi-seed --seeds 4

# TVF-Transformer attention
python run_all_ucr.py --variant tvf_transformer_latent --multi-seed --seeds 4
```

**Test multiple missing rates**
```bash
python run_all_ucr.py \
    --variant pyramidal_latent \
    --miss-rates 0.0 0.3 0.6 0.9 \
    --ucr-missing-scheme per-value \
    --multi-seed \
    --seeds 4
```

**Custom dataset subset**
```bash
python run_all_ucr.py \
    --variant pyramidal_latent \
    --datasets Coffee ECG200 Wafer \
    --multi-seed
```

## Command-Line Arguments

### Essential Arguments
```bash
--dataset {ucr,uea,periodic}     # Dataset type
--ucr-dataset NAME               # Specific UCR dataset (e.g., Coffee)
--sde-rnn                        # Use SDE-RNN model
--classif                        # Classification task
```

### Attention Configuration
```bash
# Pyramidal Attention
--pyramidal-latent-attention     # Enable pyramidal attention
--attention-levels N             # Number of pyramid levels (default: 3)

# TVF Attention  
--tvf-latent-attention           # Enable TVF attention
--tvf-method {lstm,transformer}  # TVF implementation

# Channel Attention
--channel-latent-attention       # Enable channel attention

# Shared
--attention-hidden-dim N         # Hidden dimension for attention (default: 32)
```

### Missing Data Simulation
```bash
--ucr-missing-rate R             # Missing rate [0.0-1.0] (default: 0.0)
--ucr-missing-scheme S           # {per-value, per-time, per-dim}
```

### Model Hyperparameters
```bash
--latents N                      # Latent dimension (default: 25)
--gru-units N                    # GRU hidden units (default: 100)
--units N                        # Decoder hidden units (default: 100)
--batch-size N                   # Batch size (default: 32)
--kl-coef K                      # KL coefficient (default: 1.0)
```

### Training
```bash
--niters N                       # Training iterations (default: 100)
--random-seed N                  # Random seed (default: 42)
--lr R                           # Learning rate (default: 1e-3)
```

### Batch Script (`run_all_ucr.py`)
```bash
--variant {baseline, pyramidal_latent, tvf_lstm_latent, ...}
--multi-seed                     # Enable multi-seed experiments
--seeds N                        # Number of seeds (default: 3)
--seed-list 42 2023 777          # Custom seed list
--miss-rates 0.0 0.3 0.6         # Multiple missing rates
--datasets D1 D2 D3              # Subset of datasets
--exclude D4 D5                  # Exclude datasets
--timeout N                      # Per-run timeout in seconds
```

## Tested Datasets

**UCR Datasets (20 total):**
```
Coffee, ECG200, ECGFiveDays, Earthquakes, ItalyPowerDemand,
Lightning2, MoteStrain, ProximalPhalanxOutlineAgeGroup,
ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, Car,
SmoothSubspace, SonyAIBORobotSurface2, Strawberry, 
SyntheticControl, TwoPatterns, Wafer, MiddlePhalanxOutlineAgeGroup
```

All datasets auto-download on first use.

## Output Structure
```
ucr_results_<variant>_<timestamp>/
├── config.json                          # Experiment configuration
├── results.json                         # All run results
└── <variant>_<dataset>_miss<R>_seed<N>/
    ├── <experiment>.log                 # Training logs
    ├── model.pth                        # Saved checkpoint
    └── metrics.json                     # Per-epoch metrics
```

**Example `results.json`:**
```json
[
  {
    "dataset": "Coffee",
    "seed": 42,
    "miss_rate": 0.3,
    "variant": "pyramidal_latent",
    "status": "success",
    "time": 125.3,
    "final_accuracy": 0.9286
  }
]
```

## Project Structure
```
.
├── lib/
│   ├── sde_rnn.py                  # SDE-RNN with latent attention support
│   ├── pyramidal_attention.py      # Pyramidal attention module
│   ├── attention_mechanisms.py     # Channel & TVF attention
│   ├── encoder_decoder.py          # Encoder with attention integration
│   ├── ucr_adapter.py              # UCR dataset loader
│   ├── uea_adapter.py              # UEA dataset loader
│   ├── diffeq_solver.py            # ODE/SDE solvers
│   ├── base_models.py              # Base model classes
│   └── utils.py                    # Utility functions
├── run_models.py                   # Single experiment runner
├── run_all_ucr.py                  # Batch experiment script
└── README.md
```

## Reproducibility

All experiments use fixed seeds and deterministic settings:
- Default seeds: `[42, 2023, 777, 9999]`
- PyTorch deterministic mode enabled
- CUDA deterministic algorithms when available

**Reproduce paper results:**
```bash
# Run all experiments (20 datasets × 4 variants × 4 seeds × 4 missing rates = 1280 runs)
for variant in baseline pyramidal_latent tvf_lstm_latent tvf_transformer_latent; do
    python run_all_ucr.py \
        --variant $variant \
        --miss-rates 0.0 0.3 0.6 0.9 \
        --multi-seed \
        --seeds 4
done
```

## Performance Notes

**Training Time (per dataset):**
- Baseline: ~2-5 minutes
- With attention: ~3-8 minutes
- Large datasets (e.g., ElectricDevices): 15-30 minutes

**GPU Recommended:**
- Batch experiments benefit from GPU acceleration
- CPU mode works but slower (~3x)

## Citation

This work builds on:
```bibtex
@inproceedings{rubanova2019latent,
  title={Latent ODEs for Irregularly-Sampled Time Series},
  author={Rubanova, Yulia and Chen, Ricky TQ and Duvenaud, David},
  booktitle={NeurIPS},
  year={2019}
}
```

If you use this code, please cite:
```bibtex
@software{SDE-Attention,
  title={SDE-RNN with Latent-Level Attention for Time Series Classification},
  author={Yuting Fang},
  year={2025},
  url={https://github.com/ytfang23/SDE-Attention}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

## Contact

- Issues: [GitHub Issues](https://github.com/ytfang23/SDE-Attention/issues)
- Email: z5518340@ad.unsw.edu.au
## Acknowledgments

- Original Latent ODE implementation by Yulia Rubanova
- UCR/UEA Time Series Archives
- PyTorch and torchsde teams
