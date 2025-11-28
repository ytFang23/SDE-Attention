#!/bin/bash
# test_ucr_attention.sh
# test Attention Variants on UCR
# Including: Baseline, Pyramidal, TVF-LSTM, TVF-Transformer

set -e

MODEL="sde-rnn"
SEEDS=4
NITERS=100

echo "=========================================="
echo "Testing Attention on UCR Datasets"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Seeds: ${SEEDS}"
echo "Iterations: ${NITERS}"
echo ""
echo "Variants to test:"
echo "  1. Baseline"
echo "  2. Pyramidal Attention (levels=3)"
echo "  3. TVF-LSTM (hidden_dim=32)"
echo "  4. TVF-Transformer (hidden_dim=16)"
echo ""
echo "Evaluation: Mean accuracy ± std across seeds"
echo ""

# UCR datasets from top_ucr_datasets.txt
UCR_DATASETS=(
    "Wafer"
    "ProximalPhalanxOutlineAgeGroup"
    "ProximalPhalanxTW"
    "MoteStrain"
    "Earthquakes"
    "ProximalPhalanxOutlineCorrect"
    "SonyAIBORobotSurface2"
    "Strawberry"
    "TwoPatterns"
    "MiddlePhalanxOutlineAgeGroup"
)

# Seeds and miss rates
SEED_LIST=(42 777 9999)
#MISS_RATE_LIST=(0.0 0.3 0.6 0.9)
MISS_RATE_LIST=(0.9)
echo "UCR datasets (${#UCR_DATASETS[@]}): ${UCR_DATASETS[*]}"
echo "Seeds: ${SEED_LIST[*]}"
echo "Miss rates: ${MISS_RATE_LIST[*]}"
echo ""

# ===========================================
# 1. Baseline (no attention)
# ===========================================
echo "=========================================="
echo ">>> Step 1/4: Running BASELINE"
echo "=========================================="

for miss_rate in "${MISS_RATE_LIST[@]}"; do
    echo ""
    echo ">>> Miss Rate: ${miss_rate} - BASELINE"

    python run_all_ucr.py \
        --variant baseline \
        --datasets "${UCR_DATASETS[@]}" \
        --multi-seed \
        --seed-list "${SEED_LIST[@]}" \
        --niters ${NITERS} \
        --miss-rates ${miss_rate} \
        --output-dir "ucr_results/baseline_miss${miss_rate}" || {
            echo "    ⚠️  Baseline miss_rate ${miss_rate} failed, continuing..."
            continue
        }
done

echo ""
echo "✅ Baseline experiments completed!"
echo ""

# ===========================================
# 2. Pyramidal Attention
# ===========================================
echo "=========================================="
echo ">>> Step 2/4: Running PYRAMIDAL ATTENTION"
echo "=========================================="

for miss_rate in "${MISS_RATE_LIST[@]}"; do
    echo ""
    echo ">>> Miss Rate: ${miss_rate} - PYRAMIDAL ATTENTION"

    python run_all_ucr.py \
        --variant pyramidal_latent \
        --datasets "${UCR_DATASETS[@]}" \
        --multi-seed \
        --seed-list "${SEED_LIST[@]}" \
        --niters ${NITERS} \
        --miss-rates ${miss_rate} \
        --output-dir "ucr_results/pyramidal_miss${miss_rate}" || {
            echo "    ⚠️  Pyramidal miss_rate ${miss_rate} failed, continuing..."
            continue
        }
done

echo ""
echo "✅ Pyramidal attention experiments completed!"
echo ""

# ===========================================
# 3. TVF-LSTM Attention
# ===========================================
echo "=========================================="
echo ">>> Step 3/4: Running TVF-LSTM ATTENTION"
echo "=========================================="

for miss_rate in "${MISS_RATE_LIST[@]}"; do
    echo ""
    echo ">>> Miss Rate: ${miss_rate} - TVF-LSTM"

    python run_all_ucr.py \
        --variant tvf_lstm_latent \
        --datasets "${UCR_DATASETS[@]}" \
        --multi-seed \
        --seed-list "${SEED_LIST[@]}" \
        --niters ${NITERS} \
        --miss-rates ${miss_rate} \
        --output-dir "ucr_results/tvf_lstm_miss${miss_rate}" || {
            echo "    ⚠️  TVF-LSTM miss_rate ${miss_rate} failed, continuing..."
            continue
        }
done

echo ""
echo "✅ TVF-LSTM attention experiments completed!"
echo ""

# ===========================================
# 4. TVF-Transformer Attention
# ===========================================
echo "=========================================="
echo ">>> Step 4/4: Running TVF-TRANSFORMER ATTENTION"
echo "=========================================="

for miss_rate in "${MISS_RATE_LIST[@]}"; do
    echo ""
    echo ">>> Miss Rate: ${miss_rate} - TVF-Transformer"

    python run_all_ucr.py \
        --variant tvf_transformer_latent \
        --datasets "${UCR_DATASETS[@]}" \
        --multi-seed \
        --seed-list "${SEED_LIST[@]}" \
        --niters ${NITERS} \
        --miss-rates ${miss_rate} \
        --output-dir "ucr_results/tvf_transformer_miss${miss_rate}" || {
            echo "    ⚠️  TVF-Transformer miss_rate ${miss_rate} failed, continuing..."
            continue
        }
done

echo ""
echo "✅ TVF-Transformer attention experiments completed!"
echo ""


# ===========================================
# 5. Channel Latent Attention
# ===========================================
echo "=========================================="
echo ">>> Step 5/5: Running CHANNEL LATENT ATTENTION"
echo "=========================================="

for miss_rate in "${MISS_RATE_LIST[@]}"; do
    echo ""
    echo ">>> Miss Rate: ${miss_rate} - Channel Latent"

    python run_all_ucr.py \
        --variant channel_latent \
        --datasets "${UCR_DATASETS[@]}" \
        --multi-seed \
        --seed-list "${SEED_LIST[@]}" \
        --niters ${NITERS} \
        --miss-rates ${miss_rate} \
        --output-dir "ucr_results/channel_latent_miss${miss_rate}" || {
            echo "    âš ï¸  Channel Latent miss_rate ${miss_rate} failed, continuing..."
            continue
        }
done

# ===========================================
# Summary
# ===========================================
echo "=========================================="
echo "ðŸŽ‰ ALL UCR ATTENTION EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Total experiments: $((${#UCR_DATASETS[@]} * 5 * ${#SEED_LIST[@]} * ${#MISS_RATE_LIST[@]}))"
echo "  - ${#UCR_DATASETS[@]} UCR datasets"
echo "  - 5 variants (baseline, pyramidal, tvf_lstm, tvf_transformer, channel_latent)"
echo "  - ${#SEED_LIST[@]} seeds each"
echo "  - ${#MISS_RATE_LIST[@]} miss rates each"
echo ""
echo "Results structure:"
echo "  ucr_results/"
echo "    â”œâ”€â”€ baseline_miss0.0/"
echo "    â”œâ”€â”€ baseline_miss0.3/"
echo "    â”œâ”€â”€ baseline_miss0.6/"
echo "    â”œâ”€â”€ baseline_miss0.9/"
echo "    â”œâ”€â”€ pyramidal_miss0.0/"
echo "    â”œâ”€â”€ ..."
echo ""
echo "Next steps:"
echo "  1. Collect results:"
echo "     python collect_ucr_results.py ucr_results/ ucr_collected_results"
echo ""
echo "  2. View summary:"
echo "     ls ucr_collected_results/"
echo ""