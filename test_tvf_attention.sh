#!/bin/bash
# test_tvf_attention.sh
# 测试 TVF (Time-Varying Feature) Attention 在高维 UEA 数据集上的表现
# 包括两种变体: TVF-LSTM 和 TVF-Transformer

set -e

MODEL="sde-rnn"
SEEDS=4
NITERS=100

echo "=========================================="
echo "Testing TVF Attention on High-Dim UEA Datasets"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Seeds: ${SEEDS}"
echo "Iterations: ${NITERS}"
echo ""
echo "Variants to test:"
echo "  1. TVF-LSTM (hidden_dim=32)"
echo "  2. TVF-Transformer (hidden_dim=16)"
echo ""
echo "Evaluation: Mean accuracy ± std across seeds"
echo ""

HIGH_DIM_DATASETS=(
    "FingerMovements"
    "ArticularyWordRecognition"
    "BasicMotions"
    "Epilepsy"
    "SelfRegulationSCP2"
    "Libras"
    "ERing"
    "UWaveGestureLibrary"
)

# Seeds and miss rates
SEED_LIST=(42 777 9999)
MISS_RATE_LIST=(0.0 0.3)

echo "High-dimensional datasets (${#HIGH_DIM_DATASETS[@]}): ${HIGH_DIM_DATASETS[*]}"
echo "Seeds: ${SEED_LIST[*]}"
echo "Miss rates: ${MISS_RATE_LIST[*]}"
echo ""

# ===========================================
# 1. TVF-LSTM Attention
# ===========================================
echo "=========================================="
echo ">>> Step 1/2: Running TVF-LSTM ATTENTION"
echo "=========================================="

for dataset in "${HIGH_DIM_DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: ${dataset} - TVF-LSTM"

    for miss_rate in "${MISS_RATE_LIST[@]}"; do
        echo "  Miss rate: ${miss_rate}"

        for seed in "${SEED_LIST[@]}"; do
            echo "    - Seed ${seed}"
            python run_models.py \
                --dataset ${dataset} \
                --sde-rnn \
                -u 50 \
                --tvf-latent-attention \
                --tvf-method lstm \
                --attention-hidden-dim 32 \
                --niters ${NITERS} \
                --random-seed ${seed} \
                --uea-missing-rate ${miss_rate} \
                --classif || {
                    echo "    ⚠️  Seed ${seed} failed, continuing..."
                    continue
                }
        done
    done
done

echo ""
echo "✅ TVF-LSTM attention experiments completed!"
echo ""

# ===========================================
# 2. TVF-Transformer Attention
# ===========================================
echo "=========================================="
echo ">>> Step 2/2: Running TVF-TRANSFORMER ATTENTION"
echo "=========================================="

for dataset in "${HIGH_DIM_DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: ${dataset} - TVF-Transformer"

    for miss_rate in "${MISS_RATE_LIST[@]}"; do
        echo "  Miss rate: ${miss_rate}"

        for seed in "${SEED_LIST[@]}"; do
            echo "    - Seed ${seed}"
            python run_models.py \
                --dataset ${dataset} \
                --sde-rnn \
                -u 50 \
                --tvf-latent-attention \
                --tvf-method transformer \
                --attention-hidden-dim 16 \
                --niters ${NITERS} \
                --random-seed ${seed} \
                --uea-missing-rate ${miss_rate} \
                --classif || {
                    echo "    ⚠️  Seed ${seed} failed, continuing..."
                    continue
                }
        done
    done
done

echo ""
echo "✅ TVF-Transformer attention experiments completed!"
echo ""

# ===========================================
# Summary
# ===========================================
echo "=========================================="
echo "🎉 ALL TVF ATTENTION EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Total experiments: $((${#HIGH_DIM_DATASETS[@]} * 2 * ${#SEED_LIST[@]} * ${#MISS_RATE_LIST[@]}))"
echo "  - ${#HIGH_DIM_DATASETS[@]} datasets"
echo "  - 2 attention variants (TVF-LSTM, TVF-Transformer)"
echo "  - ${#SEED_LIST[@]} seeds each"
echo "  - ${#MISS_RATE_LIST[@]} miss rates each"
echo ""
echo "Next steps:"
echo "  1. Collect results: python collect_tvf_results.py ./ tvf_results.xlsx"
echo "  2. Compare with baseline/channel/pyramidal results"
echo "  3. Check Excel file for statistical analysis"
echo ""