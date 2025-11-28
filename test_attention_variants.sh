#!/bin/bash
# test_attention_variants.sh
#test attention variants on UEA
# including baseline, channel, pyramidal, tcran_channel

set -e

MODEL="sde-rnn"
SEEDS=4
NITERS=100

echo "=========================================="
echo "Testing Attention Variants on UEA Datasets"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Seeds: ${SEEDS}"
echo "Iterations: ${NITERS}"
echo ""
echo "Variants to test:"
echo "  1. Baseline"
echo "  2. Channel Attention"
echo "  3. Pyramidal Attention"
echo "  4. TCRAN Channel Attention"
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
SEED_LIST=(42 777 )
MISS_RATE_LIST=(0.0 0.3 0.6 0.9)

echo "High-dimensional datasets (${#HIGH_DIM_DATASETS[@]}): ${HIGH_DIM_DATASETS[*]}"
echo "Seeds: ${SEED_LIST[*]}"
echo "Miss rates: ${MISS_RATE_LIST[*]}"
echo ""

# ===========================================
# 1. Baseline (no attention)
# ===========================================
echo "=========================================="
echo ">>> Step 1/4: Running BASELINE"
echo "=========================================="

for dataset in "${HIGH_DIM_DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: ${dataset} - BASELINE"

    for miss_rate in "${MISS_RATE_LIST[@]}"; do
        echo "  Miss rate: ${miss_rate}"

        for seed in "${SEED_LIST[@]}"; do
            echo "    - Seed ${seed}"
            python run_models.py \
                --dataset ${dataset} \
                --sde-rnn \
                -u 50 \
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
echo "✅ Baseline experiments completed!"
echo ""

# ===========================================
# 2. Channel Attention
# ===========================================
echo "=========================================="
echo ">>> Step 2/4: Running CHANNEL ATTENTION"
echo "=========================================="

for dataset in "${HIGH_DIM_DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: ${dataset} - CHANNEL ATTENTION"

    for miss_rate in "${MISS_RATE_LIST[@]}"; do
        echo "  Miss rate: ${miss_rate}"

        for seed in "${SEED_LIST[@]}"; do
            echo "    - Seed ${seed}"
            python run_models.py \
                --dataset ${dataset} \
                --sde-rnn \
                -u 50 \
                --attention \
                --attention-type channel \
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
echo "✅ Channel attention experiments completed!"
echo ""

# ===========================================
# 3. Pyramidal Attention
# ===========================================
echo "=========================================="
echo ">>> Step 3/4: Running PYRAMIDAL ATTENTION"
echo "=========================================="

for dataset in "${HIGH_DIM_DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: ${dataset} - PYRAMIDAL ATTENTION"

    for miss_rate in "${MISS_RATE_LIST[@]}"; do
        echo "  Miss rate: ${miss_rate}"

        for seed in "${SEED_LIST[@]}"; do
            echo "    - Seed ${seed}"
            python run_models.py \
                --dataset ${dataset} \
                --sde-rnn \
                -u 50 \
                --pyramidal-latent-attention \
                --attention-levels 3 \
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
echo "✅ Pyramidal attention experiments completed!"
echo ""

# ===========================================
# 5. Latent Channel Attention (NEW)
# ===========================================
echo "=========================================="
echo ">>> Running LATENT CHANNEL ATTENTION"
echo "=========================================="

for miss_rate in "${MISS_RATE_LIST[@]}"; do
    echo ""
    echo ">>> Miss Rate: ${miss_rate} - LATENT CHANNEL ATTENTION"

    for dataset in "${HIGH_DIM_DATASETS[@]}"; do
        echo ""
        echo "  Processing dataset: ${dataset}"

        for seed in "${SEED_LIST[@]}"; do
            echo "    Seed: ${seed}"

            output_dir="uea_results/latent_channel_miss${miss_rate}/${dataset}_seed${seed}"
            mkdir -p "${output_dir}"

            python run_models.py \
                --dataset "${dataset}" \
                --niters ${NITERS} \
                -u 50 \
                --random-seed ${seed} \
                --sde-rnn \
                --channel-latent-attention \
                --classif \
                --save "${output_dir}" \
                --uea-missing-rate ${miss_rate} || {
                    echo "    ⚠️  Failed for ${dataset} seed ${seed}, continuing..."
                    continue
                }
        done
    done
done

# ===========================================
# Summary
# ===========================================
echo "=========================================="
echo "🎉 ALL UEA LATENT CHANNEL ATTENTION EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Total experiments: $((${#UEA_DATASETS[@]} * ${#SEED_LIST[@]} * ${#MISS_RATE_LIST[@]}))"
echo "  - ${HIGH_DIM_DATASETS[@]} UEA datasets"
echo "  - ${#SEED_LIST[@]} seeds each"
echo "  - ${#MISS_RATE_LIST[@]} miss rates each"
echo ""
echo "Results structure:"
echo "  uea_results/"
echo "    └── latent_channel_miss0.0/"
echo "    └── latent_channel_miss0.3/"
echo "    └── latent_channel_miss0.6/"
echo "    └── latent_channel_miss0.9/"
echo ""
echo "Next steps:"
echo "  1. Collect results and compute mean/std across seeds"
echo "  2. Compare with other attention mechanisms"
echo ""