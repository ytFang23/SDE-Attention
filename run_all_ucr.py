#!/usr/bin/env python3
"""
UCR Attention Experiments Batch Script
Enhanced version of run_all_ucr.py with attention mechanism support

Usage:
    # Baseline
    python run_all_ucr.py --model sde-rnn --variant baseline

    # Pyramidal attention
    python run_all_ucr.py --model sde-rnn --variant pyramidal_latent

    # TVF-LSTM
    python run_all_ucr.py --model sde-rnn --variant tvf_lstm_latent

    # TVF-Transformer
    python run_all_ucr.py --model sde-rnn --variant tvf_transformer_latent

    # With missing data
    python run_all_ucr.py --model sde-rnn --variant pyramidal_latent \
        --ucr-missing-rate 0.3 --multi-seed --seeds 4
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# UCR datasets for attention experiments
UCR_DATASETS = [
    "Wafer",
    "ItalyPowerDemand",
    "ProximalPhalanxOutlineAgeGroup",
    "SyntheticControl",
    "ProximalPhalanxTW",
    "SmoothSubspace",
    "MoteStrain",
    "Earthquakes",
    "ProximalPhalanxOutlineCorrect",
    "SonyAIBORobotSurface2",
    "Strawberry",
    # "ElectricDevices",
    "ECGFiveDays",
    "TwoPatterns",
    "MiddlePhalanxOutlineAgeGroup",
    "Car",
    "Lightning2",
    "Coffee"
]

DEFAULT_SEEDS = [42, 2023, 777, 9999]

ATTENTION_CONFIGS = {
    "baseline": {
        "use_attention": False
    },
    "pyramidal_latent": {
        "use_pyramidal_latent_attention": True,
        "attention_levels": 3,
        "attention_hidden_dim": 32,
    },
    "tvf_lstm_latent": {
        "use_tvf_latent_attention": True,
        "tvf_method": "lstm",
        "attention_hidden_dim": 32,
    },
    "tvf_transformer_latent": {
        "use_tvf_latent_attention": True,
        "tvf_method": "transformer",
        "attention_hidden_dim": 16,
    },
    "channel_latent": {
        "use_channel_latent_attention": True,
        "attention_hidden_dim": 32,
    },
}


def get_seeds(args):
    """Generate list of seeds"""
    if not args.multi_seed:
        return [args.seed]

    if args.seed_list:
        return args.seed_list
    elif args.seeds:
        return DEFAULT_SEEDS[:args.seeds]
    else:
        return DEFAULT_SEEDS[:3]


def run_single_experiment(dataset, seed, miss_rate, variant_config, args, output_dir):
    """Run a single experiment"""

    # Create output directory
    run_name = f"{args.variant}_{dataset}_miss{miss_rate}_seed{seed}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, f"{run_name}.log")

    # Build command
    cmd = [
        "python", "run_models.py",
        "--dataset", "ucr",
        "--ucr-dataset", dataset,
        "--sde-rnn",
        "--classif",
        "--latents", str(args.latents),
        "--batch-size", str(args.batch_size),
        "--niters", str(args.niters),
        "--random-seed", str(seed),
        "--save", run_dir,
    ]

    # Add latent attention parameters
    if variant_config.get("use_pyramidal_latent_attention", False):
        cmd.append("--pyramidal-latent-attention")
        if "attention_levels" in variant_config:
            cmd.extend(["--attention-levels", str(variant_config["attention_levels"])])
        if "attention_hidden_dim" in variant_config:
            cmd.extend(["--attention-hidden-dim", str(variant_config["attention_hidden_dim"])])

    if variant_config.get("use_tvf_latent_attention", False):
        cmd.append("--tvf-latent-attention")
        if "tvf_method" in variant_config:
            cmd.extend(["--tvf-method", variant_config["tvf_method"]])
        if "attention_hidden_dim" in variant_config:
            cmd.extend(["--attention-hidden-dim", str(variant_config["attention_hidden_dim"])])

    if variant_config.get("use_channel_latent_attention", False):
        cmd.append("--channel-latent-attention")
        if "attention_hidden_dim" in variant_config:
            cmd.extend(["--attention-hidden-dim", str(variant_config["attention_hidden_dim"])])

    # Add missing data parameters
    if miss_rate > 0:
        cmd.extend([
            "--ucr-missing-rate", str(miss_rate),
            "--ucr-missing-scheme", args.ucr_missing_scheme,
        ])

    # Add optional parameters
    if args.gru_units:
        cmd.extend(["--gru-units", str(args.gru_units)])
    if args.units:
        cmd.extend(["--units", str(args.units)])

    # Execute
    print(f"    [{args.variant}] {dataset} | miss={miss_rate} | seed={seed}")

    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                text=True,
            )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"      ✓ Success ({elapsed_time:.1f}s)")
            return {"status": "success", "time": elapsed_time}
        else:
            print(f"      ✗ Failed (exit {result.returncode})")
            return {"status": "failed", "time": elapsed_time}

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"      ⏱  Timeout ({args.timeout}s)")
        return {"status": "timeout", "time": elapsed_time}

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"      ✗ Error: {e}")
        return {"status": "error", "time": elapsed_time}


def main():
    parser = argparse.ArgumentParser(
        description="UCR Attention Experiments Batch Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Variant selection
    parser.add_argument("--variant", type=str, required=True,
                        choices=list(ATTENTION_CONFIGS.keys()),
                        help="Attention variant to test")

    # Dataset selection
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: all UCR attention datasets)")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Datasets to exclude")

    # Model parameters
    parser.add_argument("--model", type=str, default="sde-rnn",
                        choices=["sde-rnn"],
                        help="Model type (only sde-rnn supported for attention)")
    parser.add_argument("--latents", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--niters", type=int, default=100)
    parser.add_argument("--gru-units", type=int, default=None)
    parser.add_argument("--units", type=int, default=None)

    # Multi-seed configuration
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds to use")
    parser.add_argument("--seed-list", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=42)

    # Missing data
    parser.add_argument("--ucr-missing-rate", type=float, default=0.0)
    parser.add_argument("--ucr-missing-scheme", type=str, default="per-value",
                        choices=["per-value", "per-time", "per-dim"])
    parser.add_argument("--miss-rates", type=float, nargs="+", default=None,
                        help="Multiple missing rates to test (e.g., 0.0 0.3 0.6 0.9)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=3600)

    args = parser.parse_args()

    # Determine datasets
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = UCR_DATASETS.copy()

    datasets = [d for d in datasets if d not in args.exclude]

    if not datasets:
        print("Error: No datasets to run!")
        return 1

    # Determine missing rates
    if args.miss_rates:
        miss_rates = args.miss_rates
    else:
        miss_rates = [args.ucr_missing_rate]

    # Get seeds
    seeds = get_seeds(args)

    # Create output directory
    if args.name:
        exp_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"ucr_{args.variant}_{timestamp}"

    output_dir = args.output_dir if args.output_dir else f"ucr_results_{args.variant}"
    os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    print("=" * 80)
    print(f"UCR ATTENTION EXPERIMENTS: {args.variant.upper()}")
    print("=" * 80)
    print(f"Variant:     {args.variant}")
    print(f"Datasets:    {len(datasets)}")
    print(f"Seeds:       {len(seeds)} {seeds}")
    print(f"Miss rates:  {miss_rates}")
    print(f"Total runs:  {len(datasets) * len(seeds) * len(miss_rates)}")
    print(f"Output:      {output_dir}")
    print("=" * 80)
    print(f"\nDatasets: {', '.join(datasets)}\n")

    # Save configuration
    config_data = {
        "experiment_name": exp_name,
        "variant": args.variant,
        "variant_config": ATTENTION_CONFIGS[args.variant],
        "datasets": datasets,
        "seeds": seeds,
        "miss_rates": miss_rates,
        "arguments": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    # Run experiments
    variant_config = ATTENTION_CONFIGS[args.variant]
    results = []
    total_success = 0
    total_fail = 0

    start_time = time.time()

    for miss_rate in miss_rates:
        print(f"\n{'#' * 80}")
        print(f"# Missing Rate: {miss_rate}")
        print(f"{'#' * 80}")

        for i, dataset in enumerate(datasets, 1):
            print(f"\n  [{i}/{len(datasets)}] {dataset}")

            for seed in seeds:
                result = run_single_experiment(
                    dataset, seed, miss_rate, variant_config, args, output_dir
                )

                if result["status"] == "success":
                    total_success += 1
                else:
                    total_fail += 1

                results.append({
                    "dataset": dataset,
                    "seed": seed,
                    "miss_rate": miss_rate,
                    "variant": args.variant,
                    **result
                })

    total_time = time.time() - start_time

    # Save results
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total time:  {total_time:.1f}s ({total_time / 3600:.2f}h)")
    print(f"Success:     {total_success}")
    print(f"Failed:      {total_fail}")
    print(f"Success rate: {100 * total_success / (total_success + total_fail):.1f}%")
    print(f"\n📂 Results: {output_dir}/")
    print(f"📄 Config:  {config_file}")
    print(f"📊 Results: {results_file}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚡️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)