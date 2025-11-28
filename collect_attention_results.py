#!/usr/bin/env python3
"""
UEA Result collector - Matches log files with metrics files
Supports latent-level attention mechanisms
Uses last 5 evaluations for stability
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

DATASETS = [
    "FingerMovements",
    "ArticularyWordRecognition",
    "BasicMotions",
    "Epilepsy",
    "AtrialFibrillation",
    "SelfRegulationSCP2",
    "Libras",
    "ERing",
    "UWaveGestureLibrary"
]

VARIANTS = ["baseline", "pyramidal_latent", "tvf_lstm_latent", "tvf_transformer_latent", "channel_latent"]
SEEDS = [42, 777, 9999]
MISS_RATES = [0.0, 0.3, 0.6, 0.9]


def extract_seed_and_runcode(filename):
    """Extract seed and run_code from filename"""
    pattern = r'seed(\d+)_(t\d+)'
    match = re.search(pattern, filename)
    if match:
        seed = int(match.group(1))
        run_code = match.group(2)
        return seed, run_code
    return None, None


def parse_log_file(log_path):
    """Parse log file to extract experiment configuration"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        dataset_match = re.search(r'--dataset[=\s]+(["\']?)(\w+)\1', content, re.I)
        if not dataset_match:
            return None

        dataset = dataset_match.group(2)
        if dataset not in DATASETS:
            return None

        # Determine variant based on latent attention flags
        variant = 'baseline'

        if '--pyramidal-latent-attention' in content:
            variant = 'pyramidal_latent'
        elif '--tvf-latent-attention' in content:
            tvf_method_match = re.search(r'--tvf-method[=\s]+(["\']?)(\w+)\1', content, re.I)
            if tvf_method_match:
                method = tvf_method_match.group(2).lower()
                variant = f'tvf_{method}_latent'
            else:
                variant = 'tvf_lstm_latent'  # default
        elif '--channel-latent-attention' in content:
            variant = 'channel_latent'

        miss_rate = 0.0
        miss_rate_match = re.search(r'--uea-miss(?:ing)?-rate[=\s]+(["\']?)([\d.]+)\1', content, re.I)
        if miss_rate_match:
            miss_rate = float(miss_rate_match.group(2))

        return {
            'dataset': dataset,
            'variant': variant,
            'miss_rate': miss_rate
        }

    except Exception as e:
        print(f"    ✗ Error parsing {log_path.name}: {e}")
        return None


def parse_metrics_jsonl(jsonl_path, last_n=5):
    """Parse metrics JSONL - use last N evaluations"""
    try:
        epochs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        epoch_data = json.loads(line)
                        epochs.append(epoch_data)
                    except json.JSONDecodeError:
                        continue

        if not epochs:
            return None

        last_epoch = epochs[-1]

        if 'metrics_hist' not in last_epoch:
            return None

        metrics_hist = last_epoch['metrics_hist']

        if 'accuracy' not in metrics_hist:
            return None

        acc_list = metrics_hist['accuracy']

        if not isinstance(acc_list, list) or len(acc_list) == 0:
            return None

        acc_array = np.array(acc_list)

        if acc_array.max() < 1.0:
            acc_array = acc_array * 100

        n_used = min(last_n, len(acc_array))
        last_n_accs = acc_array[-n_used:]

        return {
            'accuracy_mean': float(last_n_accs.mean()),
            'accuracy_std': float(last_n_accs.std()),
            'accuracy_final': float(acc_array[-1]),
            'n_used': int(n_used)
        }

    except Exception as e:
        print(f"    ✗ Error parsing {jsonl_path.name}: {e}")
        return None


def collect_results(base_dir, last_n=5):
    """Collect results"""
    base_path = Path(base_dir)
    logs_dir = base_path / "logs"

    print("=" * 70)
    print("COLLECTING UEA LATENT ATTENTION RESULTS")
    print(f"Using last {last_n} evaluations")
    print("=" * 70)
    print()

    if not logs_dir.exists():
        print(f"✗ logs/ directory not found")
        return {}, 0

    log_files = list(logs_dir.glob("run_models_seed*.log"))
    metrics_files = list(logs_dir.glob("metrics_run_models_seed*.jsonl"))

    print(f"Found {len(log_files)} log files")
    print(f"Found {len(metrics_files)} metrics files")
    print()

    log_index = {}
    for log_file in log_files:
        seed, run_code = extract_seed_and_runcode(log_file.name)
        if seed and run_code:
            log_index[(seed, run_code)] = log_file

    metrics_index = {}
    for metrics_file in metrics_files:
        seed, run_code = extract_seed_and_runcode(metrics_file.name)
        if seed and run_code:
            metrics_index[(seed, run_code)] = metrics_file

    print(f"✓ Indexed {len(log_index)} log files")
    print(f"✓ Indexed {len(metrics_index)} metrics files")
    print()

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    matched_count = 0
    variant_counts = defaultdict(int)

    for (seed, run_code), log_file in sorted(log_index.items()):
        if (seed, run_code) not in metrics_index:
            continue

        metrics_file = metrics_index[(seed, run_code)]

        log_data = parse_log_file(log_file)
        if not log_data:
            continue

        acc_stats = parse_metrics_jsonl(metrics_file, last_n=last_n)
        if acc_stats is None:
            continue

        dataset = log_data['dataset']
        variant = log_data['variant']
        miss_rate = log_data['miss_rate']

        results[dataset][miss_rate][variant].append({
            'seed': seed,
            'run_code': run_code,
            'accuracy': acc_stats['accuracy_mean'],
            'accuracy_std': acc_stats['accuracy_std'],
            'accuracy_final': acc_stats['accuracy_final'],
            'n_evaluations': acc_stats['n_used'],
            'miss_rate': miss_rate,
            'log_file': log_file.name,
            'metrics_file': metrics_file.name
        })

        matched_count += 1
        variant_counts[variant] += 1

    print(f"✅ Matched {matched_count} experiments")
    for variant, count in sorted(variant_counts.items()):
        print(f"   {variant}: {count}")
    print()

    return results, matched_count


def generate_csvs(results, base_output='attention_results'):
    """Generate multiple CSV files"""

    # 1. All runs detail
    detail_rows = []
    for dataset in DATASETS:
        if dataset not in results:
            continue
        for miss_rate in sorted(results[dataset].keys()):
            for variant in VARIANTS:
                for run in results[dataset][miss_rate].get(variant, []):
                    detail_rows.append({
                        'Dataset': dataset,
                        'Miss_Rate': miss_rate,
                        'Variant': variant,
                        'Seed': run['seed'],
                        'Accuracy_Mean5': run['accuracy'],
                        'Accuracy_Std5': run.get('accuracy_std', 0),
                        'Accuracy_Final': run.get('accuracy_final', run['accuracy']),
                        'Run_Code': run['run_code'],
                        'Log_File': run['log_file']
                    })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(f'{base_output}_all_runs.csv', index=False)
    print(f"✓ Saved: {base_output}_all_runs.csv")

    # 2. Summary by dataset and miss rate
    summary_rows = []
    for dataset in DATASETS:
        if dataset not in results:
            continue
        for miss_rate in sorted(results[dataset].keys()):
            baseline = [r['accuracy'] for r in results[dataset][miss_rate].get('baseline', [])]
            if not baseline:
                continue

            base_mean = np.mean(baseline)
            base_std = np.std(baseline)

            row = {
                'Dataset': dataset,
                'Miss_Rate': miss_rate,
                'baseline_Mean': base_mean,
                'baseline_Std': base_std,
                'baseline_N': len(baseline)
            }

            for var in ['pyramidal_latent', 'tvf_lstm_latent', 'tvf_transformer_latent', 'channel_latent']:
                accs = [r['accuracy'] for r in results[dataset][miss_rate].get(var, [])]
                if accs:
                    mean = np.mean(accs)
                    std = np.std(accs)
                    imp_abs = mean - base_mean
                    imp_pct = (imp_abs / base_mean) * 100 if base_mean > 0 else 0

                    row[f'{var}_Mean'] = mean
                    row[f'{var}_Std'] = std
                    row[f'{var}_N'] = len(accs)
                    row[f'{var}_Imp_Abs'] = imp_abs
                    row[f'{var}_Imp_Pct'] = imp_pct
                else:
                    row[f'{var}_Mean'] = np.nan
                    row[f'{var}_Std'] = np.nan
                    row[f'{var}_N'] = 0
                    row[f'{var}_Imp_Abs'] = np.nan
                    row[f'{var}_Imp_Pct'] = np.nan

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{base_output}_summary_by_dataset.csv', index=False)
    print(f"✓ Saved: {base_output}_summary_by_dataset.csv")

    # 3. Pivot table: Dataset x Variant (for each miss rate)
    for miss_rate in MISS_RATES:
        pivot_rows = []
        for dataset in DATASETS:
            if dataset not in results or miss_rate not in results[dataset]:
                continue

            row = {'Dataset': dataset}
            for variant in VARIANTS:
                runs = results[dataset][miss_rate].get(variant, [])
                if runs:
                    accs = [r['accuracy'] for r in runs]
                    row[f'{variant}_mean'] = np.mean(accs)
                    row[f'{variant}_std'] = np.std(accs)
                else:
                    row[f'{variant}_mean'] = np.nan
                    row[f'{variant}_std'] = np.nan

            pivot_rows.append(row)

        if pivot_rows:
            pivot_df = pd.DataFrame(pivot_rows)
            pivot_df.to_csv(f'{base_output}_pivot_miss{miss_rate}.csv', index=False)
            print(f"✓ Saved: {base_output}_pivot_miss{miss_rate}.csv")

    return summary_df


def print_summary(results):
    """Print summary"""
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    for dataset in DATASETS:
        if dataset not in results:
            continue

        print(f"{dataset}:")
        for miss_rate in sorted(results[dataset].keys()):
            print(f"  Miss Rate: {miss_rate:.1f}")
            for variant in VARIANTS:
                runs = results[dataset][miss_rate].get(variant, [])
                if runs:
                    accs = [r['accuracy'] for r in runs]
                    print(f"    {variant:20s}: {np.mean(accs):5.2f}% ± {np.std(accs):4.2f}% (n={len(accs)})")
        print()


def main():
    import sys

    base_dir = sys.argv[1] if len(sys.argv) > 1 else './'
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'attention_results'

    results, matched_count = collect_results(base_dir, last_n=5)

    if matched_count == 0:
        print("✗ No results found!")
        return

    print_summary(results)

    print("=" * 70)
    print("GENERATING CSV FILES")
    print("=" * 70)
    generate_csvs(results, output_prefix)
    print("=" * 70)
    print("✅ Done!")


if __name__ == "__main__":
    main()