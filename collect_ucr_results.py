#!/usr/bin/env python3
"""
UCR Result collector V3 - Matches via checkpoint experimentID
Flow: log file → checkpoint file (extract timestamp) → logs/metrics file
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# UCR dataset names
UCR_DATASETS = [
    "Wafer", "ItalyPowerDemand", "ProximalPhalanxOutlineAgeGroup",
    "SyntheticControl", "ProximalPhalanxTW", "SmoothSubspace",
    "MoteStrain", "Earthquakes", "ProximalPhalanxOutlineCorrect",
    "SonyAIBORobotSurface2", "Strawberry", "ElectricDevices",
    "ECGFiveDays", "TwoPatterns", "MiddlePhalanxOutlineAgeGroup",
    "Car", "Lightning2", "Coffee"
]

VARIANTS = ["baseline", "pyramidal_latent", "tvf_lstm_latent", "tvf_transformer_latent", "channel_latent"]


def extract_seed_from_filename(filename):
    """Extract seed from filename"""
    pattern = r'seed(\d+)'
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None


def extract_experimentID_from_ckpt(ckpt_filename):
    """
    Extract experimentID (timestamp) from checkpoint filename
    Format: experiment_seed42_t17639994593.ckpt
    """
    pattern = r'_t(\d+)\.ckpt'
    match = re.search(pattern, ckpt_filename)
    return match.group(1) if match else None


def extract_variant_from_filename(filename_or_path):
    """
    Extract variant from filename or directory name
    Examples:
        pyramidal_latent_Car_miss0.0_seed42.log → pyramidal_latent
        tvf_lstm_latent_Water_miss0.3_seed777.log → tvf_lstm_latent
        baseline_Coffee_miss0.0_seed42.log → baseline
    """
    name = str(filename_or_path).lower()

    # Check in order of specificity
    if 'tvf_transformer_latent' in name:
        return 'tvf_transformer_latent'
    elif 'tvf_lstm_latent' in name:
        return 'tvf_lstm_latent'
    elif 'pyramidal_latent' in name:
        return 'pyramidal_latent'
    elif 'channel_latent' in name:
        return 'channel_latent'
    elif 'baseline' in name:
        return 'baseline'

    return None


def parse_log_file(log_path):
    """Parse log file to extract experiment configuration"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract dataset
        dataset_match = re.search(r'--ucr-dataset[=\s]+(["\']?)(\w+)\1', content, re.I)
        if not dataset_match:
            return None

        dataset = dataset_match.group(2)
        if dataset not in UCR_DATASETS:
            return None

        # Strategy 1: Try to determine variant from log content (command-line flags)
        variant_from_log = None

        if '--pyramidal-latent-attention' in content:
            variant_from_log = 'pyramidal_latent'
        elif '--tvf-latent-attention' in content:
            tvf_method_match = re.search(r'--tvf-method[=\s]+(["\']?)(\w+)\1', content, re.I)
            if tvf_method_match:
                method = tvf_method_match.group(2).lower()
                variant_from_log = f'tvf_{method}_latent'
            else:
                variant_from_log = 'tvf_lstm_latent'  # default
        elif '--channel-latent-attention' in content:
            variant_from_log = 'channel_latent'
        elif '--sde-rnn' in content and not any(flag in content for flag in ['--pyramidal', '--tvf', '--channel']):
            variant_from_log = 'baseline'

        # Strategy 2: Extract from filename/directory name as fallback
        variant_from_filename = extract_variant_from_filename(log_path.name)
        if not variant_from_filename:
            variant_from_filename = extract_variant_from_filename(log_path.parent.name)

        # Use filename variant as priority (more reliable)
        variant = variant_from_filename if variant_from_filename else (variant_from_log or 'baseline')

        # Extract missing rate
        miss_rate = 0.0
        miss_rate_match = re.search(r'--ucr-missing-rate[=\s]+(["\']?)([\d.]+)\1', content, re.I)
        if miss_rate_match:
            miss_rate = float(miss_rate_match.group(2))
        else:
            # Fallback: extract from filename
            miss_match = re.search(r'miss([\d.]+)', log_path.name)
            if miss_match:
                miss_rate = float(miss_match.group(1))

        return {
            'dataset': dataset,
            'variant': variant,
            'miss_rate': miss_rate,
            'variant_from_log': variant_from_log,
            'variant_from_filename': variant_from_filename
        }

    except Exception as e:
        print(f"    ✗ Error parsing {log_path.name}: {e}")
        return None


def find_checkpoint_in_same_dir(log_file):
    """
    Find checkpoint file in the same directory as log file
    Returns: (checkpoint_path, experimentID) or (None, None)
    """
    log_dir = log_file.parent

    # Search for .ckpt files in same directory
    ckpt_files = list(log_dir.glob("*.ckpt"))

    if not ckpt_files:
        return None, None

    # Extract experimentID from each checkpoint
    for ckpt_file in ckpt_files:
        exp_id = extract_experimentID_from_ckpt(ckpt_file.name)
        if exp_id:
            return ckpt_file, exp_id

    return None, None


def find_metrics_file_in_logs(seed, exp_id, logs_dir):
    """
    Find metrics file in logs/ directory
    Format: metrics_run_models_seed{seed}_t{exp_id}.jsonl
    """
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        return None

    # Construct expected filename
    metrics_filename = f"metrics_run_models_seed{seed}_t{exp_id}.jsonl"
    metrics_path = logs_path / metrics_filename

    if metrics_path.exists():
        return metrics_path

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

        # Convert to percentage if needed
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


def collect_results_from_directory(base_dir, logs_dir, last_n=5):
    """
    Collect results from directory structure

    Args:
        base_dir: Directory containing ucr_results/
        logs_dir: Directory containing logs/ (parallel to ucr_results)
        last_n: Number of last evaluations to average
    """
    base_path = Path(base_dir)
    logs_path = Path(logs_dir)

    print("=" * 80)
    print("COLLECTING UCR LATENT ATTENTION RESULTS V3")
    print("Matching via checkpoint experimentID")
    print(f"Using last {last_n} evaluations")
    print("=" * 80)
    print(f"Results directory: {base_path.resolve()}")
    print(f"Logs directory:    {logs_path.resolve()}")
    print("=" * 80)
    print()

    if not logs_path.exists():
        print(f"❌ ERROR: logs directory not found at {logs_path.resolve()}")
        print(f"   Please check the path!")
        return {}, 0

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    matched_count = 0
    variant_counts = defaultdict(int)
    failed_matches = []

    # Recursively find all log files
    log_files = list(base_path.rglob("*.log"))

    print(f"Found {len(log_files)} log files")

    # Count checkpoint files for verification
    ckpt_files = list(base_path.rglob("*.ckpt"))
    print(f"Found {len(ckpt_files)} checkpoint files")

    # Count metrics files
    metrics_files = list(logs_path.glob("metrics_*.jsonl"))
    print(f"Found {len(metrics_files)} metrics files in logs/")
    print()

    # Process log files
    for i, log_file in enumerate(log_files):
        seed = extract_seed_from_filename(log_file.name)
        if not seed:
            failed_matches.append((log_file.name, "No seed in filename"))
            continue

        # Parse log file for configuration
        log_data = parse_log_file(log_file)
        if not log_data:
            failed_matches.append((log_file.name, "Failed to parse log"))
            continue

        dataset = log_data['dataset']
        variant = log_data['variant']
        miss_rate = log_data['miss_rate']

        # Find checkpoint in same directory
        ckpt_file, exp_id = find_checkpoint_in_same_dir(log_file)
        if not ckpt_file or not exp_id:
            failed_matches.append((log_file.name, "No checkpoint found"))
            continue

        # Debug output for first few matches
        if matched_count < 5:
            print(f"[{i + 1}/{len(log_files)}] Processing: {log_file.name}")
            print(f"  📁 Directory: {log_file.parent.name}")
            print(f"  📊 Dataset: {dataset}, Variant: {variant}, Miss: {miss_rate}, Seed: {seed}")
            if 'variant_from_filename' in log_data and 'variant_from_log' in log_data:
                print(
                    f"  🏷️  Variant source: filename={log_data['variant_from_filename']}, log={log_data['variant_from_log']}")
            print(f"  🎯 Checkpoint: {ckpt_file.name}")
            print(f"  🔢 ExperimentID: {exp_id}")

        # Find metrics file in logs/
        metrics_file = find_metrics_file_in_logs(seed, exp_id, logs_path)

        if not metrics_file:
            failed_matches.append((log_file.name, f"No metrics: seed{seed}_t{exp_id}"))
            if matched_count < 5:
                print(f"  ❌ Metrics file not found: metrics_run_models_seed{seed}_t{exp_id}.jsonl")
                print()
            continue

        if matched_count < 5:
            print(f"  ✅ Metrics: {metrics_file.name}")

        # Parse metrics file
        acc_stats = parse_metrics_jsonl(metrics_file, last_n=last_n)
        if acc_stats is None:
            failed_matches.append((log_file.name, f"Failed to parse metrics"))
            if matched_count < 5:
                print(f"  ❌ Failed to parse metrics")
                print()
            continue

        if matched_count < 5:
            print(f"  📈 Accuracy: {acc_stats['accuracy_mean']:.2f}% ± {acc_stats['accuracy_std']:.2f}%")
            print()

        results[dataset][miss_rate][variant].append({
            'seed': seed,
            'accuracy': acc_stats['accuracy_mean'],
            'accuracy_std': acc_stats['accuracy_std'],
            'accuracy_final': acc_stats['accuracy_final'],
            'n_evaluations': acc_stats['n_used'],
            'log_file': str(log_file.relative_to(base_path)),
            'metrics_file': str(metrics_file.relative_to(logs_path)),
            'exp_id': exp_id
        })

        matched_count += 1
        variant_counts[variant] += 1

    print()
    print("=" * 80)
    print("MATCHING RESULTS")
    print("=" * 80)
    print(f"✅ Successfully matched: {matched_count} experiments")
    for variant, count in sorted(variant_counts.items()):
        print(f"   {variant:25s}: {count}")
    print()

    if failed_matches:
        print(f"⚠️  Failed to match: {len(failed_matches)} log files")
        print("   Top reasons:")
        reason_counts = defaultdict(int)
        for _, reason in failed_matches:
            reason_counts[reason] += 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"   - {reason}: {count}")
        print()

    # Sanity check: are all accuracies the same?
    all_accuracies = []
    for dataset_data in results.values():
        for miss_data in dataset_data.values():
            for variant_runs in miss_data.values():
                for run in variant_runs:
                    all_accuracies.append(run['accuracy'])

    if all_accuracies:
        unique_accs = set(f"{acc:.2f}" for acc in all_accuracies)
        if len(unique_accs) == 1:
            print("⚠️  WARNING: All experiments have the SAME accuracy!")
            print(f"   All results = {all_accuracies[0]:.2f}%")
            print()
        elif len(unique_accs) <= 3:
            print(f"⚠️  WARNING: Only {len(unique_accs)} unique accuracy values found")
            print(f"   Values: {sorted(set(f'{acc:.2f}' for acc in all_accuracies))}")
            print()
        else:
            print(f"✅ Found {len(unique_accs)} unique accuracy values")
            print(f"   Range: {min(all_accuracies):.2f}% - {max(all_accuracies):.2f}%")
            print()

    return results, matched_count


def generate_csvs(results, base_output='ucr_results'):
    """Generate CSV files with results"""

    # 1. All runs detail
    detail_rows = []
    for dataset in UCR_DATASETS:
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
                        'ExperimentID': run.get('exp_id', ''),
                        'Accuracy_Mean5': run['accuracy'],
                        'Accuracy_Std5': run.get('accuracy_std', 0),
                        'Accuracy_Final': run.get('accuracy_final', run['accuracy']),
                        'N_Evaluations': run.get('n_evaluations', 0),
                        'Log_File': run['log_file'],
                        'Metrics_File': run.get('metrics_file', '')
                    })

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(f'{base_output}_all_runs.csv', index=False)
        print(f"✓ Saved: {base_output}_all_runs.csv ({len(detail_rows)} rows)")
    else:
        print(f"⚠️  No data to save!")
        return

    # 2. Summary by dataset and miss rate
    summary_rows = []
    for dataset in UCR_DATASETS:
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

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f'{base_output}_summary_by_dataset.csv', index=False)
        print(f"✓ Saved: {base_output}_summary_by_dataset.csv")

    # 3. Pivot tables
    miss_rates_found = set()
    for dataset_data in results.values():
        miss_rates_found.update(dataset_data.keys())

    for miss_rate in sorted(miss_rates_found):
        pivot_rows = []
        for dataset in UCR_DATASETS:
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


def print_summary(results):
    """Print summary to terminal"""
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    for dataset in UCR_DATASETS:
        if dataset not in results:
            continue

        print(f"{dataset}:")
        for miss_rate in sorted(results[dataset].keys()):
            print(f"  Miss Rate: {miss_rate:.1f}")
            for variant in VARIANTS:
                runs = results[dataset][miss_rate].get(variant, [])
                if runs:
                    accs = [r['accuracy'] for r in runs]
                    print(f"    {variant:25s}: {np.mean(accs):5.2f}% ± {np.std(accs):4.2f}% (n={len(accs)})")
        print()


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python collect_ucr_results_v3.py <results_dir> <logs_dir> [output_prefix]")
        print()
        print("Example:")
        print("  python collect_ucr_results_v3.py ucr_results/ logs/ ucr_collected")
        print()
        return

    base_dir = sys.argv[1]
    logs_dir = sys.argv[2]
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else 'ucr_results'

    results, matched_count = collect_results_from_directory(base_dir, logs_dir, last_n=5)

    if matched_count == 0:
        print("=" * 80)
        print("❌ NO RESULTS FOUND!")
        print("=" * 80)
        print()
        print("Troubleshooting checklist:")
        print("1. Check directory structure:")
        print(f"   ls {base_dir}/")
        print(f"   ls {logs_dir}/")
        print()
        print("2. Check for checkpoint files:")
        print(f"   find {base_dir} -name '*.ckpt' | head -5")
        print()
        print("3. Check for metrics files:")
        print(f"   ls {logs_dir}/metrics_*.jsonl | head -5")
        print()
        print("4. Check log file format:")
        print(f"   head -50 $(find {base_dir} -name '*.log' | head -1)")
        return

    print_summary(results)

    print("=" * 80)
    print("GENERATING CSV FILES")
    print("=" * 80)
    generate_csvs(results, output_prefix)
    print("=" * 80)
    print("✅ DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()