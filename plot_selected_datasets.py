#!/usr/bin/env python3
"""
Plotting script for UEA latent attention experiments
Data from uea_results folder, shared legend, optimized layout
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SELECTED_DATASETS = [
    'FingerMovements',
    'Epilepsy',
    'SelfRegulationSCP2',
    'BasicMotions',
    'Libras',
    'ERing',
    'ArticularyWordRecognition',
    'UWaveGestureLibrary'
]

MISSING_RATES = [0.0, 0.3, 0.6, 0.9]

COLORS = {
    'baseline': '#1f77b4',
    'pyramidal_latent': '#2ca02c',
    'tvf_lstm_latent': '#9467bd',
    'tvf_transformer_latent': '#8c564b',
    'channel_latent': '#e377c2'
}

MODEL_LABELS = {
    'baseline': 'SDE-RNN',
    'pyramidal_latent': 'SDE-PYR',
    'tvf_lstm_latent': 'SDE-TVF-L',
    'tvf_transformer_latent': 'SDE-TVF-T',
    'channel_latent': 'SDE-SCHA'
}


def compute_mean_accuracy_by_missrate(results):
    """
    Compute mean accuracy across all datasets for each missing rate and variant

    Returns:
        DataFrame with columns: Missing_Rate, Variant, Mean_Accuracy, Std_Accuracy, N_Datasets
    """
    summary_data = []

    variants = ['baseline', 'pyramidal_latent', 'tvf_lstm_latent', 'tvf_transformer_latent', 'channel_latent']

    for miss_rate in MISSING_RATES:
        if miss_rate not in results:
            continue

        df = results[miss_rate]

        for variant in variants:
            mean_col = f'{variant}_mean'

            if mean_col in df.columns:
                # Extract all valid accuracy values for this variant
                accuracies = df[mean_col].dropna().values

                if len(accuracies) > 0:
                    summary_data.append({
                        'Missing_Rate': miss_rate,
                        'Variant': variant,
                        'Mean_Accuracy': np.mean(accuracies),
                        'Std_Accuracy': np.std(accuracies),
                        'N_Datasets': len(accuracies)
                    })

    return pd.DataFrame(summary_data)


def print_mean_accuracy_table(summary_df):
    """Print formatted table of mean accuracies"""
    print("\n" + "=" * 80)
    print("MEAN ACCURACY BY MISSING RATE AND VARIANT")
    print("(Averaged across all selected datasets)")
    print("=" * 80)
    print()

    # Pivot table for better display
    pivot = summary_df.pivot(index='Variant', columns='Missing_Rate', values='Mean_Accuracy')

    # Print header
    print(f"{'Variant':<30s}", end='')
    for miss_rate in sorted(pivot.columns):
        print(f"{f'Miss={miss_rate:.1f}':>12s}", end='')
    print()
    print("-" * 80)

    # Print each variant
    for variant in ['baseline', 'pyramidal_latent', 'tvf_lstm_latent', 'tvf_transformer_latent', 'channel_latent']:
        if variant in pivot.index:
            print(f"{MODEL_LABELS[variant]:<30s}", end='')
            for miss_rate in sorted(pivot.columns):
                if not pd.isna(pivot.loc[variant, miss_rate]):
                    print(f"{pivot.loc[variant, miss_rate]:>11.2f}%", end='')
                else:
                    print(f"{'N/A':>12s}", end='')
            print()

    print()

    # Print improvements over baseline
    print("=" * 80)
    print("IMPROVEMENT OVER BASELINE (percentage points)")
    print("=" * 80)
    print()

    baseline_row = pivot.loc['baseline'] if 'baseline' in pivot.index else None

    if baseline_row is not None:
        print(f"{'Variant':<30s}", end='')
        for miss_rate in sorted(pivot.columns):
            print(f"{f'Miss={miss_rate:.1f}':>12s}", end='')
        print()
        print("-" * 80)

        for variant in ['pyramidal_latent', 'tvf_lstm_latent', 'tvf_transformer_latent', 'channel_latent']:
            if variant in pivot.index:
                print(f"{MODEL_LABELS[variant]:<30s}", end='')
                for miss_rate in sorted(pivot.columns):
                    if not pd.isna(pivot.loc[variant, miss_rate]) and not pd.isna(baseline_row[miss_rate]):
                        improvement = pivot.loc[variant, miss_rate] - baseline_row[miss_rate]
                        sign = '+' if improvement >= 0 else ''
                        print(f"{sign}{improvement:>10.2f}pp", end='')
                    else:
                        print(f"{'N/A':>12s}", end='')
                print()

    print()


def save_mean_accuracy_csv(summary_df, output_dir='plots'):
    """Save mean accuracy summary to CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / 'uea_mean_accuracy_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"✓ Saved mean accuracy summary: {csv_file}")

    # Also save pivot table
    pivot = summary_df.pivot(index='Variant', columns='Missing_Rate', values='Mean_Accuracy')
    pivot_file = output_path / 'uea_mean_accuracy_pivot.csv'
    pivot.to_csv(pivot_file)
    print(f"✓ Saved pivot table: {pivot_file}")


def load_results(base_path='./uea_results'):
    """Load results from CSV files

    Args:
        base_path: Directory containing the CSV files (default: ./uea_results)
    """
    results = {}
    base_path = Path(base_path)

    print(f"Looking for data in: {base_path.absolute()}")

    for miss_rate in MISSING_RATES:
        attn_file = base_path / f'attention_results_pivot_miss{miss_rate}.csv'

        print(f"  Checking: {attn_file.name} ... ", end='')
        if attn_file.exists():
            print("✓")
            df = pd.read_csv(attn_file)
            df = df[df['Dataset'].isin(SELECTED_DATASETS)]
            results[miss_rate] = df
        else:
            print("✗ NOT FOUND")

    return results


def plot_dataset(datasets_data, dataset_name, ax):
    models = {
        'baseline': 'baseline_mean',
        'pyramidal_latent': 'pyramidal_latent_mean',
        'tvf_lstm_latent': 'tvf_lstm_latent_mean',
        'tvf_transformer_latent': 'tvf_transformer_latent_mean',
        'channel_latent': 'channel_latent_mean',
    }

    x_positions = [0, 30, 60, 90]

    for model_name, mean_col in models.items():
        means = []
        valid_x = []

        for i, miss_rate in enumerate(MISSING_RATES):
            if miss_rate in datasets_data:
                df = datasets_data[miss_rate]
                if mean_col in df.columns:
                    value = df.loc[df['Dataset'] == dataset_name, mean_col]
                    if not value.empty and not pd.isna(value.iloc[0]):
                        means.append(value.iloc[0])
                        valid_x.append(x_positions[i])

        if means:
            ax.plot(valid_x, means, marker='o',
                    label=MODEL_LABELS[model_name],
                    color=COLORS[model_name],
                    linewidth=2.5, markersize=7)

    ax.set_xlabel('Missing rate', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title(dataset_name, fontsize=11, fontweight='bold')
    ax.set_xticks([0, 30, 60, 90])
    ax.set_xticklabels(['0%', '30%', '60%', '90%'], fontsize=9)
    ax.set_ylim(0, 80.0)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


def create_plot(results, output_dir='plots'):
    datasets = [d for d in SELECTED_DATASETS if d in results[0.0]['Dataset'].values]
    n_datasets = len(datasets)

    print(f"Plotting {n_datasets} datasets...")

    # Adjusted layout - more square
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))

    # Create figure with extra space for legend
    fig = plt.figure(figsize=(16, 4))

    # Create grid: main plots + legend space
    gs = fig.add_gridspec(n_rows + 1, n_cols,
                          height_ratios=[1] * n_rows + [0.15],
                          hspace=0.35, wspace=0.3)

    # Plot datasets
    for idx, dataset in enumerate(datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        plot_dataset(results, dataset, ax)

    # Hide unused subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')

    # Create shared legend at bottom
    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis('off')

    # Get handles and labels from first subplot
    handles = []
    labels = []
    for model_name in ['baseline', 'pyramidal_latent', 'tvf_lstm_latent', 'tvf_transformer_latent', 'channel_latent']:
        line = plt.Line2D([0], [0], color=COLORS[model_name], linewidth=3, marker='o', markersize=8)
        handles.append(line)
        labels.append(MODEL_LABELS[model_name])

    legend = legend_ax.legend(handles, labels,
                              loc='center',
                              ncol=5,
                              fontsize=12,
                              frameon=True,
                              fancybox=True,
                              shadow=True)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    png_file = output_path / 'uea_latent_attention_comparison.png'
    pdf_file = output_path / 'uea_latent_attention_comparison.pdf'

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")


def main():
    print("=" * 60)
    print("UEA - Latent Attention Visualization")
    print("=" * 60)

    results = load_results(base_path='./uea_results')

    if not results:
        print("\n✗ Error: No data files found!")
        print(f"\nExpected files in current directory:")
        for miss_rate in MISSING_RATES:
            print(f"  - attention_results_pivot_miss{miss_rate}.csv")
        return

    print(f"✓ Loaded {len(SELECTED_DATASETS)} datasets")

    # Compute mean accuracy across datasets
    print("\nComputing mean accuracy statistics...")
    summary_df = compute_mean_accuracy_by_missrate(results)

    # Print statistics table
    print_mean_accuracy_table(summary_df)

    # Save CSV files
    save_mean_accuracy_csv(summary_df, output_dir='plots')

    # Create plots
    print("\nGenerating plots...")
    create_plot(results, output_dir='plots')

    print("\n✓ Complete - plots and statistics saved to plots/")


if __name__ == '__main__':
    main()