#!/usr/bin/env python3
"""
Plotting script for UCR experiments with latent attention
Compatible with UCR results pivot tables

Usage:
    python plot_ucr_results.py --input ucr_collected_results/ --output plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# UCR datasets
SELECTED_DATASETS = [
    'Wafer',
    # 'ItalyPowerDemand',
    'ProximalPhalanxOutlineAgeGroup',
    # 'SyntheticControl',
     'ProximalPhalanxTW',
     # 'SmoothSubspace',
    'MoteStrain',
    'Earthquakes',
    'ProximalPhalanxOutlineCorrect',
    'SonyAIBORobotSurface2',
    'Strawberry',
    # 'ElectricDevices',
    # 'ECGFiveDays',
     'TwoPatterns',
    'MiddlePhalanxOutlineAgeGroup',
    # 'Car', 'Lightning2', 'Coffee'
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


def load_results(base_path='./ucr_collected_results'):
    """Load results from CSV files"""
    results = {}
    base_path = Path(base_path)

    print(f"Looking for data in: {base_path.absolute()}")

    for miss_rate in MISSING_RATES:
        pivot_file = base_path / f'ucr_collected_results_pivot_miss{miss_rate}.csv'

        print(f"  Checking: {pivot_file.name} ... ", end='')
        if pivot_file.exists():
            print("✓")
            df = pd.read_csv(pivot_file)
            df = df[df['Dataset'].isin(SELECTED_DATASETS)]
            results[miss_rate] = df
        else:
            print("✗ NOT FOUND")

    return results


def plot_dataset(datasets_data, dataset_name, ax):
    """Plot accuracy curves for one dataset across all missing rates"""
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
    ax.set_ylim(20, 100)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


def create_plot(results, output_dir='plots'):
    """Create multi-panel plot with all datasets"""
    datasets = [d for d in SELECTED_DATASETS if d in results[0.0]['Dataset'].values]
    n_datasets = len(datasets)

    print(f"Plotting {n_datasets} datasets...")

    # Layout
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

    # Get handles and labels
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

    png_file = output_path / 'ucr_latent_attention_comparison.png'
    pdf_file = output_path / 'ucr_latent_attention_comparison.pdf'

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {png_file}")
    print(f"✓ Saved: {pdf_file}")


def create_summary_table(results, output_dir='plots'):
    """Create a summary table comparing all variants"""

    summary_data = []

    for miss_rate in MISSING_RATES:
        if miss_rate not in results:
            continue

        df = results[miss_rate]

        for variant in ['baseline', 'pyramidal_latent', 'tvf_lstm_latent', 'tvf_transformer_latent', 'channel_latent']:
            mean_col = f'{variant}_mean'

            if mean_col in df.columns:
                means = df[mean_col].dropna()
                if len(means) > 0:
                    summary_data.append({
                        'Miss_Rate': miss_rate,
                        'Variant': variant,
                        'Mean_Accuracy': means.mean(),
                        'Min_Accuracy': means.min(),
                        'Max_Accuracy': means.max(),
                        'N_Datasets': len(means)
                    })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Pivot for easier viewing
        pivot = summary_df.pivot_table(
            values='Mean_Accuracy',
            index='Variant',
            columns='Miss_Rate'
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_file = output_path / 'ucr_summary_table.csv'
        pivot.to_csv(summary_file)
        print(f"✓ Saved: {summary_file}")

        # Print to console
        print("\n" + "=" * 70)
        print("SUMMARY: Mean Accuracy Across All Datasets")
        print("=" * 70)
        print(pivot.to_string())
        print()


def main():
    parser = argparse.ArgumentParser(description='Plot UCR latent attention experiment results')
    parser.add_argument('--input', type=str, default='./ucr_collected_results',
                        help='Input directory with pivot CSV files')
    parser.add_argument('--output', type=str, default='./plots',
                        help='Output directory for plots')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to plot (default: all)')

    args = parser.parse_args()

    # Override SELECTED_DATASETS if specified
    global SELECTED_DATASETS
    if args.datasets:
        SELECTED_DATASETS = args.datasets

    print("=" * 70)
    print("UCR LATENT ATTENTION EXPERIMENT VISUALIZATION")
    print("=" * 70)
    print()

    results = load_results(base_path=args.input)

    if not results:
        print("\n✗ Error: No data files found!")
        print(f"\nExpected files in {args.input}:")
        for miss_rate in MISSING_RATES:
            print(f"  - ucr_results_pivot_miss{miss_rate}.csv")
        return

    print(f"✓ Loaded data for {len(SELECTED_DATASETS)} datasets")
    print()

    # Create plots
    create_plot(results, output_dir=args.output)

    # Create summary table
    create_summary_table(results, output_dir=args.output)

    print("\n✓ Complete - outputs saved to", args.output)


if __name__ == '__main__':
    main()