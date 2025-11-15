"""
Analyze and compare benchmark results
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")


def load_experiment_metrics(experiment_dir: Path):
    """Load metrics from an experiment"""
    logs_dir = experiment_dir / "logs"

    if not logs_dir.exists():
        return None

    all_metrics = []

    # Load metrics from all ranks
    for metrics_file in logs_dir.glob("rank_*_metrics.json"):
        with open(metrics_file, 'r') as f:
            rank_metrics = json.load(f)
            all_metrics.extend(rank_metrics)

    if not all_metrics:
        return None

    return pd.DataFrame(all_metrics)


def compare_experiments(results_dir: Path, output_dir: Path):
    """Compare all experiments in results directory"""

    experiments = {}

    # Load all experiments
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics = load_experiment_metrics(exp_dir)
        if metrics is not None:
            experiments[exp_dir.name] = metrics

    if not experiments:
        print("No experiments found!")
        return

    print(f"Found {len(experiments)} experiments")

    # Calculate summary statistics
    summaries = []

    for name, df in experiments.items():
        summary = {
            'experiment': name,
            'avg_throughput': df['throughput'].mean(),
            'avg_loss': df['loss'].mean(),
            'avg_iteration_time': df['iteration_time'].mean(),
            'avg_gpu_utilization': df['gpu_utilization'].mean(),
            'avg_gpu_memory': df['gpu_memory_percent'].mean(),
            'total_iterations': len(df),
        }
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Create comparison plots
    create_comparison_plots(experiments, summary_df, output_dir)

    return summary_df


def create_comparison_plots(experiments, summary_df, output_dir):
    """Create comparison plots"""

    # 1. Throughput comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_df_sorted = summary_df.sort_values('avg_throughput', ascending=False)
    bars = ax.bar(range(len(summary_df_sorted)), summary_df_sorted['avg_throughput'])

    # Color bars by policy
    colors = []
    for name in summary_df_sorted['experiment']:
        if 'baseline' in name:
            colors.append('gray')
        elif 'proportional' in name:
            colors.append('blue')
        elif 'dynamic' in name:
            colors.append('green')
        else:
            colors.append('orange')

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticks(range(len(summary_df_sorted)))
    ax.set_xticklabels(summary_df_sorted['experiment'], rotation=45, ha='right')
    ax.set_ylabel('Throughput (samples/s)')
    ax.set_title('Training Throughput Comparison')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. GPU Utilization comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_df_sorted = summary_df.sort_values('avg_gpu_utilization', ascending=False)
    bars = ax.bar(range(len(summary_df_sorted)), summary_df_sorted['avg_gpu_utilization'])

    colors = []
    for name in summary_df_sorted['experiment']:
        if 'baseline' in name:
            colors.append('gray')
        elif 'proportional' in name:
            colors.append('blue')
        elif 'dynamic' in name:
            colors.append('green')
        else:
            colors.append('orange')

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticks(range(len(summary_df_sorted)))
    ax.set_xticklabels(summary_df_sorted['experiment'], rotation=45, ha='right')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU Utilization Comparison')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'gpu_utilization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Iteration time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_df_sorted = summary_df.sort_values('avg_iteration_time')
    bars = ax.bar(range(len(summary_df_sorted)), summary_df_sorted['avg_iteration_time'])

    colors = []
    for name in summary_df_sorted['experiment']:
        if 'baseline' in name:
            colors.append('gray')
        elif 'proportional' in name:
            colors.append('blue')
        elif 'dynamic' in name:
            colors.append('green')
        else:
            colors.append('orange')

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xticks(range(len(summary_df_sorted)))
    ax.set_xticklabels(summary_df_sorted['experiment'], rotation=45, ha='right')
    ax.set_ylabel('Iteration Time (s)')
    ax.set_title('Iteration Time Comparison')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'iteration_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Loss curves
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, df in experiments.items():
        # Calculate moving average
        window = 50
        loss_ma = df.groupby('iteration')['loss'].mean().rolling(window=window, min_periods=1).mean()

        label = name.replace('_', ' ').title()
        ax.plot(loss_ma.index, loss_ma.values, label=label, linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {output_dir}")


def calculate_speedup(summary_df):
    """Calculate speedup over baseline"""

    # Find baseline experiments
    baseline_experiments = summary_df[summary_df['experiment'].str.contains('baseline')]

    if baseline_experiments.empty:
        print("No baseline experiments found")
        return

    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS")
    print("="*80)

    for _, baseline in baseline_experiments.iterrows():
        baseline_name = baseline['experiment']
        model = baseline_name.split('_')[1] if len(baseline_name.split('_')) > 1 else 'unknown'

        print(f"\nModel: {model}")
        print(f"Baseline: {baseline_name}")
        print(f"  Throughput: {baseline['avg_throughput']:.2f} samples/s")
        print(f"  Iteration Time: {baseline['avg_iteration_time']:.3f}s")

        # Find corresponding experiments
        related = summary_df[
            (summary_df['experiment'].str.contains(model)) &
            (~summary_df['experiment'].str.contains('baseline'))
        ]

        for _, exp in related.iterrows():
            speedup = exp['avg_throughput'] / baseline['avg_throughput']
            time_reduction = (baseline['avg_iteration_time'] - exp['avg_iteration_time']) / baseline['avg_iteration_time'] * 100

            print(f"\n  {exp['experiment']}:")
            print(f"    Throughput: {exp['avg_throughput']:.2f} samples/s (speedup: {speedup:.2f}x)")
            print(f"    Iteration Time: {exp['avg_iteration_time']:.3f}s (reduction: {time_reduction:.1f}%)")
            print(f"    GPU Utilization: {exp['avg_gpu_utilization']:.1f}%")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument('--input-dir', type=str, default='experiments/benchmarks',
                       help='Input directory with benchmark results')
    parser.add_argument('--output-dir', type=str, default='experiments/analysis',
                       help='Output directory for analysis')

    args = parser.parse_args()

    results_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Compare experiments
    summary_df = compare_experiments(results_dir, output_dir)

    if summary_df is not None:
        # Calculate speedup
        calculate_speedup(summary_df)

        print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
