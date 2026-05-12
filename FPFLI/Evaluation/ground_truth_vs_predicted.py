#!/usr/bin/env python3
"""
Ground Truth vs Predicted Lifetime Analysis
Creates scatter plots and correlation analysis comparing LLE (ground truth) vs NIII predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import seaborn as sns

# Configuration
SAMPLE_SIZES = [50, 500, 1000, 2500]
TAU_SAMPLES = [1, 2, 3, 4, 5, 6]
RESULTS_DIR = Path('./evaluation_results_tau_samples')

def load_evaluation_results():
    """Load all evaluation results"""
    all_results = {}

    for sample_num in TAU_SAMPLES:
        file_path = RESULTS_DIR / f'tau_sample_{sample_num}_results.npz'
        if file_path.exists():
            data = np.load(file_path, allow_pickle=True)
            all_results[sample_num] = {
                'intensity_image': data['intensity_image'],
                'mask': data['mask'],
                'predictions': data['predictions'].item(),
                'lr_tau': data['lr_tau']
            }
            print(f"Loaded results for tau sample {sample_num}")

    return all_results

def create_ground_truth_vs_predicted_plots(all_results):
    """Create comprehensive ground truth vs predicted lifetime plots"""

    print("Creating ground truth vs predicted lifetime plots...")

    # Collect all data for correlation analysis
    all_correlations = {}

    # Create a large figure with subplots for each model size
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = plt.cm.Set1(np.linspace(0, 1, len(TAU_SAMPLES)))

    for model_idx, sample_count in enumerate(SAMPLE_SIZES):
        ax = axes[model_idx]

        # Collect data for this model across all samples
        ground_truth_all = []
        predicted_all = []
        sample_labels = []

        for sample_num in TAU_SAMPLES:
            if sample_num not in all_results:
                continue

            sample_data = all_results[sample_num]
            lr_tau = sample_data['lr_tau']  # Ground truth (LLE)
            predictions = sample_data['predictions']

            if sample_count not in predictions:
                continue

            pred = predictions[sample_count]  # NIII prediction

            # Handle different resolutions by interpolating or finding valid overlapping regions
            if lr_tau.shape != pred.shape:
                # If shapes don't match, use statistical comparison instead
                lr_valid = lr_tau[lr_tau > 0].flatten()
                pred_valid = pred[pred > 0].flatten()

                # Take random samples if arrays are different sizes
                min_size = min(len(lr_valid), len(pred_valid))
                if min_size > 0:
                    if len(lr_valid) > min_size:
                        idx = np.random.choice(len(lr_valid), min_size, replace=False)
                        lr_valid = lr_valid[idx]
                    if len(pred_valid) > min_size:
                        idx = np.random.choice(len(pred_valid), min_size, replace=False)
                        pred_valid = pred_valid[idx]
            else:
                # Same shape - use pixel-wise comparison
                valid_mask = (lr_tau > 0) & (pred > 0)
                lr_valid = lr_tau[valid_mask]
                pred_valid = pred[valid_mask]

            if len(lr_valid) > 0 and len(pred_valid) > 0:
                ground_truth_all.extend(lr_valid)
                predicted_all.extend(pred_valid)
                sample_labels.extend([sample_num] * len(lr_valid))

        if len(ground_truth_all) > 0:
            ground_truth_all = np.array(ground_truth_all)
            predicted_all = np.array(predicted_all)

            # Create scatter plot
            ax.scatter(ground_truth_all, predicted_all, alpha=0.6, s=1, c='blue')

            # Add perfect correlation line
            min_val = min(np.min(ground_truth_all), np.min(predicted_all))
            max_val = max(np.max(ground_truth_all), np.max(predicted_all))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                   label='Perfect Correlation')

            # Calculate correlation metrics
            if len(ground_truth_all) > 1:
                pearson_r, pearson_p = pearsonr(ground_truth_all, predicted_all)
                spearman_r, spearman_p = spearmanr(ground_truth_all, predicted_all)
                r2 = r2_score(ground_truth_all, predicted_all)
                rmse = np.sqrt(np.mean((ground_truth_all - predicted_all)**2))

                all_correlations[sample_count] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'r2': r2,
                    'rmse': rmse,
                    'n_points': len(ground_truth_all)
                }

                # Add correlation info to plot
                ax.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(ground_truth_all)}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Ground Truth Lifetime (LLE) [ns]')
        ax.set_ylabel('Predicted Lifetime (NIII) [ns]')
        ax.set_title(f'NIII {sample_count} Training Samples')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set equal aspect ratio and limits
        if len(ground_truth_all) > 0:
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

    plt.tight_layout()

    # Save the plot
    output_path = RESULTS_DIR / 'ground_truth_vs_predicted_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'ground_truth_vs_predicted_scatter.pdf', bbox_inches='tight')
    print(f"Ground truth vs predicted scatter plots saved to: {output_path}")

    plt.show()

    return all_correlations

def create_correlation_summary_plot(correlations):
    """Create a summary plot of correlation metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    model_sizes = list(correlations.keys())
    pearson_r = [correlations[size]['pearson_r'] for size in model_sizes]
    spearman_r = [correlations[size]['spearman_r'] for size in model_sizes]
    r2_scores = [correlations[size]['r2'] for size in model_sizes]
    rmse_values = [correlations[size]['rmse'] for size in model_sizes]

    # Pearson correlation
    axes[0,0].plot(model_sizes, pearson_r, 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Training Sample Size')
    axes[0,0].set_ylabel('Pearson Correlation (r)')
    axes[0,0].set_title('Pearson Correlation vs Training Size')
    axes[0,0].set_xscale('log')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0, 1])

    # Spearman correlation
    axes[0,1].plot(model_sizes, spearman_r, 'go-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Training Sample Size')
    axes[0,1].set_ylabel('Spearman Correlation (ρ)')
    axes[0,1].set_title('Spearman Correlation vs Training Size')
    axes[0,1].set_xscale('log')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim([0, 1])

    # R² score
    axes[1,0].plot(model_sizes, r2_scores, 'ro-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Training Sample Size')
    axes[1,0].set_ylabel('R² Score')
    axes[1,0].set_title('R² Score vs Training Size')
    axes[1,0].set_xscale('log')
    axes[1,0].grid(True, alpha=0.3)

    # RMSE
    axes[1,1].plot(model_sizes, rmse_values, 'mo-', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Training Sample Size')
    axes[1,1].set_ylabel('RMSE [ns]')
    axes[1,1].set_title('RMSE vs Training Size')
    axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = RESULTS_DIR / 'correlation_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'correlation_summary.pdf', bbox_inches='tight')
    print(f"Correlation summary plot saved to: {output_path}")

    plt.show()

def print_correlation_statistics(correlations):
    """Print detailed correlation statistics"""

    print("\n" + "="*80)
    print("GROUND TRUTH vs PREDICTED LIFETIME CORRELATION ANALYSIS")
    print("="*80)

    print(f"{'Model Size':<12} {'Pearson r':<12} {'Spearman ρ':<12} {'R²':<12} {'RMSE':<12} {'N Points':<12}")
    print("-" * 72)

    for sample_count in SAMPLE_SIZES:
        if sample_count in correlations:
            corr = correlations[sample_count]
            print(f"{sample_count:<12} {corr['pearson_r']:<12.4f} {corr['spearman_r']:<12.4f} "
                  f"{corr['r2']:<12.4f} {corr['rmse']:<12.4f} {corr['n_points']:<12}")

    print("\nCorrelation Improvements:")
    print("-" * 30)

    if 50 in correlations:
        baseline = correlations[50]

        for size in [500, 1000, 2500]:
            if size in correlations:
                current = correlations[size]
                pearson_improvement = ((current['pearson_r'] - baseline['pearson_r']) / baseline['pearson_r']) * 100
                r2_improvement = ((current['r2'] - baseline['r2']) / abs(baseline['r2']) if baseline['r2'] != 0 else 0) * 100
                rmse_improvement = ((baseline['rmse'] - current['rmse']) / baseline['rmse']) * 100

                print(f"\n{size} vs 50 samples:")
                print(f"  Pearson r improvement: {pearson_improvement:+.2f}%")
                print(f"  R² improvement: {r2_improvement:+.2f}%")
                print(f"  RMSE improvement: {rmse_improvement:+.2f}%")

def main():
    """Main analysis function"""
    print("Creating ground truth vs predicted lifetime analysis...")

    # Load evaluation results
    all_results = load_evaluation_results()

    if not all_results:
        print("No evaluation results found. Please run evaluate_tau_samples.py first.")
        return

    # Create ground truth vs predicted plots
    correlations = create_ground_truth_vs_predicted_plots(all_results)

    # Create correlation summary
    if correlations:
        print("\nCreating correlation summary plots...")
        create_correlation_summary_plot(correlations)

        # Print statistics
        print_correlation_statistics(correlations)

    print("\nGround truth vs predicted analysis complete!")
    print(f"Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()