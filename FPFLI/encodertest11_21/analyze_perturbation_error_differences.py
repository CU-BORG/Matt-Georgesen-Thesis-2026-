#!/usr/bin/env python3
"""
Analyze error differences between baseline and matched perturbations.
Performs pixel-wise paired comparison since parameters are identical.

Key capabilities:
  - Compute pixel-wise error differences: MAE_perturbation - MAE_baseline
  - Create error difference heatmaps
  - Parameter-stratified analysis (by tau2 ranges, photon bins, etc.)
  - Paired statistical tests (t-test)
  - Identify sensitive parameter regions

@author: mg
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = (r'C:\Users\mcg11923\Thesis'
               r'\training_logs_decay_bin62_12.5ns_20260306_112041'
               r'\inference_results_perturbations_matched')

OUTPUT_DIR = os.path.join(RESULTS_DIR, 'error_difference_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERTURBATION_GROUPS = [
    'perturb_delayed_decay',
    'perturb_tri_exponential',
    'perturb_flat_background',
    'perturb_50pct_photons',
    'perturb_10pct_photons',
    'perturb_1pct_photons',
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_inference_results(group_name, n_samples=10):
    """Load predicted and ground truth values from inference results."""
    group_dir = os.path.join(RESULTS_DIR, group_name)

    all_pred_tau1 = []
    all_pred_tau2 = []
    all_pred_f1 = []
    all_gt_tau1 = []
    all_gt_tau2 = []
    all_gt_f1 = []

    for i in range(1, n_samples+1):
        # Find the .mat result file for this sample
        result_file = os.path.join(group_dir, f'Sample_{i:03d}_{group_name}_results.mat')
        if not os.path.exists(result_file):
            print(f"Warning: {result_file} not found, skipping")
            continue

        with h5py.File(result_file, 'r') as f:
            all_pred_tau1.append(np.array(f['tau1_pred']))
            all_pred_tau2.append(np.array(f['tau2_pred']))
            all_pred_f1.append(np.array(f['f1_pred']))
            all_gt_tau1.append(np.array(f['tau1_gt']))
            all_gt_tau2.append(np.array(f['tau2_gt']))
            all_gt_f1.append(np.array(f['f1_gt']))

    if len(all_pred_tau1) == 0:
        return None

    results = {
        'pred_tau1': np.concatenate(all_pred_tau1),
        'pred_tau2': np.concatenate(all_pred_tau2),
        'pred_f1': np.concatenate(all_pred_f1),
        'gt_tau1': np.concatenate(all_gt_tau1),
        'gt_tau2': np.concatenate(all_gt_tau2),
        'gt_f1': np.concatenate(all_gt_f1),
    }

    return results


def compute_pixelwise_errors(results):
    """Compute pixel-wise absolute errors."""
    errors = {
        'tau1': np.abs(results['pred_tau1'] - results['gt_tau1']),
        'tau2': np.abs(results['pred_tau2'] - results['gt_tau2']),
        'f1': np.abs(results['pred_f1'] - results['gt_f1']),
    }
    return errors


def plot_error_difference_map(baseline_errors, perturb_errors, group_name, param='tau2'):
    """Create heatmap showing error increase from perturbation."""
    error_diff = perturb_errors[param] - baseline_errors[param]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Reshape to image dimensions (assuming square images)
    n_pixels = len(error_diff)
    img_size = int(np.sqrt(n_pixels))
    error_diff_img = error_diff[:img_size**2].reshape(img_size, img_size)
    baseline_img = baseline_errors[param][:img_size**2].reshape(img_size, img_size)
    perturb_img = perturb_errors[param][:img_size**2].reshape(img_size, img_size)

    # Baseline error map
    im0 = axes[0].imshow(baseline_img, cmap='viridis', vmin=0)
    axes[0].set_title(f'Baseline {param} Error', fontsize=18, fontweight='bold')
    axes[0].axis('off')
    cbar0 = plt.colorbar(im0, ax=axes[0], label=f'{param} MAE (ns)')
    cbar0.ax.tick_params(labelsize=14)
    cbar0.set_label(f'{param} MAE (ns)', fontsize=16)

    # Perturbed error map
    im1 = axes[1].imshow(perturb_img, cmap='viridis', vmin=0)
    axes[1].set_title(f'{group_name} {param} Error', fontsize=18, fontweight='bold')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], label=f'{param} MAE (ns)')
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label(f'{param} MAE (ns)', fontsize=16)

    # Error difference map (perturbation effect)
    max_diff = np.percentile(np.abs(error_diff_img), 99)
    im2 = axes[2].imshow(error_diff_img, cmap='RdYlGn_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f'{param} Error Increase\n(Perturbation Effect)', fontsize=18, fontweight='bold')
    axes[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[2], label=f'Delta {param} MAE (ns)')
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label(f'Delta {param} MAE (ns)', fontsize=16)

    plt.suptitle(f'Pixel-Wise Error Analysis: {group_name}', fontsize=20, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f'error_difference_map_{group_name}_{param}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def parameter_stratified_analysis(baseline_errors, perturb_errors, baseline_results, group_name):
    """Analyze error increase stratified by parameter values."""

    # Stratify by tau2 ranges
    tau2_bins = [(1.5, 2.5), (2.5, 3.5), (3.5, 4.5)]
    tau2_labels = ['Short (1.5-2.5 ns)', 'Medium (2.5-3.5 ns)', 'Long (3.5-4.5 ns)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for param_idx, param in enumerate(['tau1', 'tau2', 'f1']):
        error_increases = []
        labels = []

        for (low, high), label in zip(tau2_bins, tau2_labels):
            mask = (baseline_results['gt_tau2'] >= low) & (baseline_results['gt_tau2'] < high)
            if np.sum(mask) > 0:
                baseline_err = baseline_errors[param][mask]
                perturb_err = perturb_errors[param][mask]
                error_increase = perturb_err - baseline_err
                error_increases.append(error_increase)
                labels.append(label)

        # Violin plot
        parts = axes[param_idx].violinplot(error_increases, positions=range(len(labels)),
                                            showmeans=True, showmedians=True)
        axes[param_idx].set_xticks(range(len(labels)))
        axes[param_idx].set_xticklabels(labels, rotation=15, ha='right', fontsize=14)
        axes[param_idx].set_ylabel(f'Delta {param} MAE', fontsize=16)
        axes[param_idx].set_title(f'{param} Error Increase\nby tau2 Range', fontsize=18, fontweight='bold')
        axes[param_idx].tick_params(axis='y', labelsize=14)
        axes[param_idx].grid(True, alpha=0.3, axis='y')
        axes[param_idx].axhline(0, color='red', linestyle='--', linewidth=1, label='No change')
        axes[param_idx].legend(fontsize=14)

    plt.suptitle(f'Parameter-Stratified Error Analysis: {group_name}', fontsize=20, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f'parameter_stratified_{group_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def paired_statistical_test(baseline_errors, perturb_errors):
    """Perform paired t-tests for each parameter."""
    results = {}

    for param in ['tau1', 'tau2', 'f1']:
        t_stat, p_value = ttest_rel(perturb_errors[param], baseline_errors[param])
        mean_increase = np.mean(perturb_errors[param] - baseline_errors[param])
        median_increase = np.median(perturb_errors[param] - baseline_errors[param])

        results[param] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_increase': mean_increase,
            'median_increase': median_increase,
            'significant': p_value < 0.001
        }

    return results


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

def main():
    print("="*70)
    print("PERTURBATION ERROR DIFFERENCE ANALYSIS")
    print("="*70)
    print("Analyzing pixel-wise paired comparison between baseline and perturbations\n")

    # Load baseline results
    print("Loading baseline_normal results...")
    baseline_results = load_inference_results('baseline_normal')
    if baseline_results is None:
        print("ERROR: Could not load baseline results")
        return

    baseline_errors = compute_pixelwise_errors(baseline_results)
    print(f"  Loaded {len(baseline_errors['tau1'])} pixels")

    # Calculate baseline MAEs
    baseline_maes = {
        'tau1': np.mean(baseline_errors['tau1']),
        'tau2': np.mean(baseline_errors['tau2']),
        'f1': np.mean(baseline_errors['f1']),
    }
    print(f"  Baseline MAEs: tau1={baseline_maes['tau1']:.4f}, "
          f"tau2={baseline_maes['tau2']:.4f}, "
          f"f1={baseline_maes['f1']:.4f}\n")

    # Analyze each perturbation
    summary = []

    for group_name in PERTURBATION_GROUPS:
        print(f"\n{'='*70}")
        print(f"Analyzing: {group_name}")
        print(f"{'='*70}")

        # Load perturbation results
        perturb_results = load_inference_results(group_name)
        if perturb_results is None:
            print(f"  Skipping {group_name} (no results found)")
            continue

        perturb_errors = compute_pixelwise_errors(perturb_results)

        # Paired statistical test
        stats = paired_statistical_test(baseline_errors, perturb_errors)

        print("\nPaired t-test results:")
        for param in ['tau1', 'tau2', 'f1']:
            s = stats[param]
            sig_marker = "***" if s['significant'] else ""
            print(f"  {param}: mean Delta={s['mean_increase']:+.4f}, "
                  f"t={s['t_statistic']:.2f}, p={s['p_value']:.2e} {sig_marker}")

        # Create error difference maps
        for param in ['tau1', 'tau2', 'f1']:
            path = plot_error_difference_map(baseline_errors, perturb_errors, group_name, param)
            print(f"  Saved: {os.path.basename(path)}")

        # Parameter-stratified analysis
        path = parameter_stratified_analysis(baseline_errors, perturb_errors,
                                              baseline_results, group_name)
        print(f"  Saved: {os.path.basename(path)}")

        # Store summary with both absolute MAEs and increases
        summary.append({
            'group': group_name,
            'tau1_mae': np.mean(perturb_errors['tau1']),
            'tau2_mae': np.mean(perturb_errors['tau2']),
            'f1_mae': np.mean(perturb_errors['f1']),
            'tau1_increase': stats['tau1']['mean_increase'],
            'tau2_increase': stats['tau2']['mean_increase'],
            'f1_increase': stats['f1']['mean_increase'],
        })

    # Create summary bar charts
    if summary:
        create_summary_chart(summary, baseline_maes)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"All results saved to: {OUTPUT_DIR}")


def create_combined_mae_chart(summary, baseline_maes):
    """Create bar chart showing baseline once, then all perturbation MAEs."""
    groups = ['Baseline'] + [s['group'] for s in summary]
    tau1_vals = [baseline_maes['tau1']] + [s['tau1_mae'] for s in summary]
    tau2_vals = [baseline_maes['tau2']] + [s['tau2_mae'] for s in summary]
    f1_vals = [baseline_maes['f1']] + [s['f1_mae'] for s in summary]

    # Short labels
    short_labels = {
        'Baseline': 'Baseline',
        'perturb_delayed_decay': 'Delayed\nDecay',
        'perturb_tri_exponential': 'Tri-Exp',
        'perturb_flat_background': 'Flat\nBkgd',
        'perturb_50pct_photons': '50%\nPhotons',
        'perturb_10pct_photons': '10%\nPhotons',
        'perturb_1pct_photons': '1%\nPhotons',
    }
    labels = [short_labels.get(g, g) for g in groups]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    x = np.arange(len(groups))

    # Colors: baseline in distinct color, perturbations in gradient
    colors_tau1 = ['lightblue'] + ['steelblue'] * len(summary)
    colors_tau2 = ['lightsalmon'] + ['orangered'] * len(summary)
    colors_f1 = ['lightgreen'] + ['forestgreen'] * len(summary)

    # tau1
    bars = axes[0].bar(x, tau1_vals, color=colors_tau1, edgecolor='black', linewidth=0.8)
    # Add value labels
    for bar, val in zip(bars, tau1_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, val,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=14)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=14)
    axes[0].set_ylabel('Mean Absolute Error (ns)', fontsize=16)
    axes[0].set_title(r'$\tau_1$ MAE', fontsize=18, fontweight='bold')
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')
    # Add padding at top for labels
    ylim = axes[0].get_ylim()
    axes[0].set_ylim(ylim[0], ylim[1] * 1.1)

    # tau2
    bars = axes[1].bar(x, tau2_vals, color=colors_tau2, edgecolor='black', linewidth=0.8)
    # Add value labels
    for bar, val in zip(bars, tau2_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=14)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=14)
    axes[1].set_ylabel('Mean Absolute Error (ns)', fontsize=16)
    axes[1].set_title(r'$\tau_2$ MAE', fontsize=18, fontweight='bold')
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    # Add padding at top for labels
    ylim = axes[1].get_ylim()
    axes[1].set_ylim(ylim[0], ylim[1] * 1.1)

    # f1
    bars = axes[2].bar(x, f1_vals, color=colors_f1, edgecolor='black', linewidth=0.8)
    # Add value labels
    for bar, val in zip(bars, f1_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, val,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=14)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=14)
    axes[2].set_ylabel('Mean Absolute Error', fontsize=16)
    axes[2].set_title(r'$f$ MAE', fontsize=18, fontweight='bold')
    axes[2].tick_params(axis='y', labelsize=14)
    axes[2].grid(True, alpha=0.3, axis='y')
    # Add padding at top for labels
    ylim = axes[2].get_ylim()
    axes[2].set_ylim(ylim[0], ylim[1] * 1.1)

    plt.suptitle('Baseline and Perturbation MAE Comparison',
                 fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'combined_mae_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved combined MAE comparison: {os.path.basename(save_path)}")


def create_summary_chart(summary, baseline_maes):
    """Create bar charts summarizing error increases and combined MAEs across all perturbations."""
    # First create the combined MAE comparison chart
    create_combined_mae_chart(summary, baseline_maes)

    # Then create the error increase chart (existing functionality)
    groups = [s['group'] for s in summary]
    tau1_inc = [s['tau1_increase'] for s in summary]
    tau2_inc = [s['tau2_increase'] for s in summary]
    f1_inc = [s['f1_increase'] for s in summary]

    # Short labels
    short_labels = {
        'perturb_delayed_decay': 'Delayed\nDecay',
        'perturb_tri_exponential': 'Tri-Exp',
        'perturb_flat_background': 'Flat\nBkgd',
        'perturb_50pct_photons': '50%\nPhotons',
        'perturb_10pct_photons': '10%\nPhotons',
        'perturb_1pct_photons': '1%\nPhotons',
    }
    labels = [short_labels.get(g, g) for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(groups))

    # tau1
    bars = axes[0].bar(x, tau1_inc, color='tab:blue', edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, tau1_inc):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.001,
                     f'{val:+.3f}', ha='center', va='bottom', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=14)
    axes[0].set_ylabel('Mean Error Increase (ns)', fontsize=16)
    axes[0].set_title(r'$\tau_1$ Error Increase', fontsize=18, fontweight='bold')
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].grid(True, alpha=0.3, axis='y')
    # Add padding at top for labels
    ylim = axes[0].get_ylim()
    y_range = ylim[1] - ylim[0]
    axes[0].set_ylim(ylim[0], ylim[1] + y_range * 0.1)

    # tau2
    bars = axes[1].bar(x, tau2_inc, color='tab:orange', edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, tau2_inc):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                     f'{val:+.3f}', ha='center', va='bottom', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=14)
    axes[1].set_ylabel('Mean Error Increase (ns)', fontsize=16)
    axes[1].set_title(r'$\tau_2$ Error Increase', fontsize=18, fontweight='bold')
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='y')
    # Add padding at top for labels
    ylim = axes[1].get_ylim()
    y_range = ylim[1] - ylim[0]
    axes[1].set_ylim(ylim[0], ylim[1] + y_range * 0.1)

    # f1
    bars = axes[2].bar(x, f1_inc, color='tab:green', edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, f1_inc):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                     f'{val:+.3f}', ha='center', va='bottom', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=14)
    axes[2].set_ylabel('Mean Error Increase', fontsize=16)
    axes[2].set_title(r'$f$ Error Increase', fontsize=18, fontweight='bold')
    axes[2].tick_params(axis='y', labelsize=14)
    axes[2].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[2].grid(True, alpha=0.3, axis='y')
    # Add padding at top for labels
    ylim = axes[2].get_ylim()
    y_range = ylim[1] - ylim[0]
    axes[2].set_ylim(ylim[0], ylim[1] + y_range * 0.1)

    plt.suptitle('Mean Error Increase from Baseline (Paired Comparison)',
                 fontsize=20, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'error_increase_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved summary: {os.path.basename(save_path)}")


if __name__ == '__main__':
    main()
