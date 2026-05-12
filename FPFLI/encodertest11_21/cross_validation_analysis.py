#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Validation Analysis: Test Each Model on Data from ALL Levels

Creates a comprehensive 10×10 performance matrix where each of the 10 models
is tested on data from all 10 spectrum levels.

@author: Cross-validation analysis script
"""

# Fix OpenMP library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import h5py
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder

# Configuration
STD_LEVELS = 10
TAU_1_STD_ARRAY = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05]
TAU_2_STD_ARRAY = [0.10, 0.30, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50]

BASE_DATA_DIR = r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite'
BASE_MODEL_DIR = r'C:\Users\mcg11923\Thesis'

NUM_SAMPLES_PER_LEVEL = 5  # Test 5 random samples from each data level
SIGNAL_THRESHOLD = 50


def find_latest_model_dir(level_idx):
    """Find the most recent training directory for a given level"""
    pattern = os.path.join(BASE_MODEL_DIR, f'training_logs_spectrum_level_{level_idx:02d}_*')
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return sorted(dirs)[-1]


def load_model(level_idx):
    """Load a trained model for a given level"""
    model_dir = find_latest_model_dir(level_idx)
    if model_dir is None:
        return None

    model_path = os.path.join(model_dir, f'model_final_spectrum_level_{level_idx:02d}.pth')
    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = LLE_BiExp_Autoencoder(
        dim=checkpoint['hyperparameters']['dim'],
        depth=checkpoint['hyperparameters']['depth'],
        kernel_size=checkpoint['hyperparameters']['kernel_size'],
        patch_size=checkpoint['hyperparameters']['patch_size']
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model


def get_samples_for_level(data_level, num_samples=5, seed=42):
    """Get random sample files from a specific data level"""
    tau1_std = TAU_1_STD_ARRAY[data_level - 1]
    tau2_std = TAU_2_STD_ARRAY[data_level - 1]

    data_dir = os.path.join(BASE_DATA_DIR, f'spectrum_level_{data_level:02d}_std_{tau1_std:.2f}_{tau2_std:.2f}')
    pattern = os.path.join(data_dir, 'Sample_*_spectrumspectrum.mat')
    all_files = glob.glob(pattern)

    if len(all_files) == 0:
        return []

    # Select random files with fixed seed for reproducibility
    np.random.seed(seed)
    num_samples = min(num_samples, len(all_files))
    selected_files = np.random.choice(all_files, num_samples, replace=False)

    return list(selected_files)


def compute_mae_for_sample(model, sample_path):
    """Compute MAE for tau1, tau2, f1 on a single sample"""
    # Load data using h5py for MATLAB v7.3 files
    with h5py.File(sample_path, 'r') as f:
        Hist = np.array(f['Hist']).T  # [256, 256, 256]
        tau_gt = np.array(f['tau_gt_components']).T  # [256, 256, 2]
        f_gt_data = np.array(f['f_gt_components']).T  # [256, 256, 2]

    H, W, T = Hist.shape

    # Apply signal threshold
    total_photons = np.sum(Hist, axis=2)
    valid_mask = total_photons >= SIGNAL_THRESHOLD

    if not np.any(valid_mask):
        return None

    # Normalize decay curves
    Hist_norm = np.zeros_like(Hist)
    for i in range(H):
        for j in range(W):
            if valid_mask[i, j]:
                total = total_photons[i, j]
                if total > 0:
                    Hist_norm[i, j, :] = Hist[i, j, :] / total

    # Prepare data for model
    decay_batch = []
    gt_tau1_list = []
    gt_tau2_list = []
    gt_f1_list = []

    for i in range(H):
        for j in range(W):
            if valid_mask[i, j]:
                decay_batch.append(Hist_norm[i, j, :])
                gt_tau1_list.append(tau_gt[i, j, 0])
                gt_tau2_list.append(tau_gt[i, j, 1])
                gt_f1_list.append(f_gt_data[i, j, 0])

    decay_batch = np.array(decay_batch)
    gt_tau1 = np.array(gt_tau1_list)
    gt_tau2 = np.array(gt_tau2_list)
    gt_f1 = np.array(gt_f1_list)

    # Batch prediction
    batch_size = 512
    pred_tau1_all = []
    pred_tau2_all = []
    pred_f1_all = []

    with torch.no_grad():
        for start_idx in range(0, len(decay_batch), batch_size):
            end_idx = min(start_idx + batch_size, len(decay_batch))
            batch_decay = decay_batch[start_idx:end_idx]

            batch_decay_tensor = torch.FloatTensor(batch_decay).unsqueeze(1)
            irf_tensor = torch.zeros_like(batch_decay_tensor)

            latent = model.encode(batch_decay_tensor, irf_tensor)
            tau1 = latent[:, 0].cpu().numpy()
            tau2 = latent[:, 1].cpu().numpy()
            f1 = latent[:, 2].cpu().numpy()

            pred_tau1_all.extend(tau1)
            pred_tau2_all.extend(tau2)
            pred_f1_all.extend(f1)

    pred_tau1 = np.array(pred_tau1_all)
    pred_tau2 = np.array(pred_tau2_all)
    pred_f1 = np.array(pred_f1_all)

    # Compute MAE
    mae_tau1 = np.mean(np.abs(pred_tau1 - gt_tau1))
    mae_tau2 = np.mean(np.abs(pred_tau2 - gt_tau2))
    mae_f1 = np.mean(np.abs(pred_f1 - gt_f1))

    return {
        'mae_tau1': mae_tau1,
        'mae_tau2': mae_tau2,
        'mae_f1': mae_f1,
        'num_pixels': len(gt_tau1)
    }


def test_model_on_data_level(model, data_level, num_samples=5):
    """Test one model on all samples from one data level"""
    samples = get_samples_for_level(data_level, num_samples)

    if len(samples) == 0:
        return None

    results = []
    for sample_path in samples:
        mae_result = compute_mae_for_sample(model, sample_path)
        if mae_result is not None:
            results.append(mae_result)
        else:
            results.append({'mae_tau1': np.nan, 'mae_tau2': np.nan, 'mae_f1': np.nan})

    return results


def create_heatmap_plots(results_matrix, output_dir):
    """Create 10×10 heatmap plots for tau1, tau2, f1"""
    print("\nCreating heatmap visualizations...")

    # Extract average MAEs for each model-data combination
    tau1_matrix = np.zeros((STD_LEVELS, STD_LEVELS))
    tau2_matrix = np.zeros((STD_LEVELS, STD_LEVELS))
    f1_matrix = np.zeros((STD_LEVELS, STD_LEVELS))

    for model_level in range(1, STD_LEVELS + 1):
        for data_level in range(1, STD_LEVELS + 1):
            if (model_level, data_level) in results_matrix:
                results = results_matrix[(model_level, data_level)]
                tau1_matrix[model_level-1, data_level-1] = np.nanmean([r['mae_tau1'] for r in results])
                tau2_matrix[model_level-1, data_level-1] = np.nanmean([r['mae_tau2'] for r in results])
                f1_matrix[model_level-1, data_level-1] = np.nanmean([r['mae_f1'] for r in results])
            else:
                tau1_matrix[model_level-1, data_level-1] = np.nan
                tau2_matrix[model_level-1, data_level-1] = np.nan
                f1_matrix[model_level-1, data_level-1] = np.nan

    # Create heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    matrices = [tau1_matrix, tau2_matrix, f1_matrix]
    titles = ['Tau1 MAE (ns)', 'Tau2 MAE (ns)', 'F1 MAE']
    cmaps = ['YlOrRd', 'YlOrRd', 'YlGnBu']

    for idx, (matrix, title, cmap) in enumerate(zip(matrices, titles, cmaps)):
        ax = axes[idx]

        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap=cmap, ax=ax,
                   cbar_kws={'label': 'MAE'}, vmin=0)

        ax.set_xlabel('Test Data Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('Trained Model Level', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')

        # Set tick labels
        ax.set_xticklabels(range(1, STD_LEVELS + 1))
        ax.set_yticklabels(range(1, STD_LEVELS + 1))

        # Highlight diagonal
        for i in range(STD_LEVELS):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    plt.suptitle('Cross-Validation Performance Matrix (10 Models × 10 Data Levels)',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'heatmap_cross_validation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved heatmap to: {save_path}")
    plt.close()


def create_robustness_plots(results_matrix, output_dir):
    """Plot how each model performs across all data levels"""
    print("\nCreating model robustness plots...")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    parameters = ['tau1', 'tau2', 'f1']
    titles = ['Tau1 MAE Across Data Levels', 'Tau2 MAE Across Data Levels', 'F1 MAE Across Data Levels']
    ylabels = ['Tau1 MAE (ns)', 'Tau2 MAE (ns)', 'F1 MAE']

    for param_idx, (param, title, ylabel) in enumerate(zip(parameters, titles, ylabels)):
        ax = axes[param_idx]

        for model_level in range(1, STD_LEVELS + 1):
            maes = []
            data_levels = []

            for data_level in range(1, STD_LEVELS + 1):
                if (model_level, data_level) in results_matrix:
                    results = results_matrix[(model_level, data_level)]
                    mae = np.nanmean([r[f'mae_{param}'] for r in results])
                    maes.append(mae)
                    data_levels.append(data_level)

            if len(maes) > 0:
                ax.plot(data_levels, maes, 'o-', label=f'Model {model_level}',
                       linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Test Data Level', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, STD_LEVELS + 1))

    plt.suptitle('Model Robustness: Performance Across All Data Levels',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'model_robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved robustness plot to: {save_path}")
    plt.close()


def create_difficulty_plots(results_matrix, output_dir):
    """Plot how all models perform on each data level"""
    print("\nCreating data difficulty plots...")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    parameters = ['tau1', 'tau2', 'f1']
    titles = ['Tau1: Data Level Difficulty', 'Tau2: Data Level Difficulty', 'F1: Data Level Difficulty']
    ylabels = ['Tau1 MAE (ns)', 'Tau2 MAE (ns)', 'F1 MAE']

    for param_idx, (param, title, ylabel) in enumerate(zip(parameters, titles, ylabels)):
        ax = axes[param_idx]

        for data_level in range(1, STD_LEVELS + 1):
            maes = []
            model_levels = []

            for model_level in range(1, STD_LEVELS + 1):
                if (model_level, data_level) in results_matrix:
                    results = results_matrix[(model_level, data_level)]
                    mae = np.nanmean([r[f'mae_{param}'] for r in results])
                    maes.append(mae)
                    model_levels.append(model_level)

            if len(maes) > 0:
                ax.plot(model_levels, maes, 'o-', label=f'Data Level {data_level}',
                       linewidth=2, markersize=6, alpha=0.7)

        ax.set_xlabel('Trained Model Level', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, STD_LEVELS + 1))

    plt.suptitle('Data Difficulty: How All Models Perform on Each Data Level',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'data_difficulty.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved difficulty plot to: {save_path}")
    plt.close()


def create_diagonal_analysis(results_matrix, output_dir):
    """Compare in-distribution vs out-of-distribution performance"""
    print("\nCreating diagonal vs off-diagonal analysis...")

    diagonal_tau1 = []
    diagonal_tau2 = []
    diagonal_f1 = []

    offdiag_tau1 = []
    offdiag_tau2 = []
    offdiag_f1 = []

    for model_level in range(1, STD_LEVELS + 1):
        for data_level in range(1, STD_LEVELS + 1):
            if (model_level, data_level) in results_matrix:
                results = results_matrix[(model_level, data_level)]
                mae_tau1 = np.nanmean([r['mae_tau1'] for r in results])
                mae_tau2 = np.nanmean([r['mae_tau2'] for r in results])
                mae_f1 = np.nanmean([r['mae_f1'] for r in results])

                if model_level == data_level:
                    diagonal_tau1.append(mae_tau1)
                    diagonal_tau2.append(mae_tau2)
                    diagonal_f1.append(mae_f1)
                else:
                    offdiag_tau1.append(mae_tau1)
                    offdiag_tau2.append(mae_tau2)
                    offdiag_f1.append(mae_f1)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    data_pairs = [
        (diagonal_tau1, offdiag_tau1, 'Tau1 MAE (ns)'),
        (diagonal_tau2, offdiag_tau2, 'Tau2 MAE (ns)'),
        (diagonal_f1, offdiag_f1, 'F1 MAE')
    ]

    for idx, (diag, offdiag, label) in enumerate(data_pairs):
        ax = axes[idx]

        positions = [1, 2]
        bp = ax.boxplot([diag, offdiag], positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True)

        # Color boxes
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel(label, fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(['In-Distribution\n(Diagonal)', 'Out-of-Distribution\n(Off-Diagonal)'])
        ax.set_title(f'{label.split()[0]} Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_diag = np.mean(diag)
        mean_offdiag = np.mean(offdiag)
        gap = ((mean_offdiag - mean_diag) / mean_diag) * 100

        ax.text(0.5, 0.95, f'Gap: {gap:.1f}%',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               ha='center', va='top')

    plt.suptitle('In-Distribution vs Out-of-Distribution Performance',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'diagonal_vs_offdiagonal.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved diagonal analysis to: {save_path}")
    plt.close()

    return {
        'diagonal': {'tau1': diagonal_tau1, 'tau2': diagonal_tau2, 'f1': diagonal_f1},
        'offdiag': {'tau1': offdiag_tau1, 'tau2': offdiag_tau2, 'f1': offdiag_f1}
    }


def create_optimal_training_plot(results_matrix, output_dir):
    """For each test level, identify which training level gives best performance"""
    print("\nCreating optimal training level analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    parameters = ['tau1', 'tau2', 'f1']
    titles = ['Optimal Model for Tau1 Prediction', 'Optimal Model for Tau2 Prediction', 'Optimal Model for F1 Prediction']

    for param_idx, (param, title) in enumerate(zip(parameters, titles)):
        ax = axes[param_idx]

        best_model_for_data = []
        data_levels_list = []

        for data_level in range(1, STD_LEVELS + 1):
            best_mae = float('inf')
            best_model = None

            for model_level in range(1, STD_LEVELS + 1):
                if (model_level, data_level) in results_matrix:
                    results = results_matrix[(model_level, data_level)]
                    mae = np.nanmean([r[f'mae_{param}'] for r in results])

                    if mae < best_mae:
                        best_mae = mae
                        best_model = model_level

            if best_model is not None:
                best_model_for_data.append(best_model)
                data_levels_list.append(data_level)

        ax.plot(data_levels_list, best_model_for_data, 'o-', linewidth=3, markersize=10, color='green')
        ax.plot([1, STD_LEVELS], [1, STD_LEVELS], 'k--', alpha=0.3, label='Diagonal (same level)')

        ax.set_xlabel('Test Data Level', fontsize=12)
        ax.set_ylabel('Best Trained Model Level', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, STD_LEVELS + 1))
        ax.set_yticks(range(1, STD_LEVELS + 1))
        ax.set_xlim([0.5, STD_LEVELS + 0.5])
        ax.set_ylim([0.5, STD_LEVELS + 0.5])

    plt.suptitle('Optimal Training Strategy: Which Model Works Best for Each Data Level',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'optimal_training_level.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved optimal training plot to: {save_path}")
    plt.close()


def create_distance_analysis(results_matrix, output_dir):
    """Plot MAE vs distribution distance |model_level - data_level|"""
    print("\nCreating distribution distance analysis...")

    # Collect data grouped by distance
    distance_data = {i: {'tau1': [], 'tau2': [], 'f1': []} for i in range(STD_LEVELS)}

    for model_level in range(1, STD_LEVELS + 1):
        for data_level in range(1, STD_LEVELS + 1):
            if (model_level, data_level) in results_matrix:
                distance = abs(model_level - data_level)
                results = results_matrix[(model_level, data_level)]

                distance_data[distance]['tau1'].append(np.nanmean([r['mae_tau1'] for r in results]))
                distance_data[distance]['tau2'].append(np.nanmean([r['mae_tau2'] for r in results]))
                distance_data[distance]['f1'].append(np.nanmean([r['mae_f1'] for r in results]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    parameters = ['tau1', 'tau2', 'f1']
    titles = ['Tau1 MAE vs Distance', 'Tau2 MAE vs Distance', 'F1 MAE vs Distance']
    ylabels = ['Tau1 MAE (ns)', 'Tau2 MAE (ns)', 'F1 MAE']

    for param_idx, (param, title, ylabel) in enumerate(zip(parameters, titles, ylabels)):
        ax = axes[param_idx]

        distances = []
        means = []
        stds = []

        for dist in sorted(distance_data.keys()):
            if len(distance_data[dist][param]) > 0:
                distances.append(dist)
                means.append(np.mean(distance_data[dist][param]))
                stds.append(np.std(distance_data[dist][param]))

        ax.errorbar(distances, means, yerr=stds, fmt='o-', linewidth=3, markersize=10,
                   capsize=5, capthick=2, color='purple')

        ax.set_xlabel('Distribution Distance |Model Level - Data Level|', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, STD_LEVELS))

    plt.suptitle('Performance vs Distribution Distance',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'distance_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved distance analysis to: {save_path}")
    plt.close()


def generate_report(results_matrix, diagonal_stats, output_dir):
    """Generate comprehensive text report"""
    print("\nGenerating comprehensive report...")

    report_path = os.path.join(output_dir, 'cross_validation_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-VALIDATION ANALYSIS REPORT\n")
        f.write("10 Models × 10 Data Levels × 5 Samples = 500 Total Predictions\n")
        f.write("="*80 + "\n\n")

        # Overall statistics
        f.write("DIAGONAL (IN-DISTRIBUTION) STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Tau1 MAE: mean={np.mean(diagonal_stats['diagonal']['tau1']):.6f}, ")
        f.write(f"std={np.std(diagonal_stats['diagonal']['tau1']):.6f}\n")
        f.write(f"Tau2 MAE: mean={np.mean(diagonal_stats['diagonal']['tau2']):.6f}, ")
        f.write(f"std={np.std(diagonal_stats['diagonal']['tau2']):.6f}\n")
        f.write(f"F1 MAE: mean={np.mean(diagonal_stats['diagonal']['f1']):.6f}, ")
        f.write(f"std={np.std(diagonal_stats['diagonal']['f1']):.6f}\n\n")

        f.write("OFF-DIAGONAL (OUT-OF-DISTRIBUTION) STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Tau1 MAE: mean={np.mean(diagonal_stats['offdiag']['tau1']):.6f}, ")
        f.write(f"std={np.std(diagonal_stats['offdiag']['tau1']):.6f}\n")
        f.write(f"Tau2 MAE: mean={np.mean(diagonal_stats['offdiag']['tau2']):.6f}, ")
        f.write(f"std={np.std(diagonal_stats['offdiag']['tau2']):.6f}\n")
        f.write(f"F1 MAE: mean={np.mean(diagonal_stats['offdiag']['f1']):.6f}, ")
        f.write(f"std={np.std(diagonal_stats['offdiag']['f1']):.6f}\n\n")

        # Generalization gap
        tau1_gap = ((np.mean(diagonal_stats['offdiag']['tau1']) - np.mean(diagonal_stats['diagonal']['tau1'])) /
                   np.mean(diagonal_stats['diagonal']['tau1'])) * 100
        tau2_gap = ((np.mean(diagonal_stats['offdiag']['tau2']) - np.mean(diagonal_stats['diagonal']['tau2'])) /
                   np.mean(diagonal_stats['diagonal']['tau2'])) * 100
        f1_gap = ((np.mean(diagonal_stats['offdiag']['f1']) - np.mean(diagonal_stats['diagonal']['f1'])) /
                 np.mean(diagonal_stats['diagonal']['f1'])) * 100

        f.write("GENERALIZATION GAP (OOD vs In-Dist):\n")
        f.write("-"*80 + "\n")
        f.write(f"Tau1: {tau1_gap:.1f}% increase\n")
        f.write(f"Tau2: {tau2_gap:.1f}% increase\n")
        f.write(f"F1: {f1_gap:.1f}% increase\n\n")

        # Find best overall model
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*80 + "\n\n")

        # Best all-around model
        avg_maes = {}
        for model_level in range(1, STD_LEVELS + 1):
            maes = []
            for data_level in range(1, STD_LEVELS + 1):
                if (model_level, data_level) in results_matrix:
                    results = results_matrix[(model_level, data_level)]
                    # Average across all parameters
                    mae_avg = np.nanmean([r['mae_tau1'] + r['mae_tau2'] + r['mae_f1'] for r in results])
                    maes.append(mae_avg)
            avg_maes[model_level] = np.mean(maes) if maes else float('inf')

        best_model = min(avg_maes, key=avg_maes.get)
        f.write(f"1. BEST ALL-AROUND MODEL: Level {best_model}\n")
        f.write(f"   Average combined MAE across all test levels: {avg_maes[best_model]:.6f}\n\n")

        # Most robust model (lowest variance)
        variances = {}
        for model_level in range(1, STD_LEVELS + 1):
            maes = []
            for data_level in range(1, STD_LEVELS + 1):
                if (model_level, data_level) in results_matrix:
                    results = results_matrix[(model_level, data_level)]
                    mae_avg = np.nanmean([r['mae_tau1'] + r['mae_tau2'] + r['mae_f1'] for r in results])
                    maes.append(mae_avg)
            variances[model_level] = np.std(maes) if len(maes) > 1 else float('inf')

        most_robust = min(variances, key=variances.get)
        f.write(f"2. MOST ROBUST MODEL (lowest variance): Level {most_robust}\n")
        f.write(f"   Std dev of combined MAE: {variances[most_robust]:.6f}\n\n")

        f.write("3. SPECIALIST MODELS (best for specific data levels):\n")
        for data_level in range(1, STD_LEVELS + 1):
            best_for_this = None
            best_mae = float('inf')

            for model_level in range(1, STD_LEVELS + 1):
                if (model_level, data_level) in results_matrix:
                    results = results_matrix[(model_level, data_level)]
                    mae_avg = np.nanmean([r['mae_tau1'] + r['mae_tau2'] + r['mae_f1'] for r in results])
                    if mae_avg < best_mae:
                        best_mae = mae_avg
                        best_for_this = model_level

            if best_for_this is not None:
                f.write(f"   Data Level {data_level}: Use Model {best_for_this} (MAE={best_mae:.6f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"[OK] Saved report to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print(" CROSS-VALIDATION ANALYSIS")
    print(" Testing All 10 Models on Data from All 10 Levels")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Models: {STD_LEVELS}")
    print(f"  Data levels: {STD_LEVELS}")
    print(f"  Samples per level: {NUM_SAMPLES_PER_LEVEL}")
    print(f"  Total predictions: {STD_LEVELS * STD_LEVELS * NUM_SAMPLES_PER_LEVEL}")
    print("="*80 + "\n")

    # Dictionary to store results: (model_level, data_level) -> list of MAE results
    results_matrix = {}

    # Main cross-validation loop
    for model_level in range(1, STD_LEVELS + 1):
        print(f"\n{'='*80}")
        print(f"LOADING MODEL {model_level}/{STD_LEVELS}")
        print(f"{'='*80}")

        # Load model
        model = load_model(model_level)
        if model is None:
            print(f"WARNING: Model {model_level} not found, skipping...")
            continue

        print(f"[OK] Model {model_level} loaded")

        # Test on all data levels
        for data_level in range(1, STD_LEVELS + 1):
            print(f"  Testing on Data Level {data_level}/{STD_LEVELS}...", end='', flush=True)

            results = test_model_on_data_level(model, data_level, NUM_SAMPLES_PER_LEVEL)

            if results is None or len(results) == 0:
                print(" [SKIP - No data]")
                continue

            results_matrix[(model_level, data_level)] = results

            # Compute average MAE for this combination
            avg_tau1 = np.nanmean([r['mae_tau1'] for r in results])
            avg_tau2 = np.nanmean([r['mae_tau2'] for r in results])
            avg_f1 = np.nanmean([r['mae_f1'] for r in results])

            print(f" [OK] tau1={avg_tau1:.4f}, tau2={avg_tau2:.4f}, f1={avg_f1:.4f}")

    # Create output directory
    output_dir = os.path.join(BASE_MODEL_DIR, 'cross_validation_analysis')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print("="*80)

    # Generate all visualizations
    create_heatmap_plots(results_matrix, output_dir)
    create_robustness_plots(results_matrix, output_dir)
    create_difficulty_plots(results_matrix, output_dir)
    diagonal_stats = create_diagonal_analysis(results_matrix, output_dir)
    create_optimal_training_plot(results_matrix, output_dir)
    create_distance_analysis(results_matrix, output_dir)

    # Generate report
    generate_report(results_matrix, diagonal_stats, output_dir)

    print("\n" + "="*80)
    print(" CROSS-VALIDATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - heatmap_cross_validation.png")
    print("  - model_robustness.png")
    print("  - data_difficulty.png")
    print("  - diagonal_vs_offdiagonal.png")
    print("  - optimal_training_level.png")
    print("  - distance_analysis.png")
    print("  - cross_validation_report.txt")
    print()
