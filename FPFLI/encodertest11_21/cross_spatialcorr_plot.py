#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Correlation Analysis: All 10 Models × All 10 Data Levels

Tests each trained model on data from all 10 spectrum levels to create a
10×10 matrix showing how well models generalize across different spectral overlaps.

Key analyses:
- Model performance on its own training distribution (diagonal)
- Generalization to other distributions (off-diagonal)
- Transfer learning patterns
- Optimal model selection

@author: Cross-correlation analysis script
@date: 2025-12-23
"""

import os
import sys
import glob
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder
import time

# Configuration
BASE_DIR = r'C:\Users\mcg11923\Thesis'
DATA_BASE_DIR = r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite'

SPECTRUM_LEVELS = 10
TAU_1_STD_ARRAY = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05]
TAU_2_STD_ARRAY = [0.10, 0.30, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50]

NUM_TEST_SAMPLES = 3  # Test images per data level
SIGNAL_THRESHOLD = 50  # Minimum photons for valid pixels

# Model architecture (must match training)
MODEL_DIM = 16
MODEL_DEPTH = 8
MODEL_KS = 9
MODEL_PS = 8


def find_model_paths():
    """
    Find all 10 trained model paths

    Returns:
    --------
    dict: {level: model_path}
    """
    model_paths = {}

    for level in range(1, SPECTRUM_LEVELS + 1):
        pattern = os.path.join(BASE_DIR, f'training_logs_spatialcorr_level_{level:02d}_*')
        matching_dirs = glob.glob(pattern)

        if not matching_dirs:
            print(f"Warning: No training directory found for Level {level}")
            continue

        # Use most recent directory
        train_dir = sorted(matching_dirs)[-1]

        # Find model file
        model_files = glob.glob(os.path.join(train_dir, 'model_final_spatialcorr_level_*.pth'))
        if not model_files:
            print(f"Warning: No model file found in {train_dir}")
            continue

        model_paths[level] = model_files[0]

    return model_paths


def find_data_paths():
    """
    Find all 10 data directories

    Returns:
    --------
    dict: {level: data_path}
    """
    data_paths = {}

    for level in range(1, SPECTRUM_LEVELS + 1):
        tau1_std = TAU_1_STD_ARRAY[level - 1]
        tau2_std = TAU_2_STD_ARRAY[level - 1]

        pattern = os.path.join(DATA_BASE_DIR, f'spectrum_level_{level:02d}_std_{tau1_std:.2f}_{tau2_std:.2f}_spatialcorr')
        matching_dirs = glob.glob(pattern)

        if not matching_dirs:
            print(f"Warning: No data directory found for Level {level}")
            continue

        data_paths[level] = matching_dirs[0]

    return data_paths


def load_model(model_path, device):
    """
    Load trained model from checkpoint

    Parameters:
    -----------
    model_path : str
        Path to model checkpoint
    device : torch.device
        Device to load model on

    Returns:
    --------
    model : LLE_BiExp_Autoencoder
    """
    model = LLE_BiExp_Autoencoder(
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        kernel_size=MODEL_KS,
        patch_size=MODEL_PS,
        signal_length=256,
        latent_dim=3
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def load_test_images(data_dir, num_samples=3):
    """
    Load test images from data directory

    Parameters:
    -----------
    data_dir : str
        Path to data directory
    num_samples : int
        Number of test images to load

    Returns:
    --------
    list of dicts: [{'Hist': ..., 'tau_gt': ..., 'f_gt': ...}, ...]
    """
    mat_files = sorted(glob.glob(os.path.join(data_dir, '*.mat')))[:num_samples]

    if not mat_files:
        print(f"Warning: No .mat files found in {data_dir}")
        return []

    test_data = []

    for mat_file in mat_files:
        try:
            # Try scipy.io first
            import scipy.io as sio
            data = sio.loadmat(mat_file)
            Hist = data['Hist']  # [H, W, time_bins]
            tau_gt = data['tau_gt_components']  # [H, W, 2]
            f_gt = data['f_gt_components']  # [H, W, 2]
        except (NotImplementedError, KeyError):
            # Fall back to h5py
            try:
                with h5py.File(mat_file, 'r') as f:
                    Hist = np.array(f['Hist'])
                    tau_gt = np.array(f['tau_gt_components'])
                    f_gt = np.array(f['f_gt_components'])

                # Transpose h5py format
                Hist = np.transpose(Hist, (1, 2, 0))
                tau_gt = np.transpose(tau_gt, (1, 2, 0))
                f_gt = np.transpose(f_gt, (1, 2, 0))
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
                continue

        test_data.append({
            'Hist': Hist,
            'tau_gt': tau_gt,
            'f_gt': f_gt,
            'filename': os.path.basename(mat_file)
        })

    return test_data


def test_model_on_data(model, test_data, device):
    """
    Test model on data and calculate MAE

    Parameters:
    -----------
    model : LLE_BiExp_Autoencoder
    test_data : list of dicts
        Test images
    device : torch.device

    Returns:
    --------
    dict: {'tau1_mae': float, 'tau2_mae': float, 'f1_mae': float, 'combined_mae': float}
    """
    all_tau1_errors = []
    all_tau2_errors = []
    all_f1_errors = []

    for data in test_data:
        Hist = data['Hist']
        tau_gt = data['tau_gt']
        f_gt = data['f_gt']

        H, W, time_bins = Hist.shape

        # Create IRF
        irf = np.mean(Hist, axis=(0, 1))
        irf = irf / (irf.max() + 1e-8)

        # Prepare pixels
        decay_curves = Hist.reshape(H * W, time_bins)
        decay_curves = decay_curves / (decay_curves.max(axis=1, keepdims=True) + 1e-8)
        irfs = np.tile(irf, (H * W, 1))

        # Convert to tensors
        decay_tensor = torch.from_numpy(decay_curves).unsqueeze(1).float().to(device)
        irf_tensor = torch.from_numpy(irfs).unsqueeze(1).float().to(device)

        # Predict
        batch_size = 1024
        tau1_pred_list = []
        tau2_pred_list = []
        f1_pred_list = []

        with torch.no_grad():
            for i in range(0, len(decay_tensor), batch_size):
                batch_decay = decay_tensor[i:i + batch_size]
                batch_irf = irf_tensor[i:i + batch_size]

                params = model.extract_parameters(batch_decay, batch_irf)
                tau1_pred_list.append(params['tau1'].cpu())
                tau2_pred_list.append(params['tau2'].cpu())
                f1_pred_list.append(params['f1'].cpu())

        # Concatenate
        tau1_pred = torch.cat(tau1_pred_list).numpy().reshape(H, W)
        tau2_pred = torch.cat(tau2_pred_list).numpy().reshape(H, W)
        f1_pred = torch.cat(f1_pred_list).numpy().reshape(H, W)

        # Ground truth
        tau1_gt = tau_gt[:, :, 0]
        tau2_gt = tau_gt[:, :, 1]
        f1_gt = f_gt[:, :, 0]

        # Signal filtering
        total_signal = Hist.sum(axis=2)
        valid_mask = (tau1_gt > 1e-6) & (tau2_gt > 1e-6) & (total_signal >= SIGNAL_THRESHOLD)

        if valid_mask.sum() == 0:
            print(f"  Warning: No valid pixels in {data['filename']}")
            continue

        # Calculate errors
        tau1_error = np.abs(tau1_pred[valid_mask] - tau1_gt[valid_mask])
        tau2_error = np.abs(tau2_pred[valid_mask] - tau2_gt[valid_mask])
        f1_error = np.abs(f1_pred[valid_mask] - f1_gt[valid_mask])

        all_tau1_errors.extend(tau1_error)
        all_tau2_errors.extend(tau2_error)
        all_f1_errors.extend(f1_error)

    if not all_tau1_errors:
        return {
            'tau1_mae': np.nan,
            'tau2_mae': np.nan,
            'f1_mae': np.nan,
            'combined_mae': np.nan
        }

    # Calculate MAE
    tau1_mae = np.mean(all_tau1_errors)
    tau2_mae = np.mean(all_tau2_errors)
    f1_mae = np.mean(all_f1_errors)
    combined_mae = (tau1_mae + tau2_mae + f1_mae) / 3.0

    return {
        'tau1_mae': tau1_mae,
        'tau2_mae': tau2_mae,
        'f1_mae': f1_mae,
        'combined_mae': combined_mae,
        'num_pixels': len(all_tau1_errors)
    }


def create_heatmaps(matrices, output_dir):
    """
    Create heatmap visualizations

    Parameters:
    -----------
    matrices : dict
        {'combined': matrix, 'tau1': matrix, 'tau2': matrix, 'f1': matrix}
    output_dir : str
        Output directory
    """
    metrics = [
        ('combined', 'Combined MAE (avg of τ₁, τ₂, f₁)', 'RdYlGn_r'),
        ('tau1', 'τ₁ MAE (ns)', 'RdYlGn_r'),
        ('tau2', 'τ₂ MAE (ns)', 'RdYlGn_r'),
        ('f1', 'f₁ MAE (fraction)', 'RdYlGn_r')
    ]

    for metric_key, title, cmap in metrics:
        matrix = matrices[metric_key]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.4f', cmap=cmap,
                    xticklabels=range(1, SPECTRUM_LEVELS + 1),
                    yticklabels=range(1, SPECTRUM_LEVELS + 1),
                    cbar_kws={'label': 'MAE'},
                    linewidths=0.5, linecolor='gray',
                    ax=ax)

        ax.set_xlabel('Model Level (Training)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Data Level (Test)', fontsize=14, fontweight='bold')
        ax.set_title(f'Cross-Correlation Analysis: {title}', fontsize=16, fontweight='bold', pad=20)

        # Add std labels on axes
        tau1_labels = [f'L{i}\nσ₁={TAU_1_STD_ARRAY[i-1]:.2f}' for i in range(1, SPECTRUM_LEVELS + 1)]
        tau2_labels = [f'σ₂={TAU_2_STD_ARRAY[i-1]:.2f}' for i in range(1, SPECTRUM_LEVELS + 1)]

        # Add secondary axis labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.arange(SPECTRUM_LEVELS) + 0.5)
        ax2.set_xticklabels(tau2_labels, fontsize=8)
        ax2.set_xlabel('Model Training std (τ₂)', fontsize=10, fontweight='bold')

        ax3 = ax.twinx()
        ax3.set_ylim(ax.get_ylim())
        ax3.set_yticks(np.arange(SPECTRUM_LEVELS) + 0.5)
        ax3.set_yticklabels(tau2_labels, fontsize=8)
        ax3.set_ylabel('Test Data std (τ₂)', fontsize=10, fontweight='bold')

        plt.tight_layout()

        save_path = os.path.join(output_dir, f'cross_correlation_heatmap_{metric_key}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved heatmap: {save_path}")
        plt.close()


def create_generalization_plot(combined_matrix, output_dir):
    """
    Create generalization analysis plot

    Shows diagonal vs off-diagonal performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Diagonal performance (models on their own data)
    ax1 = axes[0, 0]
    diagonal = np.diag(combined_matrix)
    ax1.plot(range(1, SPECTRUM_LEVELS + 1), diagonal, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Spectrum Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (on own data)', fontsize=12, fontweight='bold')
    ax1.set_title('Diagonal Performance (Model on Own Data)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, SPECTRUM_LEVELS + 1))

    # Plot 2: Off-diagonal degradation
    ax2 = axes[0, 1]
    mean_off_diag = []
    for i in range(SPECTRUM_LEVELS):
        off_diag_vals = np.concatenate([combined_matrix[i, :i], combined_matrix[i, i+1:]])
        mean_off_diag.append(np.mean(off_diag_vals))

    degradation = [(mean_off_diag[i] - diagonal[i]) / diagonal[i] * 100 for i in range(SPECTRUM_LEVELS)]
    ax2.bar(range(1, SPECTRUM_LEVELS + 1), degradation, color='orange', alpha=0.7)
    ax2.set_xlabel('Spectrum Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Off-Diagonal Performance Degradation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(1, SPECTRUM_LEVELS + 1))
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Plot 3: Best model per data level
    ax3 = axes[1, 0]
    best_models = np.argmin(combined_matrix, axis=1) + 1
    ax3.plot(range(1, SPECTRUM_LEVELS + 1), best_models, 'o-', linewidth=2, markersize=8, color='green')
    ax3.plot([1, SPECTRUM_LEVELS], [1, SPECTRUM_LEVELS], 'r--', alpha=0.5, label='Diagonal (same level)')
    ax3.set_xlabel('Data Level (Test)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Best Model Level', fontsize=12, fontweight='bold')
    ax3.set_title('Optimal Model Selection per Data Level', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, SPECTRUM_LEVELS + 1))
    ax3.set_yticks(range(1, SPECTRUM_LEVELS + 1))
    ax3.legend()

    # Plot 4: Generalization score (average performance across all data)
    ax4 = axes[1, 1]
    avg_performance = np.mean(combined_matrix, axis=0)  # Average over data levels (rows)
    ax4.bar(range(1, SPECTRUM_LEVELS + 1), avg_performance, color='purple', alpha=0.7)
    best_universal = np.argmin(avg_performance) + 1
    ax4.axvline(x=best_universal, color='red', linestyle='--', linewidth=2,
                label=f'Best Universal Model: Level {best_universal}')
    ax4.set_xlabel('Model Level', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average MAE (across all data levels)', fontsize=12, fontweight='bold')
    ax4.set_title('Model Generalization Score', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(range(1, SPECTRUM_LEVELS + 1))
    ax4.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'generalization_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved generalization analysis: {save_path}")
    plt.close()


def create_report(matrices, output_dir):
    """
    Create statistical analysis report
    """
    combined = matrices['combined']

    with open(os.path.join(output_dir, 'cross_correlation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" CROSS-CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Diagonal analysis
        f.write("## DIAGONAL PERFORMANCE (Model on Own Data)\n\n")
        diagonal = np.diag(combined)
        for i in range(SPECTRUM_LEVELS):
            f.write(f"Level {i+1}: MAE = {diagonal[i]:.6f}\n")
        f.write(f"\nBest diagonal performance: Level {np.argmin(diagonal) + 1} (MAE = {np.min(diagonal):.6f})\n")
        f.write(f"Worst diagonal performance: Level {np.argmax(diagonal) + 1} (MAE = {np.max(diagonal):.6f})\n\n")

        # Generalization analysis
        f.write("=" * 80 + "\n")
        f.write("## GENERALIZATION ANALYSIS\n\n")

        avg_performance = np.mean(combined, axis=0)
        best_universal = np.argmin(avg_performance)
        f.write(f"Best universal model: Level {best_universal + 1}\n")
        f.write(f"  Average MAE across all data: {avg_performance[best_universal]:.6f}\n\n")

        # Transfer learning patterns
        f.write("=" * 80 + "\n")
        f.write("## TRANSFER LEARNING PATTERNS\n\n")

        for data_level in range(SPECTRUM_LEVELS):
            best_model = np.argmin(combined[data_level, :])
            f.write(f"Data Level {data_level + 1}:\n")
            f.write(f"  Best model: Level {best_model + 1} (MAE = {combined[data_level, best_model]:.6f})\n")
            f.write(f"  Own model: Level {data_level + 1} (MAE = {combined[data_level, data_level]:.6f})\n")
            if best_model != data_level:
                improvement = (combined[data_level, data_level] - combined[data_level, best_model]) / combined[data_level, data_level] * 100
                f.write(f"  Improvement: {improvement:.2f}% by using Model {best_model + 1}\n")
            f.write("\n")

        # Asymmetry analysis
        f.write("=" * 80 + "\n")
        f.write("## ASYMMETRY ANALYSIS\n\n")

        f.write("Low overlap model on high overlap data vs vice versa:\n\n")
        low_on_high = combined[9, 0]  # Data level 10, Model level 1
        high_on_low = combined[0, 9]  # Data level 1, Model level 10
        f.write(f"Model 1 (low std) on Data 10 (high std): MAE = {low_on_high:.6f}\n")
        f.write(f"Model 10 (high std) on Data 1 (low std): MAE = {high_on_low:.6f}\n")
        f.write(f"Ratio: {low_on_high / high_on_low:.2f}x\n\n")

        if low_on_high > high_on_low:
            f.write("Finding: Models trained on high variation generalize better to low variation.\n")
        else:
            f.write("Finding: Models trained on low variation struggle more with high variation.\n")

    print(f"[OK] Saved report: {os.path.join(output_dir, 'cross_correlation_report.txt')}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" CROSS-CORRELATION ANALYSIS: 10 MODELS × 10 DATA LEVELS")
    print("=" * 80)

    # Find models and data
    print("\nSearching for models and data...")
    model_paths = find_model_paths()
    data_paths = find_data_paths()

    print(f"\nFound {len(model_paths)} models")
    print(f"Found {len(data_paths)} data directories")

    if len(model_paths) != SPECTRUM_LEVELS or len(data_paths) != SPECTRUM_LEVELS:
        print("\nWARNING: Not all models or data found!")
        print(f"Models found: {sorted(model_paths.keys())}")
        print(f"Data found: {sorted(data_paths.keys())}")

    # Initialize matrices
    combined_matrix = np.full((SPECTRUM_LEVELS, SPECTRUM_LEVELS), np.nan)
    tau1_matrix = np.full((SPECTRUM_LEVELS, SPECTRUM_LEVELS), np.nan)
    tau2_matrix = np.full((SPECTRUM_LEVELS, SPECTRUM_LEVELS), np.nan)
    f1_matrix = np.full((SPECTRUM_LEVELS, SPECTRUM_LEVELS), np.nan)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = os.path.join(BASE_DIR, 'spatialcorr_analysis', 'cross_correlation')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Main loop: test each model on each data level
    print("\n" + "=" * 80)
    print(" RUNNING CROSS-CORRELATION TESTS (10×10 = 100 tests)")
    print("=" * 80 + "\n")

    total_start = time.time()
    total_tests = 0
    successful_tests = 0

    for model_level in sorted(model_paths.keys()):
        print(f"\n{'─' * 80}")
        print(f"MODEL LEVEL {model_level}")
        print(f"{'─' * 80}")

        # Load model
        print(f"Loading model from: {os.path.basename(os.path.dirname(model_paths[model_level]))}")
        model = load_model(model_paths[model_level], device)

        for data_level in sorted(data_paths.keys()):
            test_start = time.time()
            total_tests += 1

            print(f"  Testing on Data Level {data_level}...", end=' ')

            # Load test data
            test_data = load_test_images(data_paths[data_level], NUM_TEST_SAMPLES)

            if not test_data:
                print("[SKIP - No data]")
                continue

            # Test
            try:
                results = test_model_on_data(model, test_data, device)

                combined_matrix[data_level - 1, model_level - 1] = results['combined_mae']
                tau1_matrix[data_level - 1, model_level - 1] = results['tau1_mae']
                tau2_matrix[data_level - 1, model_level - 1] = results['tau2_mae']
                f1_matrix[data_level - 1, model_level - 1] = results['f1_mae']

                test_time = time.time() - test_start
                print(f"[OK] MAE={results['combined_mae']:.4f} ({test_time:.1f}s, {results['num_pixels']:,} px)")
                successful_tests += 1

            except Exception as e:
                print(f"[ERROR] {e}")

        # Clean up
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print(" TESTING COMPLETE")
    print("=" * 80)
    print(f"\nTotal tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save matrices
    print("\nSaving matrices...")
    np.savetxt(os.path.join(output_dir, 'cross_correlation_combined.csv'), combined_matrix, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(output_dir, 'cross_correlation_tau1.csv'), tau1_matrix, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(output_dir, 'cross_correlation_tau2.csv'), tau2_matrix, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(output_dir, 'cross_correlation_f1.csv'), f1_matrix, delimiter=',', fmt='%.6f')
    print("[OK] Saved CSV matrices")

    # Create visualizations
    print("\nGenerating visualizations...")
    matrices = {
        'combined': combined_matrix,
        'tau1': tau1_matrix,
        'tau2': tau2_matrix,
        'f1': f1_matrix
    }

    create_heatmaps(matrices, output_dir)
    create_generalization_plot(combined_matrix, output_dir)

    # Create report
    print("\nGenerating report...")
    create_report(matrices, output_dir)

    print("\n" + "=" * 80)
    print(" CROSS-CORRELATION ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - cross_correlation_heatmap_combined.png")
    print("  - cross_correlation_heatmap_tau1.png")
    print("  - cross_correlation_heatmap_tau2.png")
    print("  - cross_correlation_heatmap_f1.png")
    print("  - generalization_analysis.png")
    print("  - cross_correlation_*.csv (4 files)")
    print("  - cross_correlation_report.txt")
    print()
