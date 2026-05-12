#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot MAE Across All Spectrum Levels on Same Test Samples

Tests all 10 models on the same random samples and plots how prediction
error changes across the spectrum progression.

@author: MAE comparison script
"""

# Fix OpenMP library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder

# Configuration
STD_LEVELS = 10
TAU_1_STD_ARRAY = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05]
TAU_2_STD_ARRAY = [0.10, 0.30, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50]

BASE_DATA_DIR = r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite'
BASE_MODEL_DIR = r'C:\Users\mcg11923\Thesis'

NUM_RANDOM_SAMPLES = 5  # Number of random samples to test
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

    return model, checkpoint


def get_random_samples(data_dir, num_samples=5):
    """Get random sample files from a data directory"""
    pattern = os.path.join(data_dir, 'Sample_*_spectrumspectrum.mat')
    all_files = glob.glob(pattern)

    if len(all_files) == 0:
        return []

    # Select random files
    np.random.seed(42)  # Fixed seed for reproducibility
    num_samples = min(num_samples, len(all_files))
    selected_files = np.random.choice(all_files, num_samples, replace=False)

    return list(selected_files)


def compute_mae_for_sample(model, sample_path):
    """Compute MAE for tau1, tau2, f1 on a single sample"""
    # Load data using h5py for MATLAB v7.3 files
    with h5py.File(sample_path, 'r') as f:
        Hist = np.array(f['Hist']).T  # [256, 256, 256] - transpose for correct shape
        tau_gt = np.array(f['tau_gt_components']).T  # [256, 256, 2]
        f_gt_data = np.array(f['f_gt_components']).T  # [256, 256, 2]

    H, W, T = Hist.shape

    # Apply signal threshold
    total_photons = np.sum(Hist, axis=2)  # [256, 256]
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

    decay_batch = np.array(decay_batch)  # [N, 256]
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

            # Convert to torch
            batch_decay_tensor = torch.FloatTensor(batch_decay).unsqueeze(1)  # [B, 1, 256]
            irf_tensor = torch.zeros_like(batch_decay_tensor)  # Dummy IRF

            # Predict
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


def create_mae_plots(results, output_dir):
    """Create comprehensive MAE plots"""
    print("\nCreating MAE comparison plots...")

    levels = sorted(results.keys())
    num_samples = len(results[levels[0]])

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # Extract data
    tau1_stds = [TAU_1_STD_ARRAY[level-1] for level in levels]
    tau2_stds = [TAU_2_STD_ARRAY[level-1] for level in levels]

    # =========================================================================
    # Plot 1: MAE vs Level for all samples (Tau1)
    # =========================================================================
    ax1 = plt.subplot(3, 3, 1)
    for sample_idx in range(num_samples):
        mae_tau1 = [results[level][sample_idx]['mae_tau1'] for level in levels]
        ax1.plot(levels, mae_tau1, 'o-', label=f'Sample {sample_idx+1}', linewidth=2, markersize=6)

    ax1.set_xlabel('Spectrum Level', fontsize=12)
    ax1.set_ylabel('Tau1 MAE (ns)', fontsize=12)
    ax1.set_title('Tau1 Prediction Error vs Spectrum Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(levels)

    # =========================================================================
    # Plot 2: MAE vs Level for all samples (Tau2)
    # =========================================================================
    ax2 = plt.subplot(3, 3, 2)
    for sample_idx in range(num_samples):
        mae_tau2 = [results[level][sample_idx]['mae_tau2'] for level in levels]
        ax2.plot(levels, mae_tau2, 's-', label=f'Sample {sample_idx+1}', linewidth=2, markersize=6)

    ax2.set_xlabel('Spectrum Level', fontsize=12)
    ax2.set_ylabel('Tau2 MAE (ns)', fontsize=12)
    ax2.set_title('Tau2 Prediction Error vs Spectrum Level', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(levels)

    # =========================================================================
    # Plot 3: MAE vs Level for all samples (F1)
    # =========================================================================
    ax3 = plt.subplot(3, 3, 3)
    for sample_idx in range(num_samples):
        mae_f1 = [results[level][sample_idx]['mae_f1'] for level in levels]
        ax3.plot(levels, mae_f1, '^-', label=f'Sample {sample_idx+1}', linewidth=2, markersize=6)

    ax3.set_xlabel('Spectrum Level', fontsize=12)
    ax3.set_ylabel('F1 MAE', fontsize=12)
    ax3.set_title('Fraction Prediction Error vs Spectrum Level', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(levels)

    # =========================================================================
    # Plot 4: Average MAE vs Level (all parameters)
    # =========================================================================
    ax4 = plt.subplot(3, 3, 4)

    # Compute average across samples
    avg_mae_tau1 = [np.mean([results[level][s]['mae_tau1'] for s in range(num_samples)]) for level in levels]
    avg_mae_tau2 = [np.mean([results[level][s]['mae_tau2'] for s in range(num_samples)]) for level in levels]
    avg_mae_f1 = [np.mean([results[level][s]['mae_f1'] for s in range(num_samples)]) for level in levels]

    ax4.plot(levels, avg_mae_tau1, 'o-', label='Tau1', linewidth=3, markersize=8)
    ax4.plot(levels, avg_mae_tau2, 's-', label='Tau2', linewidth=3, markersize=8)
    ax4.plot(levels, avg_mae_f1, '^-', label='F1', linewidth=3, markersize=8)

    ax4.set_xlabel('Spectrum Level', fontsize=12)
    ax4.set_ylabel('Average MAE', fontsize=12)
    ax4.set_title('Average Prediction Error (All Parameters)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(levels)

    # =========================================================================
    # Plot 5: Tau1 MAE vs Tau1 Std
    # =========================================================================
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(tau1_stds, avg_mae_tau1, 'o-', linewidth=3, markersize=10, color='blue')
    ax5.set_xlabel('Tau1 Standard Deviation (ns)', fontsize=12)
    ax5.set_ylabel('Tau1 MAE (ns)', fontsize=12)
    ax5.set_title('Tau1 Error vs Data Variation', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 6: Tau2 MAE vs Tau2 Std
    # =========================================================================
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(tau2_stds, avg_mae_tau2, 's-', linewidth=3, markersize=10, color='orange')
    ax6.set_xlabel('Tau2 Standard Deviation (ns)', fontsize=12)
    ax6.set_ylabel('Tau2 MAE (ns)', fontsize=12)
    ax6.set_title('Tau2 Error vs Data Variation', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 7: Error bars (std across samples)
    # =========================================================================
    ax7 = plt.subplot(3, 3, 7)

    std_mae_tau1 = [np.std([results[level][s]['mae_tau1'] for s in range(num_samples)]) for level in levels]
    std_mae_tau2 = [np.std([results[level][s]['mae_tau2'] for s in range(num_samples)]) for level in levels]

    ax7.errorbar(levels, avg_mae_tau1, yerr=std_mae_tau1, fmt='o-', label='Tau1',
                 linewidth=2, markersize=8, capsize=5)
    ax7.errorbar(levels, avg_mae_tau2, yerr=std_mae_tau2, fmt='s-', label='Tau2',
                 linewidth=2, markersize=8, capsize=5)

    ax7.set_xlabel('Spectrum Level', fontsize=12)
    ax7.set_ylabel('MAE ± Std Dev', fontsize=12)
    ax7.set_title('Error Variation Across Samples', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks(levels)

    # =========================================================================
    # Plot 8: Relative error increase (normalized to level 1)
    # =========================================================================
    ax8 = plt.subplot(3, 3, 8)

    tau1_relative = np.array(avg_mae_tau1) / avg_mae_tau1[0] * 100
    tau2_relative = np.array(avg_mae_tau2) / avg_mae_tau2[0] * 100
    f1_relative = np.array(avg_mae_f1) / avg_mae_f1[0] * 100

    ax8.plot(levels, tau1_relative, 'o-', label='Tau1', linewidth=2, markersize=8)
    ax8.plot(levels, tau2_relative, 's-', label='Tau2', linewidth=2, markersize=8)
    ax8.plot(levels, f1_relative, '^-', label='F1', linewidth=2, markersize=8)
    ax8.axhline(y=100, color='gray', linestyle='--', label='Baseline (Level 1)')

    ax8.set_xlabel('Spectrum Level', fontsize=12)
    ax8.set_ylabel('Relative Error (%)', fontsize=12)
    ax8.set_title('Error Increase Relative to Level 1', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks(levels)

    # =========================================================================
    # Plot 9: Summary statistics table
    # =========================================================================
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = "Summary Statistics\n\n"
    summary_text += f"Samples tested: {num_samples}\n"
    summary_text += f"Levels analyzed: {len(levels)}\n\n"

    summary_text += "Average MAE at Level 1:\n"
    summary_text += f"  Tau1: {avg_mae_tau1[0]:.6f} ns\n"
    summary_text += f"  Tau2: {avg_mae_tau2[0]:.6f} ns\n"
    summary_text += f"  F1:   {avg_mae_f1[0]:.6f}\n\n"

    summary_text += "Average MAE at Level 10:\n"
    summary_text += f"  Tau1: {avg_mae_tau1[-1]:.6f} ns\n"
    summary_text += f"  Tau2: {avg_mae_tau2[-1]:.6f} ns\n"
    summary_text += f"  F1:   {avg_mae_f1[-1]:.6f}\n\n"

    summary_text += "Error Increase (Level 10 / Level 1):\n"
    summary_text += f"  Tau1: {tau1_relative[-1]/100:.2f}x\n"
    summary_text += f"  Tau2: {tau2_relative[-1]/100:.2f}x\n"
    summary_text += f"  F1:   {f1_relative[-1]/100:.2f}x\n"

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle('MAE Comparison Across Spectrum Levels (Same Test Samples)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'mae_comparison_across_levels.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved MAE comparison plot to: {save_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print(" MAE COMPARISON ACROSS SPECTRUM LEVELS")
    print("="*80)
    print(f"\nTesting all {STD_LEVELS} models on {NUM_RANDOM_SAMPLES} random samples")
    print("="*80 + "\n")

    # Get random samples from level 1 dataset (we'll test all models on these)
    tau1_std = TAU_1_STD_ARRAY[0]
    tau2_std = TAU_2_STD_ARRAY[0]
    data_dir_level1 = os.path.join(BASE_DATA_DIR, f'spectrum_level_01_std_{tau1_std:.2f}_{tau2_std:.2f}')

    print(f"Selecting {NUM_RANDOM_SAMPLES} random samples from Level 1 dataset...")
    test_samples = get_random_samples(data_dir_level1, NUM_RANDOM_SAMPLES)

    if len(test_samples) == 0:
        print("ERROR: No test samples found!")
        exit(1)

    print(f"Selected {len(test_samples)} samples:")
    for i, sample in enumerate(test_samples):
        print(f"  {i+1}. {os.path.basename(sample)}")

    # Dictionary to store results: results[level][sample_idx] = {'mae_tau1': ..., 'mae_tau2': ..., 'mae_f1': ...}
    results = {}

    # Test each model on the same samples
    for level_idx in range(1, STD_LEVELS + 1):
        tau1_std = TAU_1_STD_ARRAY[level_idx - 1]
        tau2_std = TAU_2_STD_ARRAY[level_idx - 1]

        print(f"\n{'='*80}")
        print(f"LEVEL {level_idx}/{STD_LEVELS}: Testing model (std={tau1_std:.2f}, {tau2_std:.2f})")
        print(f"{'='*80}")

        # Load model
        model_data = load_model(level_idx)
        if model_data is None:
            print(f"WARNING: Model not found for level {level_idx}, skipping...")
            continue

        model, checkpoint = model_data
        print(f"[OK] Model loaded successfully")

        # Test on each sample
        level_results = []
        for sample_idx, sample_path in enumerate(test_samples):
            print(f"  Testing on sample {sample_idx+1}/{len(test_samples)}...", end='')

            mae_result = compute_mae_for_sample(model, sample_path)

            if mae_result is None:
                print(" [SKIP - No valid pixels]")
                level_results.append({'mae_tau1': np.nan, 'mae_tau2': np.nan, 'mae_f1': np.nan})
            else:
                print(f" [OK] tau1={mae_result['mae_tau1']:.6f}, tau2={mae_result['mae_tau2']:.6f}, f1={mae_result['mae_f1']:.6f}")
                level_results.append(mae_result)

        results[level_idx] = level_results

    # Create output directory
    output_dir = os.path.join(BASE_MODEL_DIR, 'spectrum_mae_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Create plots
    create_mae_plots(results, output_dir)

    # Save detailed results to text file
    report_path = os.path.join(output_dir, 'mae_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MAE COMPARISON ACROSS SPECTRUM LEVELS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test samples: {NUM_RANDOM_SAMPLES}\n")
        f.write(f"Signal threshold: {SIGNAL_THRESHOLD} photons\n\n")

        f.write("Test samples:\n")
        for i, sample in enumerate(test_samples):
            f.write(f"  {i+1}. {os.path.basename(sample)}\n")
        f.write("\n")

        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")

        for level_idx in sorted(results.keys()):
            tau1_std = TAU_1_STD_ARRAY[level_idx - 1]
            tau2_std = TAU_2_STD_ARRAY[level_idx - 1]

            f.write(f"Level {level_idx:2d} (tau1_std={tau1_std:.2f}, tau2_std={tau2_std:.2f}):\n")

            for sample_idx, mae_result in enumerate(results[level_idx]):
                f.write(f"  Sample {sample_idx+1}: ")
                f.write(f"tau1_MAE={mae_result['mae_tau1']:.6f}, ")
                f.write(f"tau2_MAE={mae_result['mae_tau2']:.6f}, ")
                f.write(f"f1_MAE={mae_result['mae_f1']:.6f}\n")

            # Compute averages
            avg_tau1 = np.nanmean([r['mae_tau1'] for r in results[level_idx]])
            avg_tau2 = np.nanmean([r['mae_tau2'] for r in results[level_idx]])
            avg_f1 = np.nanmean([r['mae_f1'] for r in results[level_idx]])

            f.write(f"  Average: tau1={avg_tau1:.6f}, tau2={avg_tau2:.6f}, f1={avg_f1:.6f}\n\n")

    print(f"\n[OK] Saved detailed report to: {report_path}")

    print("\n" + "="*80)
    print(" MAE COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - mae_comparison_across_levels.png")
    print("  - mae_comparison_report.txt")
    print()
