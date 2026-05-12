"""
Run inference on real FLIM data with PHOTON-STRATIFIED ANALYSIS - DECAY-ONLY (NO IRF).
32x32 INTERMEDIATE DIMENSIONS VERSION

Stratifies pixels into photon bins to analyze how prediction accuracy
varies with signal strength. This helps identify minimum photon thresholds
for reliable measurements and understand error sources.

MODIFIED VERSION: Uses DECAY-ONLY input (no IRF), no correlation analysis.
                  Uses 32x32 intermediate dimension model.

Photon bins: [50-200], [200-400], [400-600], [600-800], [800-1200]
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import time

try:
    import scipy.io as io
except ImportError:
    io = None

# Import the correct model architecture (DECAY-ONLY, NO IRF, 32x32 DIMS)
from LLE_BiExp_Autoencoder_11_21_NoIRF_32x32 import LLE_BiExp_Autoencoder


# ============================================================================
# Dataset for Real FLIM Data - DECAY-ONLY (NO IRF, NO FILTERING)
# ============================================================================

class RealFLIMDataset(Dataset):
    """
    Dataset for real FLIM data - DECAY-ONLY (NO IRF).
    Loads ALL valid pixels (no photon filtering).
    """
    def __init__(self, mat_files, sample_every=4):
        """
        Args:
            mat_files: List of .mat file paths
            sample_every: Sample every N pixels (default 4 for speed)
        """
        self.mat_files = mat_files
        self.sample_every = sample_every
        self.data = []

        print(f"Loading {len(mat_files)} real FLIM samples (DECAY-ONLY)...")
        for mat_file in mat_files:
            self._load_sample(mat_file)

        print(f"Loaded {len(self.data)} pixels from {len(mat_files)} samples")

    def _load_sample(self, mat_file):
        """Load a single .mat file and extract ALL valid pixels (no photon filtering, no IRF)"""
        try:
            data = io.loadmat(mat_file)
            hist = data['Hist']  # [H, W, 256]
            tau_gt = data['tau_gt_components']  # [H, W, 2]
            f_gt = data['f_gt_components']  # [H, W, 2]
            photons = data['photons']  # [H, W]

        except (NotImplementedError, ValueError):
            # MATLAB v7.3 format
            import h5py
            with h5py.File(mat_file, 'r') as f:
                hist = np.array(f['Hist'])  # [256, H, W]
                tau_gt = np.array(f['tau_gt_components'])  # [2, H, W]
                f_gt = np.array(f['f_gt_components'])  # [2, H, W]
                photons = np.array(f['photons'])  # [H, W]

                # Transpose to [H, W, 256]
                hist = np.transpose(hist, (2, 1, 0))
                tau_gt = np.transpose(tau_gt, (2, 1, 0))
                f_gt = np.transpose(f_gt, (2, 1, 0))
                photons = photons.T

        file_key = os.path.basename(mat_file)
        H, W, _ = hist.shape

        # Collect ALL valid pixels (no photon filtering for stratified analysis)
        for i in range(0, H, self.sample_every):
            for j in range(0, W, self.sample_every):
                # Filter out invalid pixels (tau1/tau2 = 0)
                tau1_gt = tau_gt[i, j, 0]
                tau2_gt = tau_gt[i, j, 1]
                f_gt_raw = f_gt[i, j, 0]  # Fraction of tau1 (short component)
                f_gt_pixel = 1.0 - f_gt_raw  # Remap to fraction of tau2 (long/bound component)
                ph = photons[i, j]

                # Only require positive taus and minimum photons
                if tau1_gt > 0.01 and tau2_gt > 0.01 and ph > 50:
                    decay = hist[i, j, :]  # [256]

                    # Normalize decay
                    decay_max = decay.max()
                    if decay_max > 0:
                        decay = decay / decay_max

                    self.data.append({
                        'decay': decay.astype(np.float32),
                        'tau1_gt': tau1_gt,
                        'tau2_gt': tau2_gt,
                        'f_gt': f_gt_pixel,
                        'photons': ph,
                        'sample': file_key
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        decay = torch.from_numpy(item['decay']).unsqueeze(0)  # [1, 256]

        return {
            'decay': decay,
            'tau1_gt': item['tau1_gt'],
            'tau2_gt': item['tau2_gt'],
            'f_gt': item['f_gt'],
            'photons': item['photons'],
            'sample': item['sample']
        }


# ============================================================================
# Inference Function - DECAY-ONLY
# ============================================================================

def run_inference_on_real_data(model, dataloader, device):
    """
    Run inference on real FLIM data (DECAY-ONLY) and collect model predictions vs Double Exponential Fit results.

    Returns:
        results: dict with model predictions and Double Exponential Fit results for each sample
    """
    model.eval()

    results = {
        'tau1_pred': [],
        'tau2_pred': [],
        'f_pred': [],
        'tau1_gt': [],
        'tau2_gt': [],
        'f_gt': [],
        'photons': [],
        'samples': []
    }

    with torch.no_grad():
        for batch in dataloader:
            decay = batch['decay'].to(device)  # [B, 1, 256]

            # Run inference with DECAY-ONLY (no IRF)
            _, latent = model(decay)

            # Extract predictions (apply activation constraints as in training)
            tau1_pred = torch.relu(latent[:, 0])  # tau1 > 0 (short lifetime, free state)
            tau2_pred = torch.relu(latent[:, 1])  # tau2 > 0 (long lifetime, bound state)
            f_pred = torch.sigmoid(latent[:, 2])  # Model directly predicts f2 (long/bound fraction)

            # Store results
            results['tau1_pred'].extend(tau1_pred.cpu().numpy())
            results['tau2_pred'].extend(tau2_pred.cpu().numpy())
            results['f_pred'].extend(f_pred.cpu().numpy())

            results['tau1_gt'].extend(batch['tau1_gt'].numpy())
            results['tau2_gt'].extend(batch['tau2_gt'].numpy())
            results['f_gt'].extend(batch['f_gt'].numpy())

            results['photons'].extend(batch['photons'].numpy())
            results['samples'].extend(batch['sample'])

    # Convert to numpy arrays
    for key in ['tau1_pred', 'tau2_pred', 'f_pred', 'tau1_gt', 'tau2_gt', 'f_gt', 'photons']:
        results[key] = np.array(results[key])

    return results


# ============================================================================
# Photon Stratification Functions
# ============================================================================

def stratify_results_by_photons(results, photon_bins):
    """
    Stratify results into photon bins.

    Args:
        results: dict with model predictions and Double Exponential Fit results
        photon_bins: list of (min, max) tuples defining bins

    Returns:
        dict mapping bin_name -> results for that bin
    """
    stratified = {}

    for bin_min, bin_max in photon_bins:
        bin_name = f"{bin_min}-{bin_max}"

        # Find pixels in this photon range
        mask = (results['photons'] >= bin_min) & (results['photons'] < bin_max)

        if mask.sum() == 0:
            print(f"  WARNING: No pixels in bin {bin_name}")
            continue

        # Extract results for this bin
        bin_results = {
            'tau1_pred': results['tau1_pred'][mask],
            'tau2_pred': results['tau2_pred'][mask],
            'f_pred': results['f_pred'][mask],
            'tau1_gt': results['tau1_gt'][mask],
            'tau2_gt': results['tau2_gt'][mask],
            'f_gt': results['f_gt'][mask],
            'photons': results['photons'][mask],
            'n_pixels': mask.sum()
        }

        stratified[bin_name] = bin_results

    return stratified


def calculate_stratified_metrics(stratified_results):
    """Calculate MAE for each photon bin"""
    metrics = {}

    for bin_name, bin_results in stratified_results.items():
        mae_tau1 = np.abs(bin_results['tau1_pred'] - bin_results['tau1_gt']).mean()
        mae_tau2 = np.abs(bin_results['tau2_pred'] - bin_results['tau2_gt']).mean()
        mae_f = np.abs(bin_results['f_pred'] - bin_results['f_gt']).mean()

        metrics[bin_name] = {
            'mae_tau1': mae_tau1,
            'mae_tau2': mae_tau2,
            'mae_f': mae_f,
            'n_pixels': bin_results['n_pixels']
        }

    return metrics


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_stratified_predictions(stratified_results, group_name, save_dir):
    """
    Create multi-panel plot showing model predictions vs Double Exponential Fit results for each photon bin.
    Includes MAE vs photon count progression plot.
    Also saves individual plots for each parameter and photon bin.
    """
    n_bins = len(stratified_results)

    # Create individual plots subfolder
    individual_dir = os.path.join(save_dir, 'individual_plots')
    os.makedirs(individual_dir, exist_ok=True)

    # Create figure with 3 rows (tau1, tau2, f) x N columns (one per bin)
    fig, axes = plt.subplots(3, n_bins + 1, figsize=(4*(n_bins+1), 12))

    bin_names = sorted(stratified_results.keys(),
                       key=lambda x: int(x.split('-')[0]))

    # Plot scatter for each bin
    for col_idx, bin_name in enumerate(bin_names):
        bin_results = stratified_results[bin_name]

        # tau1
        ax = axes[0, col_idx]
        # Clip values to 0-1.2 ns range (edge binning)
        tau1_gt_clipped = np.clip(bin_results['tau1_gt'], 0, 1.2)
        tau1_pred_clipped = np.clip(bin_results['tau1_pred'], 0, 1.2)
        ax.scatter(tau1_gt_clipped, tau1_pred_clipped,
                   alpha=0.3, s=5, c='blue')
        ax.plot([0, 1.2], [0, 1.2], 'r--', linewidth=1.5)
        # Add mean marker
        mean_gt = bin_results['tau1_gt'].mean()
        mean_pred = bin_results['tau1_pred'].mean()
        ax.scatter([mean_gt], [mean_pred], c='red', marker='*', s=200,
                   edgecolors='black', linewidths=1.5, zorder=10, label='Mean')
        mae = np.abs(bin_results['tau1_pred'] - bin_results['tau1_gt']).mean()
        ax.set_title(f'{bin_name} photons\nMAE: {mae:.4f}', fontsize=9)
        ax.set_xlabel('Double Exponential Fit tau1 (ns)', fontsize=8)
        ax.set_ylabel('Model tau1 (ns)', fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)

        # Save individual tau1 plot
        fig_ind = plt.figure(figsize=(6, 5))
        ax_ind = fig_ind.add_subplot(111)
        ax_ind.scatter(tau1_gt_clipped, tau1_pred_clipped,
                       alpha=0.3, s=15, c='blue')
        ax_ind.plot([0, 1.2], [0, 1.2], 'r--', linewidth=2)
        # Add mean marker
        ax_ind.scatter([mean_gt], [mean_pred], c='red', marker='*', s=300,
                       edgecolors='black', linewidths=2, zorder=10, label='Mean')
        ax_ind.set_title(f'{group_name} - {bin_name} photons\ntau1 MAE: {mae:.4f}', fontsize=12, fontweight='bold')
        ax_ind.set_xlabel('Double Exponential Fit tau1 (ns)', fontsize=11)
        ax_ind.set_ylabel('Model Prediction tau1 (ns)', fontsize=11)
        ax_ind.set_xlim(0, 1.2)
        ax_ind.set_ylim(0, 1.2)
        ax_ind.grid(True, alpha=0.3)
        ax_ind.legend(loc='upper left', fontsize=9)
        individual_path = os.path.join(individual_dir, f'{group_name}_tau1_photons_{bin_name}.png')
        fig_ind.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close(fig_ind)

        # tau2
        ax = axes[1, col_idx]
        # Clip values to 0-5 ns range (edge binning)
        tau2_gt_clipped = np.clip(bin_results['tau2_gt'], 0, 5)
        tau2_pred_clipped = np.clip(bin_results['tau2_pred'], 0, 5)
        ax.scatter(tau2_gt_clipped, tau2_pred_clipped,
                   alpha=0.3, s=5, c='green')
        ax.plot([0, 5], [0, 5], 'r--', linewidth=1.5)
        # Add mean marker
        mean_gt_tau2 = bin_results['tau2_gt'].mean()
        mean_pred_tau2 = bin_results['tau2_pred'].mean()
        ax.scatter([mean_gt_tau2], [mean_pred_tau2], c='red', marker='*', s=200,
                   edgecolors='black', linewidths=1.5, zorder=10, label='Mean')
        mae = np.abs(bin_results['tau2_pred'] - bin_results['tau2_gt']).mean()
        ax.set_title(f'MAE: {mae:.4f}', fontsize=9)
        ax.set_xlabel('Double Exponential Fit tau2 (ns)', fontsize=8)
        ax.set_ylabel('Model tau2 (ns)', fontsize=8)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.grid(True, alpha=0.3)

        # Save individual tau2 plot
        fig_ind = plt.figure(figsize=(6, 5))
        ax_ind = fig_ind.add_subplot(111)
        ax_ind.scatter(tau2_gt_clipped, tau2_pred_clipped,
                       alpha=0.3, s=15, c='green')
        ax_ind.plot([0, 5], [0, 5], 'r--', linewidth=2)
        # Add mean marker
        ax_ind.scatter([mean_gt_tau2], [mean_pred_tau2], c='red', marker='*', s=300,
                       edgecolors='black', linewidths=2, zorder=10, label='Mean')
        ax_ind.set_xlim(0, 5)
        ax_ind.set_ylim(0, 5)
        ax_ind.set_title(f'{group_name} - {bin_name} photons\ntau2 MAE: {mae:.4f}', fontsize=12, fontweight='bold')
        ax_ind.set_xlabel('Double Exponential Fit tau2 (ns)', fontsize=11)
        ax_ind.set_ylabel('Model Prediction tau2 (ns)', fontsize=11)
        ax_ind.grid(True, alpha=0.3)
        ax_ind.legend(loc='upper left', fontsize=9)
        individual_path = os.path.join(individual_dir, f'{group_name}_tau2_photons_{bin_name}.png')
        fig_ind.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close(fig_ind)

        # f
        ax = axes[2, col_idx]
        ax.scatter(bin_results['f_gt'], bin_results['f_pred'],
                   alpha=0.3, s=5, c='purple')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5)
        # Add mean marker
        mean_gt_f = bin_results['f_gt'].mean()
        mean_pred_f = bin_results['f_pred'].mean()
        ax.scatter([mean_gt_f], [mean_pred_f], c='red', marker='*', s=200,
                   edgecolors='black', linewidths=1.5, zorder=10, label='Mean')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        mae = np.abs(bin_results['f_pred'] - bin_results['f_gt']).mean()
        ax.set_title(f'MAE: {mae:.4f}', fontsize=9)
        ax.set_xlabel('Double Exponential Fit f', fontsize=8)
        ax.set_ylabel('Model f', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Save individual f plot
        fig_ind = plt.figure(figsize=(6, 5))
        ax_ind = fig_ind.add_subplot(111)
        ax_ind.scatter(bin_results['f_gt'], bin_results['f_pred'],
                       alpha=0.3, s=15, c='purple')
        ax_ind.plot([0, 1], [0, 1], 'r--', linewidth=2)
        # Add mean marker
        ax_ind.scatter([mean_gt_f], [mean_pred_f], c='red', marker='*', s=300,
                       edgecolors='black', linewidths=2, zorder=10, label='Mean')
        ax_ind.set_xlim(0, 1)
        ax_ind.set_ylim(0, 1)
        ax_ind.set_title(f'{group_name} - {bin_name} photons\nf MAE: {mae:.4f}', fontsize=12, fontweight='bold')
        ax_ind.set_xlabel('Double Exponential Fit f', fontsize=11)
        ax_ind.set_ylabel('Model Prediction f', fontsize=11)
        ax_ind.grid(True, alpha=0.3)
        ax_ind.legend(loc='upper left', fontsize=9)
        individual_path = os.path.join(individual_dir, f'{group_name}_f_photons_{bin_name}.png')
        fig_ind.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close(fig_ind)

    # MAE progression plots in last column
    bin_centers = [np.mean([int(x) for x in bn.split('-')]) for bn in bin_names]

    mae_tau1_vals = [np.abs(stratified_results[bn]['tau1_pred'] -
                             stratified_results[bn]['tau1_gt']).mean()
                     for bn in bin_names]
    mae_tau2_vals = [np.abs(stratified_results[bn]['tau2_pred'] -
                             stratified_results[bn]['tau2_gt']).mean()
                     for bn in bin_names]
    mae_f_vals = [np.abs(stratified_results[bn]['f_pred'] -
                          stratified_results[bn]['f_gt']).mean()
                  for bn in bin_names]

    # tau1 progression
    axes[0, -1].plot(bin_centers, mae_tau1_vals, 'o-', linewidth=2, markersize=8)
    axes[0, -1].set_xlabel('Photon Count (bin center)', fontsize=8)
    axes[0, -1].set_ylabel('tau1 MAE (ns)', fontsize=8)
    axes[0, -1].set_title('MAE vs Photon Count', fontsize=9)
    axes[0, -1].grid(True, alpha=0.3)

    # Save individual tau1 progression plot
    fig_ind = plt.figure(figsize=(7, 5))
    ax_ind = fig_ind.add_subplot(111)
    ax_ind.plot(bin_centers, mae_tau1_vals, 'o-', linewidth=2.5, markersize=10, color='blue')
    ax_ind.set_xlabel('Photon Count (bin center)', fontsize=11)
    ax_ind.set_ylabel('tau1 MAE (ns)', fontsize=11)
    ax_ind.set_title(f'{group_name} - tau1 MAE vs Photon Count', fontsize=12, fontweight='bold')
    ax_ind.grid(True, alpha=0.3)
    individual_path = os.path.join(individual_dir, f'{group_name}_tau1_mae_vs_photons.png')
    fig_ind.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close(fig_ind)

    # tau2 progression
    axes[1, -1].plot(bin_centers, mae_tau2_vals, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, -1].set_xlabel('Photon Count (bin center)', fontsize=8)
    axes[1, -1].set_ylabel('tau2 MAE (ns)', fontsize=8)
    axes[1, -1].set_title('MAE vs Photon Count', fontsize=9)
    axes[1, -1].grid(True, alpha=0.3)

    # Save individual tau2 progression plot
    fig_ind = plt.figure(figsize=(7, 5))
    ax_ind = fig_ind.add_subplot(111)
    ax_ind.plot(bin_centers, mae_tau2_vals, 'o-', linewidth=2.5, markersize=10, color='green')
    ax_ind.set_xlabel('Photon Count (bin center)', fontsize=11)
    ax_ind.set_ylabel('tau2 MAE (ns)', fontsize=11)
    ax_ind.set_title(f'{group_name} - tau2 MAE vs Photon Count', fontsize=12, fontweight='bold')
    ax_ind.grid(True, alpha=0.3)
    individual_path = os.path.join(individual_dir, f'{group_name}_tau2_mae_vs_photons.png')
    fig_ind.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close(fig_ind)

    # f progression
    axes[2, -1].plot(bin_centers, mae_f_vals, 'o-', linewidth=2, markersize=8, color='purple')
    axes[2, -1].set_xlabel('Photon Count (bin center)', fontsize=8)
    axes[2, -1].set_ylabel('f MAE', fontsize=8)
    axes[2, -1].set_title('MAE vs Photon Count', fontsize=9)
    axes[2, -1].grid(True, alpha=0.3)

    # Save individual f progression plot
    fig_ind = plt.figure(figsize=(7, 5))
    ax_ind = fig_ind.add_subplot(111)
    ax_ind.plot(bin_centers, mae_f_vals, 'o-', linewidth=2.5, markersize=10, color='purple')
    ax_ind.set_xlabel('Photon Count (bin center)', fontsize=11)
    ax_ind.set_ylabel('f MAE', fontsize=11)
    ax_ind.set_title(f'{group_name} - f MAE vs Photon Count', fontsize=12, fontweight='bold')
    ax_ind.grid(True, alpha=0.3)
    individual_path = os.path.join(individual_dir, f'{group_name}_f_mae_vs_photons.png')
    fig_ind.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close(fig_ind)

    plt.suptitle(f'Model Prediction vs Double Exponential Fit Results - {group_name}\n(Photon-Stratified Analysis, DECAY-ONLY, 32x32 Dims)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{group_name}_photon_stratified_32x32.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_mean_std_comparison(stratified_results, group_name, save_dir):
    """
    Create bar chart comparing mean ± Std of Double Exponential Fit vs Model predictions.

    Args:
        stratified_results: dict mapping bin_name -> results for that bin
        group_name: Name of the group (e.g., 'Control_Group')
        save_dir: Directory to save plots

    Returns:
        Path to the saved combined plot
    """
    individual_dir = os.path.join(save_dir, 'individual_plots')
    os.makedirs(individual_dir, exist_ok=True)

    bin_names = sorted(stratified_results.keys(),
                       key=lambda x: int(x.split('-')[0]))
    n_bins = len(bin_names)

    # Prepare data for plotting
    tau1_spc_means = []
    tau1_spc_stds = []
    tau1_model_means = []
    tau1_model_stds = []

    tau2_spc_means = []
    tau2_spc_stds = []
    tau2_model_means = []
    tau2_model_stds = []

    f_spc_means = []
    f_spc_stds = []
    f_model_means = []
    f_model_stds = []

    for bin_name in bin_names:
        bin_results = stratified_results[bin_name]
        n = len(bin_results['tau1_gt'])

        # tau1
        tau1_spc_means.append(bin_results['tau1_gt'].mean())
        tau1_spc_stds.append(bin_results['tau1_gt'].std())  # Std
        tau1_model_means.append(bin_results['tau1_pred'].mean())
        tau1_model_stds.append(bin_results['tau1_pred'].std())  # Std

        # tau2
        tau2_spc_means.append(bin_results['tau2_gt'].mean())
        tau2_spc_stds.append(bin_results['tau2_gt'].std())  # Std
        tau2_model_means.append(bin_results['tau2_pred'].mean())
        tau2_model_stds.append(bin_results['tau2_pred'].std())  # Std

        # f
        f_spc_means.append(bin_results['f_gt'].mean())
        f_spc_stds.append(bin_results['f_gt'].std())  # Std
        f_model_means.append(bin_results['f_pred'].mean())
        f_model_stds.append(bin_results['f_pred'].std())  # Std

    # Create combined 3-panel bar chart
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    x = np.arange(n_bins)
    width = 0.35

    # tau1
    ax = axes[0]
    ax.bar(x - width/2, tau1_spc_means, width, yerr=tau1_spc_stds,
           label='Double Exponential Fit', color='steelblue', capsize=5, alpha=0.8)
    ax.bar(x + width/2, tau1_model_means, width, yerr=tau1_model_stds,
           label='Model (32x32)', color='coral', capsize=5, alpha=0.8)
    ax.set_ylabel(r'$\tau_1$ (ns)', fontsize=12, fontweight='bold')
    ax.set_title(f'{group_name} - Mean ± Std Comparison (DECAY-ONLY, 32x32 Dims)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # tau2
    ax = axes[1]
    ax.bar(x - width/2, tau2_spc_means, width, yerr=tau2_spc_stds,
           label='Double Exponential Fit', color='steelblue', capsize=5, alpha=0.8)
    ax.bar(x + width/2, tau2_model_means, width, yerr=tau2_model_stds,
           label='Model (32x32)', color='coral', capsize=5, alpha=0.8)
    ax.set_ylabel(r'$\tau_2$ (ns)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # f
    ax = axes[2]
    ax.bar(x - width/2, f_spc_means, width, yerr=f_spc_stds,
           label='Double Exponential Fit', color='steelblue', capsize=5, alpha=0.8)
    ax.bar(x + width/2, f_model_means, width, yerr=f_model_stds,
           label='Model (32x32)', color='coral', capsize=5, alpha=0.8)
    ax.set_ylabel('f', fontsize=12, fontweight='bold')
    ax.set_xlabel('Photon Bin', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{group_name}_mean_std_comparison_32x32.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save individual parameter plots
    # tau1 individual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, tau1_spc_means, width, yerr=tau1_spc_stds,
           label='Double Exponential Fit', color='steelblue', capsize=5, alpha=0.8)
    ax.bar(x + width/2, tau1_model_means, width, yerr=tau1_model_stds,
           label='Model (32x32)', color='coral', capsize=5, alpha=0.8)
    ax.set_ylabel(r'$\tau_1$ (ns)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Photon Bin', fontsize=12, fontweight='bold')
    ax.set_title(f'{group_name} - ' + r'$\tau_1$ Mean ± Std Comparison (DECAY-ONLY, 32x32 Dims)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    individual_path = os.path.join(individual_dir, f'{group_name}_tau1_mean_std_32x32.png')
    plt.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close()

    # tau2 individual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, tau2_spc_means, width, yerr=tau2_spc_stds,
           label='Double Exponential Fit', color='steelblue', capsize=5, alpha=0.8)
    ax.bar(x + width/2, tau2_model_means, width, yerr=tau2_model_stds,
           label='Model (32x32)', color='coral', capsize=5, alpha=0.8)
    ax.set_ylabel(r'$\tau_2$ (ns)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Photon Bin', fontsize=12, fontweight='bold')
    ax.set_title(f'{group_name} - ' + r'$\tau_2$ Mean ± Std Comparison (DECAY-ONLY, 32x32 Dims)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    individual_path = os.path.join(individual_dir, f'{group_name}_tau2_mean_std_32x32.png')
    plt.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close()

    # f individual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, f_spc_means, width, yerr=f_spc_stds,
           label='Double Exponential Fit', color='steelblue', capsize=5, alpha=0.8)
    ax.bar(x + width/2, f_model_means, width, yerr=f_model_stds,
           label='Model (32x32)', color='coral', capsize=5, alpha=0.8)
    ax.set_ylabel('f', fontsize=12, fontweight='bold')
    ax.set_xlabel('Photon Bin', fontsize=12, fontweight='bold')
    ax.set_title(f'{group_name} - f Mean ± Std Comparison (DECAY-ONLY, 32x32 Dims)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    individual_path = os.path.join(individual_dir, f'{group_name}_f_mean_std_32x32.png')
    plt.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Configuration
    MODEL_PATH = r'C:\Users\mcg11923\Thesis\training_logs_decay_bin62_12.5ns_f2_longfrac_NoIRF_32x32_20260509_093731\model_final_decay_bin62_12.5ns_f2_longfrac_NoIRF_32x32.pth'
    REAL_DATA_DIR = r'E:\RealdataSCC74A\mat_files'
    OUT_DIR = r'C:\Users\mcg11923\Thesis\training_logs_decay_bin62_12.5ns_f2_longfrac_NoIRF_32x32_20260509_093731\real_data_photon_stratified_NoIRF_32x32'

    # Photon bins for stratification
    PHOTON_BINS = [
        (50, 200),
        (200, 400),
        (400, 600),
        (600, 800),
        (800, 1200)
    ]

    SAMPLE_EVERY = 4  # Sample every N pixels for speed
    BATCH_SIZE = 64

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on {device.type.upper()}\n")

    # Load model
    print("=" * 80)
    print("LOADING MODEL (DECAY-ONLY, NO IRF, 32x32 DIMS)")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}\n")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to the trained NoIRF 32x32 model checkpoint.")
        print("You need to train the model first using Training_LLE_BiExp_Autoencoder_DecayBin50_LongFraction_NoIRF_32x32.py")
        sys.exit(1)

    # Load checkpoint to get hyperparameters
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Get hyperparameters from checkpoint
    if 'hyperparameters' in checkpoint:
        hyperparams = checkpoint['hyperparameters']
        print(f"Model hyperparameters from checkpoint:")
        print(f"  dim: {hyperparams['dim']}")
        print(f"  depth: {hyperparams['depth']}")
        print(f"  kernel_size: {hyperparams['kernel_size']}")
        print(f"  patch_size: {hyperparams['patch_size']}")
        print(f"  latent_dim: {hyperparams['latent_dim']}")
        print(f"  Intermediate dimensions: {hyperparams['dim']} channels × 32 time bins\n")

        # Create model with saved hyperparameters
        model = LLE_BiExp_Autoencoder(
            dim=hyperparams['dim'],
            depth=hyperparams['depth'],
            kernel_size=hyperparams['kernel_size'],
            patch_size=hyperparams['patch_size'],
            signal_length=256,
            latent_dim=hyperparams['latent_dim']
        ).to(device)
    else:
        # Default hyperparameters for 32x32 model
        print("Using default hyperparameters (dim=32, depth=8, ks=9, ps=8)\n")
        model = LLE_BiExp_Autoencoder(
            dim=32,
            depth=8,
            kernel_size=9,
            patch_size=8,
            signal_length=256,
            latent_dim=3
        ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("Model loaded successfully (DECAY-ONLY, 32x32 DIMS)\n")

    # Find all .mat files organized by group
    groups = ['Control_Group', 'FPlusMinus_Group', 'FPlusPlus_Group', 'Rot_Group']

    # Map directory names to display names
    group_display_names = {
        'Control_Group': 'Routine Culture',
        'FPlusMinus_Group': 'FPlusMinus_Group',
        'FPlusPlus_Group': 'FPlusPlus_Group',
        'Rot_Group': 'Rot_Group'
    }

    print("=" * 80)
    print("PHOTON-STRATIFIED INFERENCE (DECAY-ONLY, NO IRF, 32x32 DIMS)")
    print("=" * 80)
    print(f"Data directory: {REAL_DATA_DIR}")
    print(f"Sampling: every {SAMPLE_EVERY} pixels")
    print(f"Photon bins: {PHOTON_BINS}")
    print(f"Batch size: {BATCH_SIZE}\n")

    all_group_results = {}

    for group in groups:
        group_dir = os.path.join(REAL_DATA_DIR, group)
        display_name = group_display_names[group]
        if not os.path.exists(group_dir):
            print(f"[WARNING] Group directory not found: {group_dir}\n")
            continue

        mat_files = sorted(glob.glob(os.path.join(group_dir, '*.mat')))
        if len(mat_files) == 0:
            print(f"[WARNING] No .mat files found in {group_dir}\n")
            continue

        print("-" * 80)
        print(f"GROUP: {display_name}")
        print("-" * 80)
        print(f"Found {len(mat_files)} samples\n")

        # Create dataset and dataloader
        dataset = RealFLIMDataset(mat_files, sample_every=SAMPLE_EVERY)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"Total pixels: {len(dataset):,}\n")

        # Run inference
        start_time = time.time()
        results = run_inference_on_real_data(model, dataloader, device)
        elapsed = time.time() - start_time

        print(f"Inference completed in {elapsed:.1f}s\n")

        # Stratify by photon count
        print("Stratifying by photon count...")
        stratified_results = stratify_results_by_photons(results, PHOTON_BINS)

        # Calculate metrics for each bin
        stratified_metrics = calculate_stratified_metrics(stratified_results)

        print(f"\n{display_name} PHOTON-STRATIFIED RESULTS:")
        print(f"{'Photon Bin':<15} {'N Pixels':<10} {'tau1 MAE':<12} {'tau2 MAE':<12} {'f MAE':<10}")
        print("-" * 70)
        for bin_name in sorted(stratified_metrics.keys(), key=lambda x: int(x.split('-')[0])):
            metrics = stratified_metrics[bin_name]
            print(f"{bin_name:<15} {metrics['n_pixels']:<10} "
                  f"{metrics['mae_tau1']:<12.4f} {metrics['mae_tau2']:<12.4f} {metrics['mae_f']:<10.4f}")
        print()

        # Plot stratified predictions
        plot_path = plot_stratified_predictions(stratified_results, display_name, OUT_DIR)
        print(f"Stratified plot saved: {plot_path}\n")

        # Plot mean ± std comparison
        mean_std_path = plot_mean_std_comparison(stratified_results, display_name, OUT_DIR)
        print(f"Mean ± Std comparison saved: {mean_std_path}\n")

        # Store results
        all_group_results[display_name] = {
            'stratified_results': stratified_results,
            'stratified_metrics': stratified_metrics,
            'n_samples': len(mat_files),
            'total_pixels': len(results['tau1_pred'])
        }

    # Save summary
    summary_path = os.path.join(OUT_DIR, 'photon_stratified_summary_NoIRF_32x32.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PHOTON-STRATIFIED INFERENCE (DECAY-ONLY, NO IRF, 32x32 DIMS)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data: {REAL_DATA_DIR}\n")
        f.write(f"Photon bins: {PHOTON_BINS}\n")
        f.write(f"Input: DECAY-ONLY (no IRF)\n")
        f.write(f"Model: 32x32 intermediate dimensions\n\n")

        for group, group_data in all_group_results.items():
            f.write("-" * 80 + "\n")
            f.write(f"GROUP: {group}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Samples: {group_data['n_samples']}\n")
            f.write(f"Total pixels: {group_data['total_pixels']}\n\n")

            f.write(f"{'Photon Bin':<15} {'N Pixels':<10} {'tau1 MAE':<12} {'tau2 MAE':<12} {'f MAE':<10}\n")
            f.write("-" * 70 + "\n")

            for bin_name in sorted(group_data['stratified_metrics'].keys(),
                                   key=lambda x: int(x.split('-')[0])):
                metrics = group_data['stratified_metrics'][bin_name]
                f.write(f"{bin_name:<15} {metrics['n_pixels']:<10} "
                       f"{metrics['mae_tau1']:<12.4f} {metrics['mae_tau2']:<12.4f} {metrics['mae_f']:<10.4f}\n")
            f.write("\n")

    print("=" * 80)
    print("PHOTON-STRATIFIED ANALYSIS COMPLETE (DECAY-ONLY, NO IRF, 32x32 DIMS)")
    print("=" * 80)
    print(f"Summary saved: {summary_path}")
    print(f"All plots saved to: {OUT_DIR}")
    print("\nPhoton-stratified inference complete!")
