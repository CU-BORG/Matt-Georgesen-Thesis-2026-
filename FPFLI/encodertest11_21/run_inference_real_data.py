"""
Run inference on real FLIM data using the bin 62 trained model.
Compares model predictions vs ground truth from SPC image files.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

try:
    import scipy.io as io
except ImportError:
    io = None

# Import the correct model architecture
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder


# ============================================================================
# Dataset for Real FLIM Data
# ============================================================================

class RealFLIMDataset(Dataset):
    """
    Dataset for real FLIM data packaged in .mat files.
    Each sample contains decay histograms and ground truth parameters from SPC images.
    IRF is computed from histogram data (sum over all pixels, normalized).
    Filters out bottom 20% of pixels by photon count for better signal quality.
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
        self.irf_cache = {}  # Cache IRF per file
        self.photon_thresholds = []  # Track photon thresholds per file

        print(f"Loading {len(mat_files)} real FLIM samples...")
        for mat_file in mat_files:
            self._load_sample(mat_file)

        if len(self.photon_thresholds) > 0:
            avg_threshold = np.mean(self.photon_thresholds)
            print(f"Loaded {len(self.data)} pixels from {len(mat_files)} samples")
            print(f"Average photon threshold (20th percentile): {avg_threshold:.0f}")
        else:
            print(f"Loaded {len(self.data)} pixels from {len(mat_files)} samples")

    def _load_sample(self, mat_file):
        """Load a single .mat file and extract valid pixels"""
        try:
            data = io.loadmat(mat_file)
            hist = data['Hist']  # [H, W, 256]
            tau_gt = data['tau_gt_components']  # [H, W, 2]
            f_gt = data['f_gt_components']  # [H, W, 2]
            photons = data['photons']  # [H, W]

            # Compute IRF from histogram (sum over all spatial pixels)
            irf = np.sum(hist, axis=(0, 1))  # [256]
            irf = irf / (irf.max() + 1e-8)  # Normalize

        except (NotImplementedError, ValueError):
            # MATLAB v7.3 format - not expected for real data, but handle it
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

                # Compute IRF
                irf = np.sum(hist, axis=(0, 1))
                irf = irf / (irf.max() + 1e-8)

        # Cache IRF for this file
        file_key = os.path.basename(mat_file)
        self.irf_cache[file_key] = irf.astype(np.float32)

        H, W, _ = hist.shape

        # PASS 1: Collect all valid pixels with their photon counts
        temp_pixels = []
        for i in range(0, H, self.sample_every):
            for j in range(0, W, self.sample_every):
                # Filter out invalid pixels (tau1/tau2 = 0)
                tau1_gt = tau_gt[i, j, 0]
                tau2_gt = tau_gt[i, j, 1]
                f1_gt_raw = f_gt[i, j, 0]  # Fraction of tau1 (short component)
                f1_gt = 1.0 - f1_gt_raw  # Remap to fraction of tau2 (long/bound component)
                ph = photons[i, j]

                if tau1_gt > 0.01 and tau2_gt > 0.01 and ph > 100:
                    decay = hist[i, j, :]  # [256]

                    # Normalize decay
                    decay_max = decay.max()
                    if decay_max > 0:
                        decay = decay / decay_max

                    temp_pixels.append({
                        'decay': decay.astype(np.float32),
                        'tau1_gt': tau1_gt,
                        'tau2_gt': tau2_gt,
                        'f1_gt': f1_gt,
                        'photons': ph,
                        'sample': file_key
                    })

        # PASS 2: Filter out bottom 20% by photon count
        if len(temp_pixels) > 0:
            photon_counts = np.array([p['photons'] for p in temp_pixels])
            photon_threshold = np.percentile(photon_counts, 20)  # 20th percentile
            self.photon_thresholds.append(photon_threshold)

            for pixel in temp_pixels:
                if pixel['photons'] >= photon_threshold:
                    self.data.append(pixel)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        decay = torch.from_numpy(item['decay']).unsqueeze(0)  # [1, 256]
        irf = torch.from_numpy(self.irf_cache[item['sample']]).unsqueeze(0)  # [1, 256]

        return {
            'decay': decay,
            'irf': irf,
            'tau1_gt': item['tau1_gt'],
            'tau2_gt': item['tau2_gt'],
            'f1_gt': item['f1_gt'],
            'photons': item['photons'],
            'sample': item['sample']
        }


# ============================================================================
# Inference Function
# ============================================================================

def run_inference_on_real_data(model, dataloader, device):
    """
    Run inference on real FLIM data and collect predictions vs ground truth.

    Returns:
        results: dict with predictions and ground truth for each sample
    """
    model.eval()

    results = {
        'tau1_pred': [],
        'tau2_pred': [],
        'f1_pred': [],
        'tau1_gt': [],
        'tau2_gt': [],
        'f1_gt': [],
        'photons': [],
        'samples': []
    }

    with torch.no_grad():
        for batch in dataloader:
            decay = batch['decay'].to(device)  # [B, 1, 256]
            irf = batch['irf'].to(device)  # [B, 1, 256]

            # Run inference with both decay and IRF
            _, latent = model(decay, irf)

            # Extract predictions (apply activation constraints as in training)
            tau1_pred = torch.relu(latent[:, 0])  # tau1 > 0 (short lifetime, free state)
            tau2_pred = torch.relu(latent[:, 1])  # tau2 > 0 (long lifetime, bound state)
            f1_pred_raw = torch.sigmoid(latent[:, 2])  # Fraction of tau1 (short component)
            f1_pred = 1.0 - f1_pred_raw  # Remap to fraction of tau2 (long/bound component)

            # Store results
            results['tau1_pred'].extend(tau1_pred.cpu().numpy())
            results['tau2_pred'].extend(tau2_pred.cpu().numpy())
            results['f1_pred'].extend(f1_pred.cpu().numpy())

            results['tau1_gt'].extend(batch['tau1_gt'].numpy())
            results['tau2_gt'].extend(batch['tau2_gt'].numpy())
            results['f1_gt'].extend(batch['f1_gt'].numpy())

            results['photons'].extend(batch['photons'].numpy())
            results['samples'].extend(batch['sample'])

    # Convert to numpy arrays
    for key in ['tau1_pred', 'tau2_pred', 'f1_pred', 'tau1_gt', 'tau2_gt', 'f1_gt', 'photons']:
        results[key] = np.array(results[key])

    return results


def calculate_metrics(results):
    """Calculate MAE and other metrics"""
    mae_tau1 = np.abs(results['tau1_pred'] - results['tau1_gt']).mean()
    mae_tau2 = np.abs(results['tau2_pred'] - results['tau2_gt']).mean()
    mae_f1 = np.abs(results['f1_pred'] - results['f1_gt']).mean()

    return {
        'mae_tau1': mae_tau1,
        'mae_tau2': mae_tau2,
        'mae_f1': mae_f1
    }


def plot_predictions_vs_gt(results, group_name, save_dir):
    """Create scatter plots of predictions vs ground truth with fixed axis ranges

    Saves both combined multi-panel plot and individual plots for each parameter.
    """
    # Create individual plots subfolder
    individual_dir = os.path.join(save_dir, 'individual_plots')
    os.makedirs(individual_dir, exist_ok=True)

    # Create combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    params = [
        ('tau1_pred', 'tau1_gt', 'tau1', 'tau1 (ns)', axes[0], None, 'blue'),  # Auto range for tau1
        ('tau2_pred', 'tau2_gt', 'tau2', 'tau2 (ns)', axes[1], (0, 10), 'green'),  # Fixed 0-10 ns
        ('f1_pred', 'f1_gt', 'f1', 'f1 (fraction)', axes[2], (0, 1), 'purple')  # Fixed 0-1
    ]

    for pred_key, gt_key, param_name, label, ax, axis_range, color in params:
        pred = results[pred_key]
        gt = results[gt_key]
        mae = np.abs(pred - gt).mean()

        # Determine axis limits
        if axis_range is not None:
            min_val, max_val = axis_range
        else:
            min_val = min(gt.min(), pred.min())
            max_val = max(gt.max(), pred.max())

        # Plot on combined figure
        ax.scatter(gt, pred, alpha=0.3, s=10, c=color)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        if axis_range is not None:
            ax.set_xlim(axis_range)
            ax.set_ylim(axis_range)
        ax.text(0.05, 0.95, f'MAE: {mae:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(f'Ground Truth {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(f'{label} - {group_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Create individual plot for this parameter
        fig_ind = plt.figure(figsize=(6, 5))
        ax_ind = fig_ind.add_subplot(111)

        ax_ind.scatter(gt, pred, alpha=0.3, s=15, c=color)
        ax_ind.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        if axis_range is not None:
            ax_ind.set_xlim(axis_range)
            ax_ind.set_ylim(axis_range)
        ax_ind.text(0.05, 0.95, f'MAE: {mae:.4f}', transform=ax_ind.transAxes,
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_ind.set_xlabel(f'Ground Truth {label}', fontsize=12)
        ax_ind.set_ylabel(f'Predicted {label}', fontsize=12)
        ax_ind.set_title(f'{label} - {group_name}', fontsize=14, fontweight='bold')
        ax_ind.legend(fontsize=10)
        ax_ind.grid(True, alpha=0.3)

        # Save individual plot
        individual_path = os.path.join(individual_dir, f'{group_name}_{param_name}_pred_vs_gt.png')
        fig_ind.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close(fig_ind)

    # Save combined plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{group_name}_predictions_vs_gt.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Configuration
    MODEL_PATH = r'C:\Users\mcg11923\Thesis\training_logs_decay_bin62_20260301_162546\model_final_decay_bin62.pth'
    REAL_DATA_DIR = r'E:\RealdataSCC74A\mat_files'
    OUT_DIR = r'C:\Users\mcg11923\Thesis\training_logs_decay_bin62_20260301_162546\real_data_inference_filtered'

    SAMPLE_EVERY = 4  # Sample every N pixels for speed
    BATCH_SIZE = 64

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on {device.type.upper()}\n")

    # Load model
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}\n")

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
        print(f"  latent_dim: {hyperparams['latent_dim']}\n")

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
        # Default hyperparameters (from training script)
        print("Using default hyperparameters (dim=16, depth=8, ks=9, ps=8)\n")
        model = LLE_BiExp_Autoencoder(
            dim=16,
            depth=8,
            kernel_size=9,
            patch_size=8,
            signal_length=256,
            latent_dim=3
        ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("Model loaded successfully\n")

    # Find all .mat files organized by group
    groups = ['Control_Group', 'FPlusMinus_Group', 'FPlusPlus_Group', 'Rot_Group']

    print("=" * 80)
    print("RUNNING INFERENCE ON REAL FLIM DATA")
    print("=" * 80)
    print(f"Data directory: {REAL_DATA_DIR}")
    print(f"Sampling: every {SAMPLE_EVERY} pixels")
    print(f"Filtering: Bottom 20% of pixels by photon count excluded")
    print(f"Batch size: {BATCH_SIZE}\n")

    all_results = {}
    summary_table = []

    for group in groups:
        group_dir = os.path.join(REAL_DATA_DIR, group)
        if not os.path.exists(group_dir):
            print(f"[WARNING] Group directory not found: {group_dir}\n")
            continue

        mat_files = sorted(glob.glob(os.path.join(group_dir, '*.mat')))
        if len(mat_files) == 0:
            print(f"[WARNING] No .mat files found in {group_dir}\n")
            continue

        print("-" * 80)
        print(f"GROUP: {group}")
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

        # Calculate metrics
        metrics = calculate_metrics(results)

        print(f"Inference completed in {elapsed:.1f}s\n")
        print(f"{group} RESULTS:")
        print(f"  tau1 MAE: {metrics['mae_tau1']:.4f} ns")
        print(f"  tau2 MAE: {metrics['mae_tau2']:.4f} ns")
        print(f"  f1 MAE: {metrics['mae_f1']:.4f}")
        print(f"  Pixels analyzed: {len(results['tau1_pred']):,}\n")

        # Plot predictions vs ground truth
        plot_path = plot_predictions_vs_gt(results, group, OUT_DIR)
        print(f"Plot saved: {plot_path}\n")

        # Store results
        all_results[group] = {
            'results': results,
            'metrics': metrics,
            'n_pixels': len(results['tau1_pred']),
            'n_samples': len(mat_files)
        }

        summary_table.append({
            'group': group,
            'n_samples': len(mat_files),
            'n_pixels': len(results['tau1_pred']),
            'mae_tau1': metrics['mae_tau1'],
            'mae_tau2': metrics['mae_tau2'],
            'mae_f1': metrics['mae_f1']
        })

    # Create summary table
    print("=" * 80)
    print("SUMMARY - ALL GROUPS")
    print("=" * 80)
    print(f"{'Group':<12} {'Samples':<10} {'Pixels':<12} {'tau1 MAE':<12} {'tau2 MAE':<12} {'f1 MAE':<10}")
    print("-" * 80)

    for row in summary_table:
        print(f"{row['group']:<12} {row['n_samples']:<10} {row['n_pixels']:<12,} "
              f"{row['mae_tau1']:<12.4f} {row['mae_tau2']:<12.4f} {row['mae_f1']:<10.4f}")

    # Overall statistics
    total_pixels = sum(row['n_pixels'] for row in summary_table)
    mean_mae_tau1 = np.mean([row['mae_tau1'] for row in summary_table])
    mean_mae_tau2 = np.mean([row['mae_tau2'] for row in summary_table])
    mean_mae_f1 = np.mean([row['mae_f1'] for row in summary_table])

    print("-" * 80)
    print(f"{'OVERALL':<12} {sum(row['n_samples'] for row in summary_table):<10} {total_pixels:<12,} "
          f"{mean_mae_tau1:<12.4f} {mean_mae_tau2:<12.4f} {mean_mae_f1:<10.4f}")
    print("=" * 80)

    # Save summary to file
    summary_path = os.path.join(OUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REAL FLIM DATA INFERENCE SUMMARY - FILTERED (TOP 80% BY SIGNAL)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data: {REAL_DATA_DIR}\n")
        f.write(f"Sampling: every {SAMPLE_EVERY} pixels\n")
        f.write(f"Filtering: Bottom 20% of pixels by photon count excluded\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Group':<12} {'Samples':<10} {'Pixels':<12} {'tau1 MAE':<12} {'tau2 MAE':<12} {'f1 MAE':<10}\n")
        f.write("-" * 80 + "\n")
        for row in summary_table:
            f.write(f"{row['group']:<12} {row['n_samples']:<10} {row['n_pixels']:<12,} "
                   f"{row['mae_tau1']:<12.4f} {row['mae_tau2']:<12.4f} {row['mae_f1']:<10.4f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'OVERALL':<12} {sum(row['n_samples'] for row in summary_table):<10} {total_pixels:<12,} "
               f"{mean_mae_tau1:<12.4f} {mean_mae_tau2:<12.4f} {mean_mae_f1:<10.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"\nSummary saved: {summary_path}")
    print(f"All plots saved to: {OUT_DIR}")
    print("\nInference on real FLIM data complete!")
