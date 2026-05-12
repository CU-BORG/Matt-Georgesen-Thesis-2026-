#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Stage Training for LLE Bi-Exponential Autoencoder - SPECTRUM-LITE VERSION

Training on spectrum-lite multi-exponential data with GAUSSIAN-distributed tau values:
  - Tau1: Gaussian(mu=0.4 ns, sigma=0.05 ns)
  - Tau2: Gaussian(mu=4.5 ns, sigma=0.3 ns)
  - Varying fractions

This provides limited spectral variation to test model performance between
spectrumless (no variation) and full spectrum (wide variation).

Based on Visschers et al. 2021 methodology:
  Stage 1: Unsupervised reconstruction - learn to encode/decode signal
  Stage 2: Supervised parameter learning - map latent space to physical parameters
  Stage 3: Joint optimization - balance reconstruction + parameter accuracy

@author:mg
@date: 2025-12-08
"""

# Fix OpenMP library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import datetime
import torch
import numpy as np
import scipy.io as io
import glob
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder
import sys
sys.path.append('../')
try:
    from utils import paramter_initialize
    from early_stopping import EarlyStopping
except ImportError:
    print("Warning: utils.py or early_stopping.py not found in parent directory")
    print("Using fallback implementations...")
    def paramter_initialize(model):
        """Fallback parameter initialization"""
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    class EarlyStopping:
        """Fallback early stopping"""
        def __init__(self, patience=10, verbose=False):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float('inf')

        def __call__(self, epoch, val_loss, model, path, stage=''):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path, stage)
            elif score < self.best_score:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path, stage)
                self.counter = 0

        def save_checkpoint(self, val_loss, model, path, stage):
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), f'{path}/checkpoint_{stage}_best.pth')
            self.val_loss_min = val_loss

from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
try:
    from torchsummary import summary
except ImportError:
    print("Warning: torchsummary not available, skipping model summary")
    def summary(model, input_size, device='cpu'):
        pass


def load_biexp_data_from_directory(data_root, test_ratio=0.2, BATCH_SIZE=128):
    """
    Load bi-exponential FLIM data from directory (SPECTRUM-LITE VERSION)

    Expected .mat file structure:
    - 'Hist': [H, W, time_bins] - decay histograms
    - 'tau_gt_components' or 'tau_components': [H, W, 2] - Gaussian-distributed lifetimes
    - 'f_gt_components' or 'f_components': [H, W, 2] - fraction components (VARYING)
    - (Optional) 'tau_gt_avg': [H, W] - average lifetime

    Returns:
    --------
    train_set, test_set: DataLoaders
    """
    mat_files = glob.glob(os.path.join(data_root, '*.mat'))
    print(f"Found {len(mat_files)} .mat files in {data_root}")

    decays_list = []
    irfs_list = []
    tau1_list = []
    tau2_list = []
    f1_list = []

    for idx, mat_file in enumerate(mat_files):
        if idx % 10 == 0:
            print(f"Processing file {idx}/{len(mat_files)}...")

        try:
            # Try scipy.io first (for older MATLAB files)
            try:
                data = io.loadmat(mat_file)
                use_h5py = False
            except NotImplementedError:
                # Fall back to h5py for MATLAB v7.3 (memory-efficient loading)
                import h5py
                use_h5py = True
                h5file = h5py.File(mat_file, 'r')
                data = h5file  # Keep reference, don't load all at once

            # Extract histogram dimensions
            if 'Hist' in data:
                if use_h5py:
                    # h5py: [time_bins, H, W] format
                    time_bins, H, W = data['Hist'].shape

                    # Sample IRF from first few pixels (memory efficient)
                    irf = np.mean(data['Hist'][:, :10, :10], axis=(1, 2))
                    irf = irf / (irf.max() + 1e-8)

                    # Load parameters (small arrays)
                    tau_components = np.array(data['tau_gt_components'])  # [2, H, W]
                    f_components = np.array(data['f_gt_components'])  # [2, H, W]

                    # Sample decay curves (don't load all pixels)
                    # Sample every 4th pixel to reduce memory
                    sample_step = 4
                    decay_curves = []
                    tau1_vals = []
                    tau2_vals = []
                    f1_vals = []

                    for i in range(0, H, sample_step):
                        for j in range(0, W, sample_step):
                            decay_curve = data['Hist'][:, i, j]  # Load single pixel
                            decay_curves.append(decay_curve)
                            tau1_vals.append(tau_components[0, i, j])
                            tau2_vals.append(tau_components[1, i, j])
                            f1_vals.append(f_components[0, i, j])

                    decay_curves = np.array(decay_curves)  # [N, time_bins]
                    tau1_values = np.array(tau1_vals)
                    tau2_values = np.array(tau2_vals)
                    f1_values = np.array(f1_vals)
                    irfs = np.tile(irf, (len(decay_curves), 1))

                    h5file.close()

                else:
                    # scipy.io format: [H, W, time_bins]
                    Hist = data['Hist']
                    H, W, time_bins = Hist.shape

                    tau_components = data['tau_gt_components']  # [H, W, 2]
                    f_components = data['f_gt_components']

                    # Extract decay curves from each pixel
                    decay_curves = Hist.reshape(H*W, time_bins)

                    # IRF: use sum of all decays (normalized)
                    irf = np.sum(Hist, axis=(0, 1))
                    irf = irf / (irf.max() + 1e-8)
                    irfs = np.tile(irf, (H*W, 1))

                    # Extract parameter values
                    tau1_values = tau_components[:, :, 0].reshape(H*W)
                    tau2_values = tau_components[:, :, 1].reshape(H*W)
                    f1_values = f_components[:, :, 0].reshape(H*W)
            else:
                print(f"Warning: {mat_file} missing Hist, skipping...")
                if use_h5py:
                    h5file.close()
                continue

            # Add to lists
            decays_list.append(decay_curves)
            irfs_list.append(irfs)
            tau1_list.append(tau1_values)
            tau2_list.append(tau2_values)
            f1_list.append(f1_values)

        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Concatenate all data
    all_decays = np.vstack(decays_list).astype(np.float32)
    all_irfs = np.vstack(irfs_list).astype(np.float32)
    all_tau1 = np.hstack(tau1_list).astype(np.float32)
    all_tau2 = np.hstack(tau2_list).astype(np.float32)
    all_f1 = np.hstack(f1_list).astype(np.float32)

    print(f"\nTotal samples: {all_decays.shape[0]}")
    print(f"Decay shape: {all_decays.shape}")
    print(f"Parameter ranges (SPECTRUM-LITE DATA):")
    print(f"  tau1: [{all_tau1.min():.4f}, {all_tau1.max():.4f}] ns (Gaussian around 0.4)")
    print(f"  tau2: [{all_tau2.min():.4f}, {all_tau2.max():.4f}] ns (Gaussian around 4.5)")
    print(f"  f1: [{all_f1.min():.4f}, {all_f1.max():.4f}] (varying)")
    print(f"  tau1 mean: {all_tau1.mean():.4f}, std: {all_tau1.std():.4f}")
    print(f"  tau2 mean: {all_tau2.mean():.4f}, std: {all_tau2.std():.4f}")

    # Normalize decays
    for i in range(all_decays.shape[0]):
        max_val = all_decays[i].max()
        if max_val > 0:
            all_decays[i] = all_decays[i] / max_val

    # Reshape for Conv1d: [N, 1, time_bins]
    all_decays = all_decays.reshape(-1, 1, time_bins)
    all_irfs = all_irfs.reshape(-1, 1, time_bins)

    # Convert to tensors
    decay_tensor = torch.from_numpy(all_decays).float()
    irf_tensor = torch.from_numpy(all_irfs).float()
    tau1_tensor = torch.from_numpy(all_tau1).float()
    tau2_tensor = torch.from_numpy(all_tau2).float()
    f1_tensor = torch.from_numpy(all_f1).float()

    # Create dataset
    torch_dataset = TensorDataset(
        decay_tensor, irf_tensor,
        tau1_tensor, tau2_tensor, f1_tensor
    )

    # Split train/test
    test_size = round(test_ratio * len(torch_dataset))
    train_size = len(torch_dataset) - test_size
    train, test = random_split(torch_dataset, [train_size, test_size])

    train_set = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_set = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_set, test_set


# Import training functions from the main script
from Training_LLE_BiExp_Autoencoder_11_21 import (
    train_stage1_reconstruction,
    train_stage2_parameters,
    train_stage3_joint
)


# =============================================================================
# MAIN TRAINING SCRIPT - SPECTRUM PROGRESSION (10 LEVELS)
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" THREE-STAGE AUTOENCODER TRAINING - SPECTRUM PROGRESSION")
    print(" Following Visschers et al. 2021 methodology")
    print(" Training 10 models on increasing spectral variation:")
    print("   - Level 1: Minimal variation (spectrumless-like)")
    print("   - Levels 2-9: Progressive spectral overlap")
    print("   - Level 10: Full spectral overlap")
    print("=" * 80 + "\n")

    # Define 10 standard deviation levels
    STD_LEVELS = 10
    TAU_1_STD_ARRAY = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05]
    TAU_2_STD_ARRAY = [0.10, 0.30, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50]

    # Model hyperparameters (same for all levels)
    dim = 16
    depth = 8
    ks = 9
    ps = 8

    USE_GPU = True
    total_start = time.time()

    # Base directory for spectrum level datasets
    base_data_dir = r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite'

    print(f"Base data directory: {base_data_dir}")
    print(f"Training {STD_LEVELS} models on {STD_LEVELS} different spectrum levels\n")

    # Main loop: Train one model per spectrum level
    for level_idx in range(1, STD_LEVELS + 1):
        level_start = time.time()

        # Get current level's standard deviations
        tau1_std = TAU_1_STD_ARRAY[level_idx - 1]
        tau2_std = TAU_2_STD_ARRAY[level_idx - 1]

        print("=" * 80)
        print(f"SPECTRUM LEVEL {level_idx}/{STD_LEVELS}")
        print(f"  Tau1_std = {tau1_std:.2f} ns")
        print(f"  Tau2_std = {tau2_std:.2f} ns")
        print("=" * 80)

        # Data directory for this level
        data_root = os.path.join(base_data_dir, f'spectrum_level_{level_idx:02d}_std_{tau1_std:.2f}_{tau2_std:.2f}')

        print(f"\nData directory: {data_root}")
        print(f"Directory exists: {os.path.exists(data_root)}")

        if not os.path.exists(data_root):
            print(f"\nWARNING: Data directory not found! Skipping level {level_idx}")
            print(f"Please ensure the dataset exists at: {data_root}")
            continue

        # Load data for this level
        print(f"\nLoading Level {level_idx} bi-exponential FLIM data...")
        print(f"Expected: Gaussian-distributed tau values (std={tau1_std:.2f}, {tau2_std:.2f})")
        train_set, test_set = load_biexp_data_from_directory(
            data_root, test_ratio=0.2, BATCH_SIZE=128
        )

        # Create level-specific checkpoint directory
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = f'./training_logs_spectrum_level_{level_idx:02d}_{timestamp}'
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\nCheckpoint directory: {ckpt_dir}")

        # Create model (new model for each level)
        print(f"\nInitializing autoencoder model for Level {level_idx}...")
        model = LLE_BiExp_Autoencoder(
            dim=dim,
            depth=depth,
            kernel_size=ks,
            patch_size=ps,
            signal_length=256,
            latent_dim=3
        )

        if level_idx == 1:
            print("\nModel architecture:")
            summary(model, [(1, 256), (1, 256)], device='cpu')

        if USE_GPU:
            model.cuda()

        paramter_initialize(model)

        # =====================================================================
        # THREE-STAGE TRAINING FOR THIS LEVEL
        # =====================================================================

        all_records = {}

        # STAGE 1: Reconstruction
        print("\n" + "=" * 80)
        print(f"LEVEL {level_idx} - STAGE 1: Learning to reconstruct decay signals")
        print("=" * 80)
        record_s1 = train_stage1_reconstruction(
            train_set, test_set, model,
            learning_rate=1e-4,
            Epoch=50,
            USE_GPU=USE_GPU,
            log_interval=50,
            patience=15,
            path_dir=ckpt_dir
        )
        all_records['stage1'] = record_s1

        # Save Stage 1 checkpoint
        torch.save({
            'model': model.state_dict(),
            'record': record_s1,
            'stage': 'stage1_reconstruction',
            'spectrum_level': level_idx,
            'tau1_std': tau1_std,
            'tau2_std': tau2_std
        }, os.path.join(ckpt_dir, 'checkpoint_stage1.pth'))

        # STAGE 2: Parameter Learning
        print("\n" + "=" * 80)
        print(f"LEVEL {level_idx} - STAGE 2: Learning Gaussian-distributed tau and varying fractions")
        print("=" * 80)
        record_s2 = train_stage2_parameters(
            train_set, test_set, model,
            learning_rate=5e-5,
            Epoch=30,
            USE_GPU=USE_GPU,
            log_interval=50,
            patience=10,
            path_dir=ckpt_dir
        )
        all_records['stage2'] = record_s2

        # Save Stage 2 checkpoint
        torch.save({
            'model': model.state_dict(),
            'record': record_s2,
            'stage': 'stage2_parameters',
            'spectrum_level': level_idx,
            'tau1_std': tau1_std,
            'tau2_std': tau2_std
        }, os.path.join(ckpt_dir, 'checkpoint_stage2.pth'))

        # STAGE 3: Joint Optimization
        print("\n" + "=" * 80)
        print(f"LEVEL {level_idx} - STAGE 3: Joint optimization of reconstruction + parameters")
        print("=" * 80)
        record_s3 = train_stage3_joint(
            train_set, test_set, model,
            learning_rate=1e-5,
            Epoch=50,
            USE_GPU=USE_GPU,
            log_interval=50,
            patience=20,
            alpha=0.5,  # Equal weight to reconstruction and parameters
            path_dir=ckpt_dir
        )
        all_records['stage3'] = record_s3

        # Save final model for this level
        torch.save({
            'model': model.state_dict(),
            'all_records': all_records,
            'stage': 'stage3_final',
            'spectrum_level': level_idx,
            'tau1_std': tau1_std,
            'tau2_std': tau2_std,
            'tau_distributions': {
                'tau1': {'mean': 0.4, 'std': tau1_std},
                'tau2': {'mean': 4.5, 'std': tau2_std}
            },
            'hyperparameters': {
                'dim': dim,
                'depth': depth,
                'kernel_size': ks,
                'patch_size': ps,
                'signal_length': 256,
                'latent_dim': 3
            },
            'info': f'Three-stage trained autoencoder on spectrum level {level_idx} (std={tau1_std:.2f}, {tau2_std:.2f})'
        }, os.path.join(ckpt_dir, f'model_final_spectrum_level_{level_idx:02d}.pth'))

        level_time = time.time() - level_start
        print(f"\n{'=' * 80}")
        print(f" LEVEL {level_idx} TRAINING COMPLETED")
        print(f" Level time: {level_time:.2f}s ({level_time/60:.2f} min)")
        print(f" Model saved to: {ckpt_dir}/model_final_spectrum_level_{level_idx:02d}.pth")
        print(f"{'=' * 80}\n")

        # Clean up GPU memory before next level
        if USE_GPU:
            del model
            torch.cuda.empty_cache()

    # End of level loop

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(" ALL TRAINING COMPLETED - SPECTRUM PROGRESSION")
    print("=" * 80)
    print(f"\nTrained {STD_LEVELS} models on {STD_LEVELS} spectrum levels")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min, {total_time/3600:.2f} hours)")
    print(f"\nSpectrum progression:")
    for i in range(STD_LEVELS):
        print(f"  Level {i+1:2d}: tau1_std={TAU_1_STD_ARRAY[i]:.2f} ns, tau2_std={TAU_2_STD_ARRAY[i]:.2f} ns")
    print("\nModels ready for comparative analysis across spectral variation levels!")
    print("=" * 80 + "\n")
