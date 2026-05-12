#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Stage Training for LLE Bi-Exponential Autoencoder - FULLY FLAT RANDOM (BATCH)

Training script for new fully flat random datasets with different sample sizes
Supports datasets with: 1, 10, 100, 200, 500, 1000 images

Usage:
    python Training_LLE_BiExp_Autoencoder_FullyFlatRandom_Batch.py <sample_size>

Example:
    python Training_LLE_BiExp_Autoencoder_FullyFlatRandom_Batch.py 100

Dataset specifications:
  - Tau1: Uniform[0.20, 0.70] ns (NO spatial correlation) - UPDATED RANGE
  - Tau2: Uniform[1.20, 4.50] ns (NO spatial correlation)
  - Fraction f1: Uniform[0.1, 0.9] (NO spatial correlation)
  - ALL parameters completely independent per-pixel
  - Data stored on E drive

Based on Visschers et al. 2021 methodology:
  Stage 1: Unsupervised reconstruction - learn to encode/decode signal
  Stage 2: Supervised parameter learning - map latent space to physical parameters
  Stage 3: Joint optimization - balance reconstruction + parameter accuracy

@author: mg
"""

# Fix OpenMP library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import datetime
import torch
import numpy as np
import sys
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder

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

try:
    from torchsummary import summary
except ImportError:
    print("Warning: torchsummary not available, skipping model summary")
    def summary(model, input_size, device='cpu'):
        pass

# Import the shared data loader and all three training-stage functions
from Training_LLE_BiExp_Autoencoder_SpatialCorr import load_biexp_data_from_directory
from Training_LLE_BiExp_Autoencoder_11_21 import (
    train_stage1_reconstruction,
    train_stage2_parameters,
    train_stage3_joint
)


# =============================================================================
# MAIN TRAINING SCRIPT - FULLY FLAT RANDOM (batch sample sizes)
# =============================================================================
if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python Training_LLE_BiExp_Autoencoder_FullyFlatRandom_Batch.py <sample_size>")
        print("Available sample sizes: 1, 10, 100, 200, 500, 1000")
        print("\nExample:")
        print("  python Training_LLE_BiExp_Autoencoder_FullyFlatRandom_Batch.py 100")
        sys.exit(1)

    try:
        sample_size = int(sys.argv[1])
    except ValueError:
        print(f"ERROR: Invalid sample size '{sys.argv[1]}'. Must be an integer.")
        sys.exit(1)

    valid_sizes = [1, 10, 100, 200, 500, 1000]
    if sample_size not in valid_sizes:
        print(f"ERROR: Sample size {sample_size} not available.")
        print(f"Available sizes: {valid_sizes}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(" THREE-STAGE AUTOENCODER TRAINING - FULLY FLAT RANDOM")
    print(" Following Visschers et al. 2021 methodology")
    print(f" Dataset : fully_flatrandom_n{sample_size:04d} ({sample_size} files, 256x256, 256 bins)")
    print(" Tau1    : Uniform[0.20, 0.70] ns (NO spatial correlation) - UPDATED")
    print(" Tau2    : Uniform[1.20, 4.50] ns (NO spatial correlation)")
    print(" Fraction: Uniform[0.1, 0.9] (NO spatial correlation)")
    print(" All parameters completely independent per-pixel")
    print("=" * 80 + "\n")

    # =========================================================================
    # MODEL HYPERPARAMETERS  (same as all other runs)
    # =========================================================================
    dim   = 16
    depth = 8
    ks    = 9
    ps    = 8

    USE_GPU = True
    total_start = time.time()

    # =========================================================================
    # DATA PATH (E DRIVE)
    # =========================================================================
    data_root = f'E:\\fully_flatrandom_n{sample_size:04d}'

    print(f"Data directory : {data_root}")
    print(f"Directory exists: {os.path.exists(data_root)}")

    if not os.path.exists(data_root):
        print(f"\nERROR: Data directory not found!")
        print(f"Please ensure the dataset exists at: {data_root}")
        print(f"Run Generate_MultiExp_Batch_SampleSizes.m first to create the datasets.")
        sys.exit(1)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print(f"\nLoading fully_flatrandom_n{sample_size:04d} FLIM data...")

    # Adjust test ratio based on sample size
    if sample_size == 1:
        test_ratio = 0.0  # Use all data for training when only 1 image
        print("  WARNING: Only 1 image - using it for both training and testing")
    elif sample_size <= 10:
        test_ratio = 0.1  # Use 10% for testing with small datasets
    else:
        test_ratio = 0.2  # Standard 20% for larger datasets

    train_set, test_set = load_biexp_data_from_directory(
        data_root, test_ratio=test_ratio, BATCH_SIZE=128
    )

    print(f"  Training samples: {len(train_set.dataset) if hasattr(train_set.dataset, '__len__') else 'N/A'}")
    print(f"  Test ratio: {test_ratio:.1%}")

    # =========================================================================
    # CHECKPOINT DIRECTORY
    # =========================================================================
    now       = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    ckpt_dir  = f'./training_logs_fully_flatrandom_n{sample_size:04d}_{timestamp}'
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint directory: {ckpt_dir}")

    # Save dataset info
    dataset_info_path = os.path.join(ckpt_dir, 'dataset_info.txt')
    with open(dataset_info_path, 'w') as f:
        f.write("FULLY FLAT RANDOM DATASET INFO (UPDATED RANGES)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: fully_flatrandom_n{sample_size:04d}\n")
        f.write(f"Total images: {sample_size} (256x256 pixels, 256 time-bins)\n")
        f.write(f"Test ratio: {test_ratio:.1%}\n\n")
        f.write("Parameter Distributions (all UNIFORM, NO spatial correlation):\n")
        f.write("  Tau1: Uniform[0.20, 0.70] ns - UPDATED (narrower for better learning)\n")
        f.write("  Tau2: Uniform[1.20, 4.50] ns\n")
        f.write("  Fraction f1: Uniform[0.1, 0.9]\n\n")
        f.write("Key Features:\n")
        f.write("  - Each pixel completely independent\n")
        f.write("  - NO spatial correlation for any parameter\n")
        f.write("  - Updated tau1 range [0.2, 0.7] vs old [0.05, 0.9]\n")
        f.write("  - All histograms approximately uniform\n")
        f.write("  - Data stored on E drive\n")

    # =========================================================================
    # MODEL
    # =========================================================================
    print("\nInitializing autoencoder model...")
    model = LLE_BiExp_Autoencoder(
        dim=dim,
        depth=depth,
        kernel_size=ks,
        patch_size=ps,
        signal_length=256,
        latent_dim=3
    )

    print("\nModel architecture:")
    summary(model, [(1, 256), (1, 256)], device='cpu')

    # Save model architecture
    arch_path = os.path.join(ckpt_dir, 'model_architecture.txt')
    with open(arch_path, 'w') as f:
        f.write("LLE Bi-Exponential Autoencoder Architecture\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  dim (channel dimension): {dim}\n")
        f.write(f"  depth (number of blocks): {depth}\n")
        f.write(f"  kernel_size: {ks}\n")
        f.write(f"  patch_size: {ps}\n")
        f.write(f"  signal_length: 256\n")
        f.write(f"  latent_dim: 3 (tau1, tau2, f1)\n\n")
        f.write(str(model))

    if USE_GPU:
        model.cuda()

    paramter_initialize(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Save parameter count
    param_count_path = os.path.join(ckpt_dir, 'parameter_count.txt')
    with open(param_count_path, 'w') as f:
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Model size (float32): ~{total_params * 4 / (1024**2):.2f} MB\n")

    # =========================================================================
    # THREE-STAGE TRAINING
    # =========================================================================
    all_records = {}

    # ------------------------------------------------------------------
    # STAGE 1: Reconstruction
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 1: Learning to reconstruct decay signals")
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

    torch.save({
        'model': model.state_dict(),
        'record': record_s1,
        'stage': 'stage1_reconstruction',
        'dataset': f'fully_flatrandom_n{sample_size:04d}'
    }, os.path.join(ckpt_dir, 'checkpoint_stage1.pth'))

    # ------------------------------------------------------------------
    # STAGE 2: Parameter Learning
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 2: Learning tau1, tau2, f1 from fully flat random ground truth")
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

    torch.save({
        'model': model.state_dict(),
        'record': record_s2,
        'stage': 'stage2_parameters',
        'dataset': f'fully_flatrandom_n{sample_size:04d}'
    }, os.path.join(ckpt_dir, 'checkpoint_stage2.pth'))

    # ------------------------------------------------------------------
    # STAGE 3: Joint Optimization
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 3: Joint optimisation (reconstruction + parameters)")
    print("=" * 80)
    record_s3 = train_stage3_joint(
        train_set, test_set, model,
        learning_rate=1e-5,
        Epoch=50,
        USE_GPU=USE_GPU,
        log_interval=50,
        patience=20,
        alpha=0.5,
        path_dir=ckpt_dir
    )
    all_records['stage3'] = record_s3

    # ------------------------------------------------------------------
    # SAVE FINAL MODEL
    # ------------------------------------------------------------------
    torch.save({
        'model': model.state_dict(),
        'all_records': all_records,
        'stage': 'stage3_final',
        'dataset': f'fully_flatrandom_n{sample_size:04d}',
        'sample_size': sample_size,
        'hyperparameters': {
            'dim': dim,
            'depth': depth,
            'kernel_size': ks,
            'patch_size': ps,
            'signal_length': 256,
            'latent_dim': 3
        },
        'data_info': {
            'tau1_range': [0.20, 0.70],  # UPDATED
            'tau2_range': [1.20, 4.50],
            'f1_range': [0.1, 0.9],
            'distribution': 'uniform',
            'spatial_correlation': False,
            'description': 'Fully flat random - updated tau1 range [0.2, 0.7] for better learning'
        },
        'info': f'Three-stage trained autoencoder on fully_flatrandom_n{sample_size:04d} dataset (Visschers et al. 2021)'
    }, os.path.join(ckpt_dir, f'model_final_fully_flatrandom_n{sample_size:04d}.pth'))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f" TRAINING COMPLETED - FULLY FLAT RANDOM (n={sample_size})")
    print(f"{'=' * 80}")
    print(f"  Total time   : {total_time:.2f} s  ({total_time/60:.1f} min)")
    print(f"  Checkpoints  : {ckpt_dir}/")
    print(f"  Final model  : {ckpt_dir}/model_final_fully_flatrandom_n{sample_size:04d}.pth")
    print(f"  Dataset info : {ckpt_dir}/dataset_info.txt")
    print(f"  Architecture : {ckpt_dir}/model_architecture.txt")
    print("=" * 80 + "\n")

    # Save final summary
    summary_path = os.path.join(ckpt_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"TRAINING SUMMARY - FULLY FLAT RANDOM (n={sample_size})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: fully_flatrandom_n{sample_size:04d}\n")
        f.write(f"Sample size: {sample_size} images\n")
        f.write(f"Test ratio: {test_ratio:.1%}\n")
        f.write(f"Total training time: {total_time:.2f} s ({total_time/60:.1f} min)\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Model hyperparameters:\n")
        f.write(f"  dim: {dim}\n")
        f.write(f"  depth: {depth}\n")
        f.write(f"  kernel_size: {ks}\n")
        f.write(f"  patch_size: {ps}\n\n")
        f.write("Training stages:\n")
        f.write(f"  Stage 1 (Reconstruction): {len(record_s1['Train_Loss'])} epochs\n")
        f.write(f"  Stage 2 (Parameters): {len(record_s2['Train_Loss'])} epochs\n")
        f.write(f"  Stage 3 (Joint): {len(record_s3['Train_Loss'])} epochs\n\n")
        f.write("Data characteristics:\n")
        f.write("  Tau1: Uniform[0.20, 0.70] ns - UPDATED RANGE\n")
        f.write("  Tau2: Uniform[1.20, 4.50] ns\n")
        f.write("  Fraction f1: Uniform[0.1, 0.9]\n")
        f.write("  Spatial correlation: None\n")
        f.write("  All parameters independent per-pixel\n")
        f.write("  Data location: E drive\n")

    print(f"Training summary saved to: {summary_path}\n")

    # Clean up GPU memory
    if USE_GPU:
        del model
        torch.cuda.empty_cache()
