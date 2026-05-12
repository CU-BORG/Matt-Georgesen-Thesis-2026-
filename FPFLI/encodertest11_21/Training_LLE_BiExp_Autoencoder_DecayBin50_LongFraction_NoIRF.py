#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Stage Training for LLE Bi-Exponential Autoencoder - DECAY PEAK BIN 62
TRAINING FOR LONG LIFETIME FRACTION (f2) - DECAY-ONLY INPUT (NO IRF)

Single-model training run on the decay_peak_bin62_12.5ns_trainfixed dataset:
  - 100 .mat files (256x256 pixels, 256 time-bins each)
  - IRF centered at bin 62 (MATCHED TO REAL DATA)
  - Tau1: Uniform[0.20, 0.70] ns (SHORT lifetime, NO spatial correlation)
  - Tau2: Uniform[1.20, 4.50] ns (LONG lifetime, NO spatial correlation)
  - TRAINING TARGET: f2 (LONG lifetime fraction) = Uniform[0.1, 0.9]
  - Ground-truth tau1, tau2, f2 per pixel
  - Pixel sub-sampled at step=4 inside the loader (~4096 pixels/file)
  - ALL parameters completely independent per-pixel

MODIFIED VERSION: This version uses DECAY-ONLY input (no IRF)

TEMPORAL ALIGNMENT TO REAL DATA:
  - Training with decay peak at bin 62 (matches real FLIM data)
  - Real data analysis showed decay peaks at bin 62
  - Purpose: Train model with correct temporal alignment for real data inference

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
import scipy.io as io
import glob
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split
from LLE_BiExp_Autoencoder_11_21_NoIRF import LLE_BiExp_Autoencoder

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

from torch.nn import MSELoss
from torch.optim import Adam


# =============================================================================
# CUSTOM DATA LOADER - DECAY-ONLY (NO IRF), EXTRACTS f2 (LONG LIFETIME FRACTION)
# =============================================================================
def load_biexp_data_f2_from_directory_NoIRF(data_root, test_ratio=0.2, BATCH_SIZE=128):
    """
    Load bi-exponential FLIM data from directory - DECAY-ONLY INPUT (NO IRF)
    EXTRACTS f2 (LONG LIFETIME FRACTION)

    This is a modified version that:
    1. Does NOT load or process IRF data
    2. Extracts f2 (long lifetime/bound fraction) instead of f1

    Expected .mat file structure:
    - 'Hist': [H, W, time_bins] or [time_bins, H, W] - decay histograms
    - 'tau_gt_components': [H, W, 2] or [2, H, W] - lifetimes [tau1, tau2]
    - 'f_gt_components': [H, W, 2] or [2, H, W] - fractions [f1, f2]

    IMPORTANT: This loader extracts f2 (component 1) = LONG lifetime fraction
               NOT f1 (component 0) = short lifetime fraction

    Returns:
    --------
    train_set, test_set: DataLoaders with (decay, tau1, tau2, f2) - NO IRF
    """
    mat_files = glob.glob(os.path.join(data_root, '*.mat'))
    print(f"Found {len(mat_files)} .mat files in {data_root}")

    decays_list = []
    tau1_list = []
    tau2_list = []
    f2_list = []

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
                data = h5file

            # Extract histogram dimensions
            if 'Hist' in data:
                if use_h5py:
                    # h5py: [time_bins, H, W] format
                    time_bins, H, W = data['Hist'].shape

                    # Load parameters (small arrays)
                    tau_components = np.array(data['tau_gt_components'])  # [2, H, W]
                    f_components = np.array(data['f_gt_components'])  # [2, H, W]

                    # Sample decay curves (don't load all pixels)
                    # Sample every 4th pixel to reduce memory
                    sample_step = 4
                    decay_curves = []
                    tau1_vals = []
                    tau2_vals = []
                    f2_vals = []

                    for i in range(0, H, sample_step):
                        for j in range(0, W, sample_step):
                            decay_curve = data['Hist'][:, i, j]  # Load single pixel
                            decay_curves.append(decay_curve)
                            tau1_vals.append(tau_components[0, i, j])
                            tau2_vals.append(tau_components[1, i, j])
                            f2_vals.append(f_components[1, i, j])

                    decay_curves = np.array(decay_curves)  # [N, time_bins]
                    tau1_values = np.array(tau1_vals)
                    tau2_values = np.array(tau2_vals)
                    f2_values = np.array(f2_vals)

                    h5file.close()

                else:
                    # scipy.io format: [H, W, time_bins]
                    Hist = data['Hist']
                    H, W, time_bins = Hist.shape

                    tau_components = data['tau_gt_components']  # [H, W, 2]
                    f_components = data['f_gt_components']

                    # Extract decay curves from each pixel
                    decay_curves = Hist.reshape(H*W, time_bins)

                    # Extract parameter values
                    tau1_values = tau_components[:, :, 0].reshape(H*W)
                    tau2_values = tau_components[:, :, 1].reshape(H*W)
                    f2_values = f_components[:, :, 1].reshape(H*W)  # CHANGED: index 1 for f2 (LONG lifetime)
            else:
                print(f"Warning: {mat_file} missing Hist, skipping...")
                if use_h5py:
                    h5file.close()
                continue

            # Add to lists
            decays_list.append(decay_curves)
            tau1_list.append(tau1_values)
            tau2_list.append(tau2_values)
            f2_list.append(f2_values)

        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Concatenate all data
    all_decays = np.vstack(decays_list).astype(np.float32)
    all_tau1 = np.hstack(tau1_list).astype(np.float32)
    all_tau2 = np.hstack(tau2_list).astype(np.float32)
    all_f2 = np.hstack(f2_list).astype(np.float32)

    print(f"\nTotal samples: {all_decays.shape[0]}")
    print(f"Decay shape: {all_decays.shape}")
    print(f"Parameter ranges (TRAINING FOR LONG LIFETIME FRACTION f2, DECAY-ONLY INPUT):")
    print(f"  tau1 (short): [{all_tau1.min():.4f}, {all_tau1.max():.4f}] ns")
    print(f"  tau2 (long):  [{all_tau2.min():.4f}, {all_tau2.max():.4f}] ns")
    print(f"  f2 (LONG/BOUND fraction): [{all_f2.min():.4f}, {all_f2.max():.4f}]")
    print(f"  tau1 mean: {all_tau1.mean():.4f}, std: {all_tau1.std():.4f}")
    print(f"  tau2 mean: {all_tau2.mean():.4f}, std: {all_tau2.std():.4f}")
    print(f"  f2 mean: {all_f2.mean():.4f}, std: {all_f2.std():.4f}")

    # Normalize decays
    for i in range(all_decays.shape[0]):
        max_val = all_decays[i].max()
        if max_val > 0:
            all_decays[i] = all_decays[i] / max_val

    # Reshape for Conv1d: [N, 1, time_bins]
    all_decays = all_decays.reshape(-1, 1, time_bins)

    # Convert to tensors
    decay_tensor = torch.from_numpy(all_decays).float()
    tau1_tensor = torch.from_numpy(all_tau1).float()
    tau2_tensor = torch.from_numpy(all_tau2).float()
    f2_tensor = torch.from_numpy(all_f2).float()

    # Create dataset (NO IRF - only 4 elements)
    torch_dataset = TensorDataset(
        decay_tensor,
        tau1_tensor, tau2_tensor, f2_tensor
    )

    # Split train/test
    test_size = round(test_ratio * len(torch_dataset))
    train_size = len(torch_dataset) - test_size
    train, test = random_split(torch_dataset, [train_size, test_size])

    train_set = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_set = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_set, test_set


# =============================================================================
# TRAINING FUNCTIONS - DECAY-ONLY (NO IRF)
# =============================================================================
def train_stage1_reconstruction(train_set, test_set, model, learning_rate=1e-4,
                                 Epoch=50, USE_GPU=True, log_interval=100,
                                 patience=15, path_dir=''):
    """
    STAGE 1: Unsupervised Reconstruction Training (DECAY-ONLY)

    Goal: Learn to compress signal into latent space and reconstruct
    Loss: MSE(reconstruction, original_signal)

    This stage teaches the autoencoder to capture signal structure
    without any parameter supervision.
    """
    print("\n" + "=" * 80)
    print(" STAGE 1: RECONSTRUCTION TRAINING (Unsupervised, Decay-Only)")
    print("=" * 80)

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

    Train_Loss = []
    Val_Loss = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    start_time = time.time()

    for epoch in range(Epoch):
        print(f'\nEpoch {epoch+1}/{Epoch}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                use_set = train_set
            else:
                model.eval()
                use_set = test_set

            epoch_losses = []

            for batch_idx, (decays, tau1, tau2, f2) in enumerate(use_set):
                if USE_GPU:
                    decays = decays.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward: encode + decode (DECAY-ONLY)
                    reconstruction, latent = model(decays)

                    # Loss: reconstruction error only
                    loss = criterion(reconstruction, decays)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_losses.append(loss.cpu().detach().item())

                if batch_idx % log_interval == 0 and phase == 'train':
                    print(f'  Batch {batch_idx}/{len(use_set)} - Loss: {loss.item():.6f}')

            mean_loss = np.mean(epoch_losses)

            if phase == 'train':
                Train_Loss.append(mean_loss)
            else:
                Val_Loss.append(mean_loss)
                early_stopping(epoch+1, mean_loss, model, path_dir, stage='stage1')

            print(f'{phase.upper()} - Loss: {mean_loss:.6f}')

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step()

    training_time = time.time() - start_time
    print(f"\nStage 1 completed in {training_time:.2f}s")

    return {'Train_Loss': Train_Loss, 'Val_Loss': Val_Loss}


def train_stage2_parameters(train_set, test_set, model, learning_rate=5e-5,
                            Epoch=30, USE_GPU=True, log_interval=100,
                            patience=10, path_dir=''):
    """
    STAGE 2: Supervised Parameter Learning (DECAY-ONLY)

    Goal: Map latent space to physical parameters
    Loss: MSE(predicted_params, ground_truth_params)

    This stage structures the latent space to directly encode
    tau1, tau2, and f2 values.
    """
    print("\n" + "=" * 80)
    print(" STAGE 2: PARAMETER LEARNING (Supervised, Decay-Only)")
    print("=" * 80)

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    Train_Loss = []
    Val_Loss = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    start_time = time.time()

    for epoch in range(Epoch):
        print(f'\nEpoch {epoch+1}/{Epoch}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                use_set = train_set
            else:
                model.eval()
                use_set = test_set

            epoch_losses = []

            for batch_idx, (decays, tau1_gt, tau2_gt, f2_gt) in enumerate(use_set):
                if USE_GPU:
                    decays = decays.cuda()
                    tau1_gt, tau2_gt, f2_gt = tau1_gt.cuda(), tau2_gt.cuda(), f2_gt.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Extract parameters from latent space (DECAY-ONLY)
                    params = model.extract_parameters(decays)

                    # Loss: parameter prediction error
                    # Note: model outputs f1, but we're training with f2_gt
                    # So we compare with (1 - f2_gt) or train f1 directly
                    # Based on the data loader, f2_gt is the long fraction
                    # and params['f1'] is what the model predicts for the long fraction
                    loss_tau1 = criterion(params['tau1'], tau1_gt)
                    loss_tau2 = criterion(params['tau2'], tau2_gt)
                    loss_f = criterion(params['f1'], f2_gt)  # Model's f1 maps to data's f2

                    loss = loss_tau1 + loss_tau2 + loss_f

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_losses.append(loss.cpu().detach().item())

                if batch_idx % log_interval == 0 and phase == 'train':
                    print(f'  Batch {batch_idx}/{len(use_set)} - Loss: {loss.item():.6f}')

            mean_loss = np.mean(epoch_losses)

            if phase == 'train':
                Train_Loss.append(mean_loss)
            else:
                Val_Loss.append(mean_loss)
                early_stopping(epoch+1, mean_loss, model, path_dir, stage='stage2')

            print(f'{phase.upper()} - Loss: {mean_loss:.6f}')

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step()

    training_time = time.time() - start_time
    print(f"\nStage 2 completed in {training_time:.2f}s")

    return {'Train_Loss': Train_Loss, 'Val_Loss': Val_Loss}


def train_stage3_joint(train_set, test_set, model, learning_rate=1e-5,
                       Epoch=50, USE_GPU=True, log_interval=100,
                       patience=20, alpha=0.5, path_dir=''):
    """
    STAGE 3: Joint Optimization (DECAY-ONLY)

    Goal: Balance reconstruction quality and parameter accuracy
    Loss: alpha * reconstruction_loss + (1-alpha) * parameter_loss

    This final stage fine-tunes the model to optimize both objectives,
    following the paper's methodology for achieving Cramér-Rao bound.
    """
    print("\n" + "=" * 80)
    print(" STAGE 3: JOINT OPTIMIZATION (Decay-Only)")
    print(f" Alpha (reconstruction weight): {alpha}")
    print("=" * 80)

    criterion_recon = MSELoss()
    criterion_param = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)

    Train_Loss = []
    Val_Loss = []
    Train_Loss_Recon = []
    Train_Loss_Param = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    start_time = time.time()

    for epoch in range(Epoch):
        print(f'\nEpoch {epoch+1}/{Epoch}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                use_set = train_set
            else:
                model.eval()
                use_set = test_set

            epoch_losses = []
            epoch_losses_recon = []
            epoch_losses_param = []

            for batch_idx, (decays, tau1_gt, tau2_gt, f2_gt) in enumerate(use_set):
                if USE_GPU:
                    decays = decays.cuda()
                    tau1_gt, tau2_gt, f2_gt = tau1_gt.cuda(), tau2_gt.cuda(), f2_gt.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward with both outputs (DECAY-ONLY)
                    reconstruction, params = model.forward_with_parameters(decays)

                    # Reconstruction loss
                    loss_recon = criterion_recon(reconstruction, decays)

                    # Parameter loss
                    loss_tau1 = criterion_param(params['tau1'], tau1_gt)
                    loss_tau2 = criterion_param(params['tau2'], tau2_gt)
                    loss_f = criterion_param(params['f1'], f2_gt)  # Model's f1 maps to data's f2
                    loss_param = loss_tau1 + loss_tau2 + loss_f

                    # Combined loss
                    loss = alpha * loss_recon + (1 - alpha) * loss_param

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_losses.append(loss.cpu().detach().item())
                epoch_losses_recon.append(loss_recon.cpu().detach().item())
                epoch_losses_param.append(loss_param.cpu().detach().item())

                if batch_idx % log_interval == 0 and phase == 'train':
                    print(f'  Batch {batch_idx}/{len(use_set)} - '
                          f'Total: {loss.item():.6f}, '
                          f'Recon: {loss_recon.item():.6f}, '
                          f'Param: {loss_param.item():.6f}')

            mean_loss = np.mean(epoch_losses)
            mean_loss_recon = np.mean(epoch_losses_recon)
            mean_loss_param = np.mean(epoch_losses_param)

            if phase == 'train':
                Train_Loss.append(mean_loss)
                Train_Loss_Recon.append(mean_loss_recon)
                Train_Loss_Param.append(mean_loss_param)
            else:
                Val_Loss.append(mean_loss)
                early_stopping(epoch+1, mean_loss, model, path_dir, stage='stage3')

            print(f'{phase.upper()} - Total: {mean_loss:.6f}, '
                  f'Recon: {mean_loss_recon:.6f}, Param: {mean_loss_param:.6f}')

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step()

    training_time = time.time() - start_time
    print(f"\nStage 3 completed in {training_time:.2f}s")

    return {
        'Train_Loss': Train_Loss,
        'Val_Loss': Val_Loss,
        'Train_Loss_Recon': Train_Loss_Recon,
        'Train_Loss_Param': Train_Loss_Param
    }


# =============================================================================
# MAIN TRAINING SCRIPT - DECAY PEAK BIN 62 (DECAY-ONLY, NO IRF)
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" THREE-STAGE AUTOENCODER TRAINING - DECAY PEAK BIN 62")
    print(" DECAY-ONLY INPUT (NO IRF)")
    print(" TRAINING FOR LONG LIFETIME FRACTION (f2) - BOUND FRACTION")
    print(" Following Visschers et al. 2021 methodology")
    print(" TEMPORAL ALIGNMENT TO REAL DATA")
    print(" Dataset : decay_peak_bin62_12.5ns_trainfixed (100 files, 256x256, 256 bins)")
    print(" IRF     : Centered at bin 62 (t = 3.027 ns) - MATCHED TO REAL DATA")
    print(" Tau1    : Uniform[0.20, 0.70] ns (SHORT lifetime, NO spatial correlation)")
    print(" Tau2    : Uniform[1.20, 4.50] ns (LONG lifetime, NO spatial correlation)")
    print(" TARGET  : f2 (LONG/BOUND fraction) = Uniform[0.1, 0.9]")
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
    # DATA PATH - E DRIVE (12.5ns TEMPORAL CALIBRATION) - TRAINFIXED
    # =========================================================================
    data_root = r'E:\decay_peak_bin62_12.5ns_trainfixed'

    print(f"Data directory : {data_root}")
    print(f"Directory exists: {os.path.exists(data_root)}")

    if not os.path.exists(data_root):
        print(f"\nERROR: Data directory not found!")
        print(f"Please ensure the dataset exists at: {data_root}")
        print(f"Run the updated Generate_MultiExp_DecayStart_Bin50.m (12.5ns version) first.")
        sys.exit(1)

    # =========================================================================
    # LOAD DATA - DECAY-ONLY (NO IRF), USING CUSTOM LOADER FOR f2
    # =========================================================================
    print("\nLoading decay_peak_bin62_12.5ns_trainfixed FLIM data...")
    print("IMPORTANT: Using DECAY-ONLY input (no IRF)")
    print("IMPORTANT: Using f2 (LONG LIFETIME/BOUND FRACTION) as training target\n")
    # Use smaller test ratio since we only have 100 images
    train_set, test_set = load_biexp_data_f2_from_directory_NoIRF(
        data_root, test_ratio=0.15, BATCH_SIZE=128
    )

    # =========================================================================
    # CHECKPOINT DIRECTORY
    # =========================================================================
    now       = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    ckpt_dir  = f'./training_logs_decay_bin62_12.5ns_f2_longfrac_NoIRF_{timestamp}'
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint directory: {ckpt_dir}")

    # Save dataset info
    dataset_info_path = os.path.join(ckpt_dir, 'dataset_info.txt')
    with open(dataset_info_path, 'w') as f:
        f.write("DECAY PEAK BIN 62 DATASET INFO - 12.5ns CALIBRATION - DECAY-ONLY (NO IRF)\n")
        f.write("TRAINING FOR LONG LIFETIME FRACTION (f2) - BOUND FRACTION\n")
        f.write("=" * 80 + "\n\n")
        f.write("TEMPORAL ALIGNMENT TO REAL DATA (CORRECTED)\n\n")
        f.write("Dataset: decay_peak_bin62_12.5ns_trainfixed\n")
        f.write("Total images: 100 (256x256 pixels, 256 time-bins)\n")
        f.write("Input: DECAY-ONLY (no IRF)\n\n")
        f.write("Temporal Calibration:\n")
        f.write("  IRF center: Bin 62 (t = 3.027 ns)\n")
        f.write("  Bin width: 0.048828 ns (48.828 ps)\n")
        f.write("  Total time: 12.5 ns\n")
        f.write("  Real data bin width: 0.048828 ns (EXACT MATCH)\n")
        f.write("  Previous training bin width: 0.039 ns (25% mismatch - FIXED)\n\n")
        f.write("Parameter Distributions (all UNIFORM, NO spatial correlation):\n")
        f.write("  Tau1 (SHORT): Uniform[0.20, 0.70] ns\n")
        f.write("  Tau2 (LONG):  Uniform[1.20, 4.50] ns\n")
        f.write("  TRAINING TARGET: f2 (LONG/BOUND fraction) = Uniform[0.1, 0.9]\n\n")
        f.write("IMPORTANT - Training Target:\n")
        f.write("  - This model trains for f2 (component 1) = LONG lifetime fraction\n")
        f.write("  - f2 represents the bound fraction (tau2 component)\n")
        f.write("  - f1 (component 0) = short lifetime fraction\n")
        f.write("  - f1 + f2 = 1.0\n")
        f.write("  - Input: DECAY-ONLY (no IRF)\n\n")
        f.write("Key Features:\n")
        f.write("  - Each pixel completely independent\n")
        f.write("  - NO spatial correlation for any parameter\n")
        f.write("  - All histograms approximately uniform\n")
        f.write("  - DECAY-ONLY input (no IRF)\n\n")
        f.write("Experiment Purpose:\n")
        f.write("  - Train model with decay peak at bin 62 (matches real data)\n")
        f.write("  - Train specifically for LONG lifetime fraction (bound fraction)\n")
        f.write("  - Enable direct inference on real FLIM samples\n")
        f.write("  - No temporal misalignment issues with real data\n")
        f.write("  - Model output represents f2 (bound fraction), not f1\n")
        f.write("  - Test if IRF is necessary for accurate parameter extraction\n")

    # =========================================================================
    # MODEL
    # =========================================================================
    print("\nInitializing autoencoder model (DECAY-ONLY)...")
    model = LLE_BiExp_Autoencoder(
        dim=dim,
        depth=depth,
        kernel_size=ks,
        patch_size=ps,
        signal_length=256,
        latent_dim=3
    )

    print("\nModel architecture:")
    summary(model, (1, 256), device='cpu')  # Single input (decay only)

    # Save model architecture
    arch_path = os.path.join(ckpt_dir, 'model_architecture.txt')
    with open(arch_path, 'w') as f:
        f.write("LLE Bi-Exponential Autoencoder Architecture - DECAY-ONLY (NO IRF)\n")
        f.write("TRAINING FOR LONG LIFETIME FRACTION (f2)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  dim (channel dimension): {dim}\n")
        f.write(f"  depth (number of blocks): {depth}\n")
        f.write(f"  kernel_size: {ks}\n")
        f.write(f"  patch_size: {ps}\n")
        f.write(f"  signal_length: 256\n")
        f.write(f"  latent_dim: 3 (tau1, tau2, f2-LONG/BOUND fraction)\n")
        f.write(f"  input: DECAY-ONLY (no IRF)\n\n")
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
        f.write(f"Input: DECAY-ONLY (no IRF)\n")

    # =========================================================================
    # THREE-STAGE TRAINING
    # =========================================================================
    all_records = {}

    # ------------------------------------------------------------------
    # STAGE 1: Reconstruction
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 1: Learning to reconstruct decay signals (bin 62 timing, DECAY-ONLY)")
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
        'dataset': 'decay_peak_bin62_12.5ns_trainfixed',
        'irf_bin': 62,
        'bin_width_ns': 0.048828,
        'training_target': 'f2_long_lifetime_fraction',
        'input_type': 'decay_only_no_irf',
        'note': 'Training for LONG lifetime (bound) fraction, DECAY-ONLY input (no IRF)'
    }, os.path.join(ckpt_dir, 'checkpoint_stage1.pth'))

    # ------------------------------------------------------------------
    # STAGE 2: Parameter Learning
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 2: Learning tau1, tau2, f2 (LONG/BOUND fraction) from bin 62 ground truth (DECAY-ONLY)")
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
        'dataset': 'decay_peak_bin62_12.5ns_trainfixed',
        'irf_bin': 62,
        'bin_width_ns': 0.048828,
        'training_target': 'f2_long_lifetime_fraction',
        'input_type': 'decay_only_no_irf',
        'note': 'Training for LONG lifetime (bound) fraction, DECAY-ONLY input (no IRF)'
    }, os.path.join(ckpt_dir, 'checkpoint_stage2.pth'))

    # ------------------------------------------------------------------
    # STAGE 3: Joint Optimization
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STAGE 3: Joint optimisation (reconstruction + parameters, DECAY-ONLY)")
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
        'dataset': 'decay_peak_bin62_12.5ns_trainfixed',
        'irf_bin': 62,
        'training_target': 'f2_long_lifetime_fraction',
        'input_type': 'decay_only_no_irf',
        'hyperparameters': {
            'dim': dim,
            'depth': depth,
            'kernel_size': ks,
            'patch_size': ps,
            'signal_length': 256,
            'latent_dim': 3
        },
        'data_info': {
            'irf_center_bin': 62,
            'irf_center_time_ns': 62 * 0.048828,  # 3.027 ns
            'bin_width_ns': 0.048828,  # 48.828 ps - EXACT MATCH TO REAL DATA
            'total_time_ns': 12.5,  # 256 bins * 0.048828 ns/bin
            'tau1_range': [0.20, 0.70],
            'tau2_range': [1.20, 4.50],
            'f2_range': [0.1, 0.9],
            'fraction_target': 'f2_long_lifetime',  # IMPORTANT: model predicts f2 (bound fraction)
            'photon_range': [50, 1000],  # Flat random photons
            'distribution': 'uniform',
            'spatial_correlation': False,
            'description': 'Decay peak bin 62 - 12.5ns temporal calibration - Training for LONG lifetime (bound) fraction (f2) - DECAY-ONLY input (no IRF)'
        },
        'important_note': 'MODEL OUTPUTS f2 (LONG LIFETIME/BOUND FRACTION), NOT f1. f2 = fraction of tau2 component. INPUT: DECAY-ONLY (no IRF)',
        'info': 'Three-stage trained autoencoder on decay_peak_bin62_12.5ns_trainfixed - f2 (long lifetime) target - DECAY-ONLY (Visschers et al. 2021)'
    }, os.path.join(ckpt_dir, 'model_final_decay_bin62_12.5ns_f2_longfrac_NoIRF.pth'))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(" TRAINING COMPLETED - DECAY PEAK BIN 62 (12.5ns CALIBRATION, DECAY-ONLY)")
    print(" TRAINING TARGET: f2 (LONG LIFETIME/BOUND FRACTION)")
    print(" INPUT: DECAY-ONLY (NO IRF)")
    print(f"{'=' * 80}")
    print(f"  Total time   : {total_time:.2f} s  ({total_time/60:.1f} min)")
    print(f"  Checkpoints  : {ckpt_dir}/")
    print(f"  Final model  : {ckpt_dir}/model_final_decay_bin62_12.5ns_f2_longfrac_NoIRF.pth")
    print(f"  Dataset info : {ckpt_dir}/dataset_info.txt")
    print(f"  Architecture : {ckpt_dir}/model_architecture.txt")
    print(f"  Bin width    : 0.048828 ns (EXACT match to real data)")
    print(f"  IMPORTANT    : Model predicts f2 (LONG/BOUND fraction), NOT f1")
    print(f"  IMPORTANT    : DECAY-ONLY input (no IRF)")
    print("=" * 80 + "\n")

    # Save final summary
    summary_path = os.path.join(ckpt_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TRAINING SUMMARY - DECAY PEAK BIN 62 (12.5ns CALIBRATION, DECAY-ONLY)\n")
        f.write("TRAINING TARGET: f2 (LONG LIFETIME/BOUND FRACTION)\n")
        f.write("INPUT: DECAY-ONLY (NO IRF)\n")
        f.write("=" * 80 + "\n\n")
        f.write("TEMPORAL ALIGNMENT TO REAL DATA (CORRECTED)\n\n")
        f.write(f"Dataset: decay_peak_bin62_12.5ns_trainfixed\n")
        f.write(f"IRF center: Bin 62 (t = {62 * 0.048828:.3f} ns)\n")
        f.write(f"Bin width: 0.048828 ns (48.828 ps) - EXACT MATCH to real data\n")
        f.write(f"Total training time: {total_time:.2f} s ({total_time/60:.1f} min)\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("IMPORTANT - Training Target:\n")
        f.write("  THIS MODEL TRAINS FOR f2 (LONG LIFETIME/BOUND FRACTION)\n")
        f.write("  - f1 = short lifetime (tau1) fraction\n")
        f.write("  - f2 = long lifetime (tau2) fraction = BOUND FRACTION\n")
        f.write("  - Model output represents f2, NOT f1\n")
        f.write("  - f1 + f2 = 1.0\n")
        f.write("  - INPUT: DECAY-ONLY (no IRF)\n\n")
        f.write("Model hyperparameters:\n")
        f.write(f"  dim: {dim}\n")
        f.write(f"  depth: {depth}\n")
        f.write(f"  kernel_size: {ks}\n")
        f.write(f"  patch_size: {ps}\n")
        f.write(f"  input: DECAY-ONLY (no IRF)\n\n")
        f.write("Training stages:\n")
        f.write(f"  Stage 1 (Reconstruction): {len(record_s1['Train_Loss'])} epochs\n")
        f.write(f"  Stage 2 (Parameters): {len(record_s2['Train_Loss'])} epochs\n")
        f.write(f"  Stage 3 (Joint): {len(record_s3['Train_Loss'])} epochs\n\n")
        f.write("Data characteristics:\n")
        f.write("  IRF center: Bin 62 (MATCHED TO REAL DATA)\n")
        f.write("  Real data peak: Bin 62\n")
        f.write("  Tau1 (SHORT): Uniform[0.20, 0.70] ns\n")
        f.write("  Tau2 (LONG):  Uniform[1.20, 4.50] ns\n")
        f.write("  Fraction f2 (LONG/BOUND): Uniform[0.1, 0.9]\n")
        f.write("  Spatial correlation: None\n")
        f.write("  All parameters independent per-pixel\n")
        f.write("  Input: DECAY-ONLY (no IRF)\n\n")
        f.write("Next steps:\n")
        f.write("  - Run inference on real FLIM data (Control, F+-, F++, Rot groups)\n")
        f.write("  - Model will output f2 (bound fraction), not f1\n")
        f.write("  - No temporal alignment issues (training and real data both peak at bin 62)\n")
        f.write("  - Compare with model trained WITH IRF input\n")
        f.write("  - Analyze if IRF is necessary for accurate parameter extraction\n")
        f.write("  - Analyze group-based differences in bound fraction (f2)\n")

    print(f"Training summary saved to: {summary_path}\n")

    # Clean up GPU memory
    if USE_GPU:
        del model
        torch.cuda.empty_cache()
