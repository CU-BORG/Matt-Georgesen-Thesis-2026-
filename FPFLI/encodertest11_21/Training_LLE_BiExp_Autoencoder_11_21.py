#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Stage Training for LLE Bi-Exponential Autoencoder

Based on Visschers et al. 2021 methodology:
  Stage 1: Unsupervised reconstruction - learn to encode/decode signal
  Stage 2: Supervised parameter learning - map latent space to physical parameters
  Stage 3: Joint optimization - balance reconstruction + parameter accuracy

This training approach ensures the latent space directly encodes physical
parameters (tau1, tau2, f1) while maintaining signal reconstruction capability.

@author:mg
"""

import time
import os
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
    Load bi-exponential FLIM data from directory

    Expected .mat file structure:
    - 'Hist': [H, W, time_bins] - decay histograms
    - 'tau_gt_components' or 'tau_components': [H, W, 2] - lifetime components
    - 'f_gt_components' or 'f_components': [H, W, 2] - fraction components
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
        if idx % 100 == 0:
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
    print(f"Parameter ranges:")
    print(f"  tau1: [{all_tau1.min():.4f}, {all_tau1.max():.4f}] ns")
    print(f"  tau2: [{all_tau2.min():.4f}, {all_tau2.max():.4f}] ns")
    print(f"  f1: [{all_f1.min():.4f}, {all_f1.max():.4f}]")

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


##
'''# Concatenate all data
    all_decays = np.vstack(decays_list).astype(np.float32)
    all_irfs = np.vstack(irfs_list).astype(np.float32)
    all_tau1 = np.hstack(tau1_list).astype(np.float32)
    all_tau2 = np.hstack(tau2_list).astype(np.float32)
    all_f1 = np.hstack(f1_list).astype(np.float32)

    print(f"\nTotal samples: {all_decays.shape[0]}")
    print(f"Decay shape: {all_decays.shape}")
    print(f"Parameter ranges:")
    print(f"  tau1: [{all_tau1.min():.4f}, {all_tau1.max():.4f}] ns")
    print(f"  tau2: [{all_tau2.min():.4f}, {all_tau2.max():.4f}] ns")
    print(f"  f1: [{all_f1.min():.4f}, {all_f1.max():.4f}]")

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
'''

def train_stage1_reconstruction(train_set, test_set, model, learning_rate=1e-4,
                                 Epoch=50, USE_GPU=True, log_interval=100,
                                 patience=15, path_dir=''):
    """
    STAGE 1: Unsupervised Reconstruction Training

    Goal: Learn to compress signal into latent space and reconstruct
    Loss: MSE(reconstruction, original_signal)

    This stage teaches the autoencoder to capture signal structure
    without any parameter supervision.
    """
    print("\n" + "=" * 80)
    print(" STAGE 1: RECONSTRUCTION TRAINING (Unsupervised)")
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

            for batch_idx, (decays, irfs, tau1, tau2, f1) in enumerate(use_set):
                if USE_GPU:
                    decays, irfs = decays.cuda(), irfs.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward: encode + decode
                    reconstruction, latent = model(decays, irfs)

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
    STAGE 2: Supervised Parameter Learning

    Goal: Map latent space to physical parameters
    Loss: MSE(predicted_params, ground_truth_params)

    This stage structures the latent space to directly encode
    tau1, tau2, and f1 values.
    """
    print("\n" + "=" * 80)
    print(" STAGE 2: PARAMETER LEARNING (Supervised)")
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

            for batch_idx, (decays, irfs, tau1_gt, tau2_gt, f1_gt) in enumerate(use_set):
                if USE_GPU:
                    decays, irfs = decays.cuda(), irfs.cuda()
                    tau1_gt, tau2_gt, f1_gt = tau1_gt.cuda(), tau2_gt.cuda(), f1_gt.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Extract parameters from latent space
                    params = model.extract_parameters(decays, irfs)

                    # Loss: parameter prediction error
                    loss_tau1 = criterion(params['tau1'], tau1_gt)
                    loss_tau2 = criterion(params['tau2'], tau2_gt)
                    loss_f1 = criterion(params['f1'], f1_gt)

                    loss = loss_tau1 + loss_tau2 + loss_f1

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
    STAGE 3: Joint Optimization

    Goal: Balance reconstruction quality and parameter accuracy
    Loss: alpha * reconstruction_loss + (1-alpha) * parameter_loss

    This final stage fine-tunes the model to optimize both objectives,
    following the paper's methodology for achieving Cramér-Rao bound.
    """
    print("\n" + "=" * 80)
    print(" STAGE 3: JOINT OPTIMIZATION")
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

            for batch_idx, (decays, irfs, tau1_gt, tau2_gt, f1_gt) in enumerate(use_set):
                if USE_GPU:
                    decays, irfs = decays.cuda(), irfs.cuda()
                    tau1_gt, tau2_gt, f1_gt = tau1_gt.cuda(), tau2_gt.cuda(), f1_gt.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward with both outputs
                    reconstruction, params = model.forward_with_parameters(decays, irfs)

                    # Reconstruction loss
                    loss_recon = criterion_recon(reconstruction, decays)

                    # Parameter loss
                    loss_tau1 = criterion_param(params['tau1'], tau1_gt)
                    loss_tau2 = criterion_param(params['tau2'], tau2_gt)
                    loss_f1 = criterion_param(params['f1'], f1_gt)
                    loss_param = loss_tau1 + loss_tau2 + loss_f1

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
# MAIN TRAINING SCRIPT
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" THREE-STAGE AUTOENCODER TRAINING")
    print(" Following Visschers et al. 2021 methodology")
    print("=" * 80 + "\n")

    # Model hyperparameters
    dim = 16
    depth = 8
    ks = 9
    ps = 8

    USE_GPU = True
    Start = time.time()

    # Data paths - using the 500 sample multiexp dataset
    data_root = r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite\realdata_stats'

    print(f"Data directory: {data_root}")
    print(f"Directory exists: {os.path.exists(data_root)}")

    if not os.path.exists(data_root):
        print(f"\nERROR: Data directory not found!")
        print(f"Please ensure the dataset exists at: {data_root}")
        sys.exit(1)

    # Load data
    print("\nLoading bi-exponential FLIM data...")
    train_set, test_set = load_biexp_data_from_directory(
        data_root, test_ratio=0.2, BATCH_SIZE=128
    )

    # Create checkpoint directory
    now = datetime.datetime.now()
    ckpt_dir = f'./training_logs_LLE_BiExp_Autoencoder_{dim}_{depth}_{ks}_{ps}'
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint directory: {ckpt_dir}")

    # Create model
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

    if USE_GPU:
        model.cuda()

    paramter_initialize(model)

    # =========================================================================
    # THREE-STAGE TRAINING
    # =========================================================================

    all_records = {}

    # STAGE 1: Reconstruction
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
        'stage': 'stage1_reconstruction'
    }, os.path.join(ckpt_dir, 'checkpoint_stage1.pth'))

    # STAGE 2: Parameter Learning
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
        'stage': 'stage2_parameters'
    }, os.path.join(ckpt_dir, 'checkpoint_stage2.pth'))

    # STAGE 3: Joint Optimization
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

    # Save final model
    torch.save({
        'model': model.state_dict(),
        'all_records': all_records,
        'stage': 'stage3_final',
        'hyperparameters': {
            'dim': dim,
            'depth': depth,
            'kernel_size': ks,
            'patch_size': ps,
            'signal_length': 256,
            'latent_dim': 3
        },
        'info': 'Three-stage trained autoencoder following Visschers et al. 2021'
    }, os.path.join(ckpt_dir, 'model_final.pth'))

    Total_Time = time.time() - Start
    print(f"\n{'=' * 80}")
    print(f" TRAINING COMPLETED")
    print(f" Total time: {Total_Time:.2f}s ({Total_Time/60:.2f} min)")
    print(f"{'=' * 80}\n")
    print(f"Model saved to: {ckpt_dir}/model_final.pth")
    print("\nThe trained autoencoder can now:")
    print("  1. Extract parameters: tau1, tau2, f1 from decay curves")
    print("  2. Reconstruct decay signals with high fidelity")
    print("  3. Achieve parameter accuracy approaching Cramér-Rao bound")
    print()
