#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Exponential Lifetime Prediction on Sample Images
Evaluates the trained multiexp LLE model on a single sample image

@author: Evaluation Script
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Add network training path to import tau_net_multiexp
sys.path.append(r'C:\Users\mcg11923\Thesis\FPFLI\Network training\LLE_training')
from tau_net_multiexp import tauNet_MultiExp


def load_multiexp_model(model_path, n_components=2, device='cuda'):
    """Load trained multi-exponential model"""
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create model with same architecture
    model = tauNet_MultiExp(
        dim=16,
        depth=8,
        kernel_size=9,
        patch_size=8,
        n_components=n_components
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Number of components: {n_components}")
    print(f"  Best test loss: {checkpoint.get('test_loss', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def load_sample_image(image_path):
    """Load a multi-exponential sample image"""
    print(f"\nLoading sample image: {Path(image_path).name}")

    with h5py.File(image_path, 'r') as f:
        Hist = np.array(f['Hist']).T  # (H, W, bins)
        tau_gt = np.array(f['tau_gt_components']).T  # (H, W, n_components)
        f_gt = np.array(f['f_gt_components']).T  # (H, W, n_components)

        if 'Int' in f:
            Int = np.array(f['Int']).T
        else:
            Int = np.sum(Hist, axis=2)

        if 'IRF' in f:
            irf = np.array(f['IRF']).flatten()
        else:
            # Generate default IRF
            bins = Hist.shape[2]
            t = np.arange(bins)
            irf = np.exp(-(t - 14)**2 / (2 * 3**2))
            irf = irf / irf.max()

    print(f"  Image shape: {Hist.shape}")
    print(f"  Number of components: {tau_gt.shape[2]}")
    print(f"  Intensity range: [{Int.min():.1f}, {Int.max():.1f}]")
    print(f"  Tau GT range: [{tau_gt.min():.3f}, {tau_gt.max():.3f}]")

    return Hist, tau_gt, f_gt, Int, irf


def predict_image(model, Hist, irf, device='cuda', min_intensity=10):
    """Predict lifetimes for entire image"""
    print("\nPerforming prediction...")

    H, W, bins = Hist.shape
    n_components = model.n_components

    # Initialize output arrays
    tau_pred = np.zeros((H, W, n_components))
    f_pred = np.zeros((H, W, n_components))

    # Prepare IRF
    irf_tensor = torch.from_numpy(irf.astype('float32')).to(device)

    # Process pixels
    valid_count = 0
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                decay = Hist[i, j, :]
                intensity = decay.sum()

                if intensity > min_intensity:
                    # Normalize decay
                    decay_norm = decay / (decay.max() + 1e-8)

                    # Convert to tensor
                    decay_tensor = torch.from_numpy(decay_norm.astype('float32')).unsqueeze(0).unsqueeze(0).to(device)
                    irf_input = irf_tensor.unsqueeze(0).unsqueeze(0)

                    # Predict
                    tau_out, f_out = model(decay_tensor, irf_input)

                    tau_pred[i, j, :] = tau_out.cpu().numpy()
                    f_pred[i, j, :] = f_out.cpu().numpy()
                    valid_count += 1

    print(f"  Predicted {valid_count} pixels (out of {H*W})")
    print(f"  Tau prediction range: [{tau_pred[tau_pred > 0].min():.3f}, {tau_pred.max():.3f}]")

    return tau_pred, f_pred


def compute_average_lifetime(tau, f):
    """Compute average lifetime from components"""
    return np.sum(tau * f, axis=2)


def visualize_results(tau_gt, f_gt, tau_pred, f_pred, save_path=None):
    """Visualize ground truth vs predicted lifetimes"""
    print("\nVisualizing results...")

    # Compute average lifetimes
    tau_avg_gt = compute_average_lifetime(tau_gt, f_gt)
    tau_avg_pred = compute_average_lifetime(tau_pred, f_pred)

    n_components = tau_gt.shape[2]

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # Row 1: Lifetime components (GT)
    for i in range(n_components):
        ax = plt.subplot(4, n_components, i + 1)
        im = ax.imshow(tau_gt[:, :, i], cmap='viridis')
        ax.set_title(f'GT τ{i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Lifetime components (Predicted)
    for i in range(n_components):
        ax = plt.subplot(4, n_components, n_components + i + 1)
        im = ax.imshow(tau_pred[:, :, i], cmap='viridis')
        ax.set_title(f'Pred τ{i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: Fraction components (GT)
    for i in range(n_components):
        ax = plt.subplot(4, n_components, 2*n_components + i + 1)
        im = ax.imshow(f_gt[:, :, i], cmap='plasma', vmin=0, vmax=1)
        ax.set_title(f'GT f{i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 4: Fraction components (Predicted)
    for i in range(n_components):
        ax = plt.subplot(4, n_components, 3*n_components + i + 1)
        im = ax.imshow(f_pred[:, :, i], cmap='plasma', vmin=0, vmax=1)
        ax.set_title(f'Pred f{i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_components.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved component visualization to: {save_path.replace('.png', '_components.png')}")

    # Second figure: Average lifetimes and errors
    fig2 = plt.figure(figsize=(18, 6))

    # Ground truth average
    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(tau_avg_gt, cmap='viridis')
    ax1.set_title('Ground Truth τ_avg', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Predicted average
    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(tau_avg_pred, cmap='viridis', vmin=tau_avg_gt.min(), vmax=tau_avg_gt.max())
    ax2.set_title('Predicted τ_avg', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Error map
    error = np.abs(tau_avg_gt - tau_avg_pred)
    ax3 = plt.subplot(1, 4, 3)
    im3 = ax3.imshow(error, cmap='hot')
    ax3.set_title('Absolute Error', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Correlation plot
    ax4 = plt.subplot(1, 4, 4)
    valid_mask = tau_avg_gt > 0
    ax4.scatter(tau_avg_gt[valid_mask].flatten(), tau_avg_pred[valid_mask].flatten(),
                alpha=0.3, s=1)
    ax4.plot([tau_avg_gt.min(), tau_avg_gt.max()],
             [tau_avg_gt.min(), tau_avg_gt.max()], 'r--', linewidth=2, label='Perfect fit')
    ax4.set_xlabel('Ground Truth τ_avg', fontsize=12)
    ax4.set_ylabel('Predicted τ_avg', fontsize=12)
    ax4.set_title('Correlation', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved average lifetime visualization to: {save_path}")

    plt.show()

    # Compute and print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    valid_mask = tau_avg_gt > 0
    mae = np.mean(np.abs(tau_avg_gt[valid_mask] - tau_avg_pred[valid_mask]))
    rmse = np.sqrt(np.mean((tau_avg_gt[valid_mask] - tau_avg_pred[valid_mask])**2))
    mape = np.mean(np.abs((tau_avg_gt[valid_mask] - tau_avg_pred[valid_mask]) / tau_avg_gt[valid_mask])) * 100

    correlation = np.corrcoef(tau_avg_gt[valid_mask].flatten(), tau_avg_pred[valid_mask].flatten())[0, 1]

    print(f"\nAverage Lifetime (τ_avg) Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Correlation: {correlation:.4f}")

    # Component-wise metrics
    for i in range(n_components):
        valid_comp = (tau_gt[:, :, i] > 0) & (tau_pred[:, :, i] > 0)
        if np.sum(valid_comp) > 0:
            mae_comp = np.mean(np.abs(tau_gt[:, :, i][valid_comp] - tau_pred[:, :, i][valid_comp]))
            print(f"\nComponent {i+1} (τ{i+1}) MAE: {mae_comp:.4f}")

            mae_frac = np.mean(np.abs(f_gt[:, :, i][valid_comp] - f_pred[:, :, i][valid_comp]))
            print(f"Component {i+1} (f{i+1}) MAE: {mae_frac:.4f}")

    print("="*60)


def main():
    """Main evaluation function"""

    # Configuration
    model_path = r'C:\Users\mcg11923\Thesis\training_logs_multiexp_2comp_20251010_194609\best_model.pth'
    sample_image = r'C:\Users\mcg11923\Thesis\training_dataset_multiexp_s8\Sample_1000_MultiExp_C1C4.mat'
    n_components = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Multi-Exponential Lifetime Prediction Evaluation")
    print("="*60)
    print(f"Device: {device}")

    # Load model
    model = load_multiexp_model(model_path, n_components=n_components, device=device)

    # Load sample image
    Hist, tau_gt, f_gt, Int, irf = load_sample_image(sample_image)

    # Predict
    tau_pred, f_pred = predict_image(model, Hist, irf, device=device, min_intensity=10)

    # Visualize
    output_path = r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation\multiexp_prediction_result.png'
    visualize_results(tau_gt, f_gt, tau_pred, f_pred, save_path=output_path)

    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
