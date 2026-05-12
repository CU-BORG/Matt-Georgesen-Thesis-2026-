#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-Exponential FLIM Pipeline Evaluation Script

This script evaluates the original single-exponential models:
1. LLE: Local lifetime estimation (single tau at low resolution)
2. NIII: Neural implicit interpolation (upsamples to high resolution)

Usage:
    python evaluate_single_exp_pipeline.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

# Add paths for model imports
sys.path.append(r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation')

from Pred_local_lifetime import LLE
from Neural_implict_interploation import NIII_process
from NIII_model import make_coord


def downSizeImg_2D(Img_HR, k):
    """Downsample 2D image by factor k"""
    Img_HR = Img_HR.unsqueeze(0)
    f = torch.ones([1, 1, k, k]).type(torch.float)
    Img_LR = F.conv2d(Img_HR, f, stride=[k, k])
    return Img_LR.squeeze()


class SingleExpPipeline:
    """
    Single-exponential FLIM processing pipeline
    """

    def __init__(
        self,
        lle_model_path,
        niii_model_path,
        scaling_ratio=8,
        bin_width=0.039,
        intensity_threshold=10,
        device='cuda'
    ):
        """
        Initialize the single-exponential pipeline

        Parameters:
        -----------
        lle_model_path : str
            Path to trained LLE model (.pth file)
        niii_model_path : str
            Path to trained NIII model (.pth.tar file)
        scaling_ratio : int
            Downsampling factor (default: 8)
        bin_width : float
            Time bin width in nanoseconds (default: 0.039)
        intensity_threshold : float
            Minimum intensity threshold for processing pixels
        device : str
            'cuda' or 'cpu'
        """
        self.scaling_ratio = scaling_ratio
        self.bin_width = bin_width
        self.multiplier = bin_width * 100
        self.intensity_threshold = intensity_threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lle_model_path = lle_model_path
        self.niii_model_path = niii_model_path

        print(f"Initializing Single-Exponential Pipeline on {self.device}...")
        print(f"  LLE model: {lle_model_path}")
        print(f"  NIII model: {niii_model_path}")

    def stage1_lle(self, hist2d, irf):
        """
        Stage 1: Local Lifetime Estimation (Single-exponential)

        Parameters:
        -----------
        hist2d : ndarray
            Histogram data [H, W, bins]
        irf : ndarray
            Instrument response function

        Returns:
        --------
        dict containing:
            - tau_lr: Low-res lifetime [H/r, W/r]
        """
        print("\n" + "="*60)
        print("Stage 1: LLE (Local Lifetime Estimation - Single Exponential)")
        print("="*60)

        # Use LLE function from Pred_local_lifetime
        lr_tau = LLE(
            hist2d,
            irf,
            r=self.scaling_ratio,
            bin_w=self.bin_width,
            p_path=self.lle_model_path
        )

        H_lr, W_lr = lr_tau.shape
        print(f"  Low-res size: {H_lr} x {W_lr}")

        # Count valid pixels
        valid_pixels = (lr_tau > 0).sum()
        total_pixels = lr_tau.size
        print(f"  Processed {valid_pixels}/{total_pixels} valid pixels")

        # Print statistics
        if valid_pixels > 0:
            valid_taus = lr_tau[lr_tau > 0]
            print(f"  tau range: [{valid_taus.min():.3f}, {valid_taus.max():.3f}] ns")

        return {'tau_lr': lr_tau}

    def stage2_niii(self, hr_intensity, lr_results):
        """
        Stage 2: Neural Implicit Interpolation

        Parameters:
        -----------
        hr_intensity : ndarray
            High-resolution intensity image [H, W]
        lr_results : dict
            Results from stage 1 containing tau_lr

        Returns:
        --------
        dict containing:
            - tau: High-res lifetime [H, W]
            - mask: Binary mask showing valid pixels [H, W]
        """
        print("\n" + "="*60)
        print("Stage 2: NIII (Neural Implicit Interpolation)")
        print("="*60)

        H, W = hr_intensity.shape
        tau_lr = lr_results['tau_lr']

        print(f"  High-res size: {H} x {W}")
        print(f"  Low-res size: {tau_lr.shape[0]} x {tau_lr.shape[1]}")

        # Create signal mask based on intensity threshold
        signal_mask = hr_intensity > self.intensity_threshold
        valid_pixels = signal_mask.sum()
        total_pixels = H * W
        valid_ratio = valid_pixels / total_pixels * 100

        print(f"  Signal threshold: {self.intensity_threshold}")
        print(f"  Valid pixels: {valid_pixels}/{total_pixels} ({valid_ratio:.1f}%)")

        # Initialize output
        tau_hr = np.zeros((H, W), dtype=np.float32)

        # Only process if there are valid pixels
        if valid_pixels > 0:
            # Normalize inputs
            lr_tau_max = tau_lr.max() + 1e-8
            lr_tau_norm = tau_lr / lr_tau_max
            hr_int_max = hr_intensity.max() + 1e-8
            hr_int_norm = hr_intensity / hr_int_max

            # Upscale LR tau using bicubic interpolation
            lr_tau_upscale = np.array(
                Image.fromarray(lr_tau_norm).resize((W, H), Image.BICUBIC)
            )

            # Convert to tensors
            hr_int_tensor = torch.from_numpy(hr_int_norm).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
            lr_int_tensor = downSizeImg_2D(hr_int_tensor.squeeze(0), self.scaling_ratio).unsqueeze(0).unsqueeze(0)  # [1, 1, H/8, W/8]
            lr_tau_tensor = torch.from_numpy(lr_tau_norm).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H/8, W/8]
            lr_tau_upscale_tensor = torch.from_numpy(lr_tau_upscale).float()  # [H, W]

            # Create coordinate grid
            hr_coord = make_coord((H, W), flatten=True).unsqueeze(0)

            # Create residual connection (single value)
            lr_pixel = lr_tau_upscale_tensor.flatten().unsqueeze(0).unsqueeze(2)  # [1, H*W, 1]

            # Prepare data for NIII model
            data = {
                'hr_int': hr_int_tensor,
                'lr_int': lr_int_tensor,
                'lr_tau': lr_tau_tensor,
                'lr_pixel': lr_pixel,
                'hr_coord': hr_coord,
                'lr_tau_max': lr_tau_max,
            }

            # Run NIII model
            result = NIII_process(data, self.scaling_ratio)  # Returns [B, 1, H, W]

            # Extract and reshape
            tau_pred = result.squeeze()  # [H, W]

            # Apply mask - only keep predictions for high-signal pixels
            tau_hr[signal_mask] = tau_pred[signal_mask]

            # Print statistics only for valid pixels
            if tau_hr[signal_mask].size > 0:
                print(f"  tau range (valid pixels): [{tau_hr[signal_mask].min():.3f}, {tau_hr[signal_mask].max():.3f}] ns")
        else:
            print("  WARNING: No valid pixels found above threshold!")

        return {
            'tau': tau_hr,
            'mask': signal_mask
        }

    def process_image(self, hist2d, hr_intensity, irf, skip_niii=False):
        """
        Process a single image through the complete pipeline

        Parameters:
        -----------
        hist2d : ndarray
            Histogram data [H, W, bins]
        hr_intensity : ndarray
            High-resolution intensity image [H, W]
        irf : ndarray
            Instrument response function
        skip_niii : bool
            Skip NIII stage if True (only run LLE)

        Returns:
        --------
        dict containing all results
        """
        # Stage 1: LLE
        lr_results = self.stage1_lle(hist2d, irf)

        if skip_niii:
            # Create dummy HR results (just upsample LR)
            from scipy.ndimage import zoom
            tau_lr_upsampled = zoom(lr_results['tau_lr'], 8, order=1)

            # Use LLE's own pixel filtering (LLE sets tau=0 for pixels below its threshold)
            # So the mask is simply where the upsampled tau is greater than 0
            signal_mask = tau_lr_upsampled > 0

            return {
                'tau_lr': lr_results['tau_lr'],
                'tau': tau_lr_upsampled,
                'mask': signal_mask
            }

        # Stage 2: NIII
        hr_results = self.stage2_niii(hr_intensity, lr_results)

        # Combine results
        return {
            'tau_lr': lr_results['tau_lr'],
            'tau': hr_results['tau'],
            'mask': hr_results['mask']
        }


def load_training_sample(sample_path):
    """
    Load training sample data from .mat file (HDF5 format)

    Parameters:
    -----------
    sample_path : str
        Path to training sample .mat file

    Returns:
    --------
    hist2d, hr_intensity, irf, tau_gt
    """
    print(f"\nLoading sample: {Path(sample_path).name}")

    # Training samples are in HDF5 format
    with h5py.File(sample_path, 'r') as f:
        # Load histogram - transpose to get [H, W, bins] and convert to float32
        hist = np.array(f['Hist'], dtype=np.float32).T  # [256, 256, 256]

        # Load intensity if available, otherwise compute from histogram
        if 'Int' in f:
            intensity = np.array(f['Int'], dtype=np.float32).T
        else:
            intensity = np.sum(hist, axis=2, dtype=np.float32)

        # Load IRF if available, otherwise generate default
        if 'IRF' in f:
            irf = np.array(f['IRF'], dtype=np.float32).flatten()
        else:
            # Generate default Gaussian IRF
            bins = hist.shape[2]
            t = np.arange(bins, dtype=np.float32)
            irf = np.exp(-(t - 14)**2 / (2 * 3**2))
            irf = irf / irf.max()

        # Load ground truth components
        tau_gt_components = np.array(f['tau_gt_components'], dtype=np.float32).T  # [256, 256, 2]
        f_gt_components = np.array(f['f_gt_components'], dtype=np.float32).T  # [256, 256, 2]

        # Compute single-exponential ground truth: tau_gt = tau1*f1 + tau2*f2
        tau_gt = (tau_gt_components[:, :, 0] * f_gt_components[:, :, 0] +
                  tau_gt_components[:, :, 1] * f_gt_components[:, :, 1])

    print(f"  Histogram shape: {hist.shape}")
    print(f"  Intensity shape: {intensity.shape}")
    print(f"  Intensity range: [{intensity.min():.1f}, {intensity.max():.1f}]")
    print(f"  IRF length: {len(irf)}")
    print(f"  Ground truth tau range: [{tau_gt.min():.3f}, {tau_gt.max():.3f}] ns")

    return hist, intensity, irf, tau_gt


def visualize_results(results, hr_intensity, tau_gt, sample_name, save_dir):
    """
    Visualize and save results

    Parameters:
    -----------
    results : dict
        Processing results
    hr_intensity : ndarray
        High-resolution intensity image
    tau_gt : ndarray
        Ground truth tau image
    sample_name : str
        Name of the sample
    save_dir : str
        Directory to save results
    """
    print(f"\nVisualizing results for {sample_name}...")

    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get mask if available
    mask = results.get('mask', None)

    # Row 1: Ground truth, Predicted HR, Predicted LR
    # Ground Truth Tau (masked by valid pixels)
    tau_gt_masked = np.where(mask, tau_gt, np.nan) if mask is not None else tau_gt
    im1 = axes[0, 0].imshow(tau_gt_masked, cmap='viridis')
    axes[0, 0].set_title('Ground Truth Tau', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Predicted Tau HR (masked by valid pixels)
    tau_hr = results['tau'].copy()
    tau_hr_masked = np.where(mask, tau_hr, np.nan) if mask is not None else tau_hr
    im2 = axes[0, 1].imshow(tau_hr_masked, cmap='viridis')
    axes[0, 1].set_title('Predicted Tau (HR)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Predicted Tau LR
    im3 = axes[0, 2].imshow(results['tau_lr'], cmap='viridis')
    axes[0, 2].set_title('Predicted Tau (LR)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: Intensity, Error map, hide third
    # Intensity
    im4 = axes[1, 0].imshow(hr_intensity, cmap='gray')
    axes[1, 0].set_title('Intensity Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Error map (absolute difference) - using YlOrRd for less saturated colors
    if mask is not None:
        error_map = np.abs(tau_hr - tau_gt)
        error_map_masked = np.where(mask, error_map, np.nan)
        im5 = axes[1, 1].imshow(error_map_masked, cmap='YlOrRd')
        axes[1, 1].set_title('Absolute Error (ns)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        axes[1, 1].axis('off')

    # Hide third subplot in row 2
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f'{sample_name}_single_exp_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {save_path}")
    plt.close()

    # Save numpy arrays including ground truth
    results['tau_gt'] = tau_gt
    for key in results:
        np_save_path = os.path.join(save_dir, f'{sample_name}_{key}.npy')
        np.save(np_save_path, results[key])

    print(f"  Saved numpy arrays to: {save_dir}")


def main():
    """Main evaluation function"""

    # Configuration
    lle_model_path = r'C:\Users\mcg11923\Thesis\LLE_parameter_1_3ns.pth'
    niii_model_path = r'C:\Users\mcg11923\Thesis\NIII_parameter_L8.pth'

    # Use training dataset
    sample_dir = r'C:\Users\mcg11923\Thesis\training_dataset_multiexp_s8'
    output_dir = r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation\single_exp_results'

    # Number of samples to evaluate (first N)
    num_samples = 10

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Single-Exponential FLIM Pipeline Evaluation")
    print("Using first 10 samples from training dataset")
    print("="*60)

    # Initialize pipeline
    pipeline = SingleExpPipeline(
        lle_model_path=lle_model_path,
        niii_model_path=niii_model_path,
        scaling_ratio=8,
        bin_width=0.039,
        intensity_threshold=3,  # Lower threshold to include more pixels
        device='cuda'
    )

    # Find all sample files from training dataset and take first 10
    all_sample_files = sorted(glob.glob(os.path.join(sample_dir, 'Sample_*.mat')))
    selected_files = all_sample_files[:num_samples]

    print(f"\nFound {len(all_sample_files)} total training samples")
    print(f"Processing first {len(selected_files)} samples")

    print("\nSelected samples:")
    for f in selected_files:
        print(f"  - {Path(f).name}")
    print()

    # Process each selected sample
    for sample_path in selected_files:
        sample_name = Path(sample_path).stem

        try:
            # Load data from training sample (now includes ground truth)
            hist2d, hr_intensity, irf, tau_gt = load_training_sample(sample_path)

            # Process through pipeline (skip NIII due to model file format issue)
            results = pipeline.process_image(hist2d, hr_intensity, irf, skip_niii=True)

            # Visualize and save results (now includes ground truth)
            visualize_results(results, hr_intensity, tau_gt, sample_name, output_dir)

            print(f"\n[SUCCESS] Processed {sample_name}")

        except Exception as e:
            print(f"\n[ERROR] Processing {sample_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
