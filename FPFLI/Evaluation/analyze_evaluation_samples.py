#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze evaluation sample images using single-exponential LLE model
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import glob
from pathlib import Path

# Add paths for model imports
sys.path.append(r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation')

from Pred_local_lifetime import LLE
from scipy.ndimage import zoom


class EvaluationAnalyzer:
    """Analyze evaluation samples with single-exponential LLE"""

    def __init__(self, lle_model_path, scaling_ratio=8, bin_width=0.039):
        """
        Initialize analyzer

        Parameters:
        -----------
        lle_model_path : str
            Path to trained LLE model
        scaling_ratio : int
            Downsampling factor
        bin_width : float
            Time bin width in nanoseconds
        """
        self.lle_model_path = lle_model_path
        self.scaling_ratio = scaling_ratio
        self.bin_width = bin_width

        print(f"Initialized Evaluation Analyzer")
        print(f"  LLE model: {lle_model_path}")
        print(f"  Scaling ratio: {scaling_ratio}")
        print(f"  Bin width: {bin_width} ns")

    def load_sample(self, sample_path):
        """
        Load evaluation sample from .mat file

        Parameters:
        -----------
        sample_path : str
            Path to sample file

        Returns:
        --------
        hist, intensity, irf
        """
        print(f"\nLoading: {Path(sample_path).name}")

        try:
            # Try loading as HDF5 format first
            with h5py.File(sample_path, 'r') as f:
                hist = np.array(f['Hist'], dtype=np.float32).T

                if 'Int' in f:
                    intensity = np.array(f['Int'], dtype=np.float32).T
                else:
                    intensity = np.sum(hist, axis=2, dtype=np.float32)

                if 'IRF' in f:
                    irf = np.array(f['IRF'], dtype=np.float32).flatten()
                else:
                    # Generate default Gaussian IRF
                    bins = hist.shape[2]
                    t = np.arange(bins, dtype=np.float32)
                    irf = np.exp(-(t - 14)**2 / (2 * 3**2))
                    irf = irf / irf.max()

        except (OSError, KeyError):
            # Fall back to scipy for older MATLAB format
            data = sio.loadmat(sample_path)

            if 'Hist' in data:
                hist = data['Hist'].astype(np.float32)
            else:
                raise KeyError("No 'Hist' field found in .mat file")

            if 'Int' in data:
                intensity = data['Int'].astype(np.float32).squeeze()
            else:
                intensity = np.sum(hist, axis=2, dtype=np.float32)

            if 'IRF' in data:
                irf = data['IRF'].astype(np.float32).flatten()
            else:
                # Generate default Gaussian IRF
                bins = hist.shape[2]
                t = np.arange(bins, dtype=np.float32)
                irf = np.exp(-(t - 14)**2 / (2 * 3**2))
                irf = irf / irf.max()

        print(f"  Histogram shape: {hist.shape}")
        print(f"  Intensity shape: {intensity.shape}")
        print(f"  Intensity range: [{intensity.min():.1f}, {intensity.max():.1f}]")
        print(f"  IRF length: {len(irf)}")

        return hist, intensity, irf

    def process_sample(self, hist, irf):
        """
        Process sample through LLE

        Parameters:
        -----------
        hist : ndarray
            Histogram data [H, W, bins]
        irf : ndarray
            Instrument response function

        Returns:
        --------
        tau_lr, tau_hr (upsampled)
        """
        print("\n" + "="*50)
        print("Running LLE (Single-Exponential)")
        print("="*50)

        # Run LLE
        tau_lr = LLE(
            hist,
            irf,
            r=self.scaling_ratio,
            bin_w=self.bin_width,
            p_path=self.lle_model_path
        )

        # Upsample to high resolution
        tau_hr = zoom(tau_lr, self.scaling_ratio, order=1)

        # Count valid pixels
        valid_lr = (tau_lr > 0).sum()
        valid_hr = (tau_hr > 0).sum()
        total_lr = tau_lr.size
        total_hr = tau_hr.size

        print(f"  LR valid pixels: {valid_lr}/{total_lr} ({valid_lr/total_lr*100:.1f}%)")
        print(f"  HR valid pixels: {valid_hr}/{total_hr} ({valid_hr/total_hr*100:.1f}%)")

        if valid_lr > 0:
            valid_taus_lr = tau_lr[tau_lr > 0]
            print(f"  LR tau range: [{valid_taus_lr.min():.3f}, {valid_taus_lr.max():.3f}] ns")

        if valid_hr > 0:
            valid_taus_hr = tau_hr[tau_hr > 0]
            print(f"  HR tau range: [{valid_taus_hr.min():.3f}, {valid_taus_hr.max():.3f}] ns")

        return tau_lr, tau_hr

    def visualize_results(self, intensity, tau_lr, tau_hr, sample_name, save_dir):
        """
        Visualize and save results

        Parameters:
        -----------
        intensity : ndarray
            Intensity image
        tau_lr : ndarray
            Low-res tau
        tau_hr : ndarray
            High-res tau
        sample_name : str
            Sample name
        save_dir : str
            Directory to save results
        """
        print(f"\nVisualizing results...")

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Intensity
        im1 = axes[0].imshow(intensity, cmap='gray')
        axes[0].set_title('Intensity Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Tau LR
        tau_lr_display = np.where(tau_lr > 0, tau_lr, np.nan)
        im2 = axes[1].imshow(tau_lr_display, cmap='viridis')
        axes[1].set_title('Tau (LR - 32x32)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Tau HR
        tau_hr_display = np.where(tau_hr > 0, tau_hr, np.nan)
        im3 = axes[2].imshow(tau_hr_display, cmap='viridis')
        axes[2].set_title('Tau (HR - Upsampled)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(f'Single-Exponential LLE Analysis: {sample_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(save_dir, f'{sample_name}_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")
        plt.close()

        # Save numpy arrays
        np.save(os.path.join(save_dir, f'{sample_name}_intensity.npy'), intensity)
        np.save(os.path.join(save_dir, f'{sample_name}_tau_lr.npy'), tau_lr)
        np.save(os.path.join(save_dir, f'{sample_name}_tau_hr.npy'), tau_hr)


def main():
    """Main analysis function"""

    # Configuration
    lle_model_path = r'C:\Users\mcg11923\Thesis\LLE_parameter_1_3ns.pth'
    sample_dir = r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation\sample_data'
    output_dir = r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation\evaluation_analysis_results'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Evaluation Sample Analysis")
    print("Single-Exponential LLE Model")
    print("="*60)

    # Initialize analyzer
    analyzer = EvaluationAnalyzer(
        lle_model_path=lle_model_path,
        scaling_ratio=8,
        bin_width=0.039
    )

    # Find all sample files
    sample_files = sorted(glob.glob(os.path.join(sample_dir, '*.mat')))

    print(f"\nFound {len(sample_files)} sample files:")
    for f in sample_files:
        print(f"  - {Path(f).name}")

    # Process each sample
    for sample_path in sample_files:
        sample_name = Path(sample_path).stem

        try:
            # Load sample
            hist, intensity, irf = analyzer.load_sample(sample_path)

            # Process through LLE
            tau_lr, tau_hr = analyzer.process_sample(hist, irf)

            # Visualize results
            analyzer.visualize_results(intensity, tau_lr, tau_hr, sample_name, output_dir)

            print(f"\n[SUCCESS] Processed {sample_name}")

        except Exception as e:
            print(f"\n[ERROR] Processing {sample_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        print("\n" + "-"*60)

    print("\n" + "="*60)
    print("Analysis completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
