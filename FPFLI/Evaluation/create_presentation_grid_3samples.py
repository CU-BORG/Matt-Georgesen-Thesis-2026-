#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a presentation-ready grid showing 3 samples with large labels
Uses purple-to-orange colormap for lifetime predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os
from matplotlib.colors import LinearSegmentedColormap

# Create custom black-to-purple-to-orange-to-yellowish-white colormap
colors_purple_orange = ['#000000', '#5e2a84', '#ff6f00', '#fffacd']  # Black -> deep purple -> orange -> yellowish white
n_bins = 256
cmap_purple_orange = LinearSegmentedColormap.from_list('purple_orange', colors_purple_orange, N=n_bins)

def create_presentation_grid(results_dir, output_path, sample_indices=[0, 1, 2]):
    """
    Create a presentation grid with 3 samples and large labels

    Parameters:
    -----------
    results_dir : str
        Directory containing the saved numpy arrays
    output_path : str
        Path to save the grid figure
    sample_indices : list
        Indices of samples to include (default: first 3)
    """

    # Find all tau prediction files
    tau_files = sorted(glob.glob(os.path.join(results_dir, '*_tau.npy')))

    print(f"Found {len(tau_files)} total samples")
    print(f"Using samples at indices: {sample_indices}")

    # Load selected samples
    intensities = []
    predictions = []
    ground_truths = []
    sample_names = []

    for idx in sample_indices:
        if idx >= len(tau_files):
            print(f"Warning: Index {idx} out of range, skipping")
            continue

        tau_file = tau_files[idx]
        sample_name = Path(tau_file).stem.replace('_tau', '')
        sample_names.append(sample_name)

        # Load prediction
        tau = np.load(tau_file)
        predictions.append(tau)

        # Load ground truth
        tau_gt_file = tau_file.replace('_tau.npy', '_tau_gt.npy')
        if os.path.exists(tau_gt_file):
            tau_gt = np.load(tau_gt_file)
            ground_truths.append(tau_gt)
        else:
            ground_truths.append(None)

        # Load intensity
        intensity_from_sample = None
        sample_mat_path = os.path.join(
            r'C:\Users\mcg11923\Thesis\training_dataset_multiexp_s8',
            f'{sample_name}.mat'
        )
        if os.path.exists(sample_mat_path):
            import h5py
            try:
                with h5py.File(sample_mat_path, 'r') as f:
                    if 'Int' in f:
                        intensity_from_sample = np.array(f['Int'], dtype=np.float32).T
            except:
                pass

        if intensity_from_sample is not None:
            intensities.append(intensity_from_sample)
        else:
            intensities.append(np.ones_like(tau))

        print(f"  Loaded: {sample_name}")

    n_samples = len(predictions)

    # Find global min/max for tau and error (excluding zeros)
    all_tau_values = []
    all_error_values = []

    for tau, tau_gt in zip(predictions, ground_truths):
        nonzero_tau = tau[tau > 0]
        if len(nonzero_tau) > 0:
            all_tau_values.extend(nonzero_tau.flatten())

        if tau_gt is not None:
            error = np.abs(tau - tau_gt)
            nonzero_error = error[tau > 0]
            if len(nonzero_error) > 0:
                all_error_values.extend(nonzero_error.flatten())

    tau_vmin = np.min(all_tau_values) if len(all_tau_values) > 0 else 0
    tau_vmax = np.max(all_tau_values) if len(all_tau_values) > 0 else 1
    error_vmin = 0
    error_vmax = np.max(all_error_values) if len(all_error_values) > 0 else 1

    print(f"\nGlobal tau range: [{tau_vmin:.3f}, {tau_vmax:.3f}] ns")
    print(f"Global error range: [{error_vmin:.3f}, {error_vmax:.3f}] ns")

    # Build composite image by concatenating
    from matplotlib import cm

    def create_triplet(intensity, tau, tau_gt, tau_vmin, tau_vmax, error_vmin, error_vmax):
        """Create intensity-tau-error triplet for one sample"""
        # Normalize intensity to 0-1
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

        # Prepare tau display (normalized)
        tau_display = tau.copy()
        tau_display[tau <= 0] = np.nan
        tau_norm = (tau_display - tau_vmin) / (tau_vmax - tau_vmin + 1e-8)

        # Prepare error display (normalized)
        if tau_gt is not None:
            error = np.abs(tau - tau_gt)
            error_display = error.copy()
            error_display[tau <= 0] = np.nan
            error_norm = (error_display - error_vmin) / (error_vmax - error_vmin + 1e-8)
        else:
            error_norm = np.zeros_like(tau)

        # Convert normalized values to RGB using colormaps
        # Intensity: grayscale
        intensity_rgb = cm.gray(intensity_norm)[:, :, :3]

        # Tau: purple to orange (custom colormap)
        tau_rgb = cmap_purple_orange(tau_norm)[:, :, :3]
        tau_rgb[np.isnan(tau_norm)] = 1.0  # White background for NaN

        # Error: YlOrRd
        error_rgb = cm.YlOrRd(error_norm)[:, :, :3]
        error_rgb[np.isnan(error_norm)] = 1.0  # White background for NaN

        # Concatenate horizontally with small gap
        gap = np.ones((intensity.shape[0], 3, 3))  # 3 pixel white gap
        triplet = np.concatenate([intensity_rgb, gap, tau_rgb, gap, error_rgb], axis=1)
        return triplet

    # Create all triplets and stack vertically
    triplets = []
    for idx in range(n_samples):
        triplet = create_triplet(intensities[idx], predictions[idx], ground_truths[idx],
                                tau_vmin, tau_vmax, error_vmin, error_vmax)
        triplets.append(triplet)

    # Stack vertically with gaps
    gap_row = np.ones((5, triplets[0].shape[1], 3))  # 5 pixel gap between rows
    rows_with_gaps = [triplets[0]]
    for triplet in triplets[1:]:
        rows_with_gaps.append(gap_row)
        rows_with_gaps.append(triplet)
    composite = np.concatenate(rows_with_gaps, axis=0)

    # Create figure with extra space for colorbar
    fig_height = composite.shape[0] / 80  # Larger for presentation
    fig_width = (composite.shape[1] + 100) / 80  # Extra space for colorbar

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create main axis for composite image
    ax_main = fig.add_axes([0, 0, composite.shape[1]/(composite.shape[1]+100), 1])

    # Show composite image
    ax_main.imshow(composite)
    ax_main.axis('off')

    # Add column titles with large fonts
    triplet_width = triplets[0].shape[1]
    img_width = (triplet_width - 6) // 3  # Remove gaps, divide by 3
    y_title = -30

    ax_main.text(img_width//2, y_title, 'Intensity', ha='center', va='bottom',
                fontsize=20, fontweight='bold')
    ax_main.text(img_width + 3 + img_width//2, y_title, 'Predicted Tau (ns)', ha='center', va='bottom',
                fontsize=20, fontweight='bold')
    ax_main.text(2*img_width + 6 + img_width//2, y_title, 'Absolute Error (ns)', ha='center', va='bottom',
                fontsize=20, fontweight='bold')

    # Create colorbars in the extra space on the right
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    # Calculate colorbar position (in figure coordinates)
    cbar_left = (composite.shape[1] + 20) / (composite.shape[1] + 100)
    cbar_width = 0.025

    # Tau colorbar (top half) - purple to orange
    ax_tau_cbar = fig.add_axes([cbar_left, 0.55, cbar_width, 0.35])
    norm_tau = Normalize(vmin=tau_vmin, vmax=tau_vmax)
    cbar_tau = ColorbarBase(ax_tau_cbar, cmap=cmap_purple_orange, norm=norm_tau)
    cbar_tau.set_label('Tau (ns)', fontsize=16, fontweight='bold')
    cbar_tau.ax.tick_params(labelsize=14)

    # Error colorbar (bottom half)
    ax_err_cbar = fig.add_axes([cbar_left, 0.1, cbar_width, 0.35])
    norm_err = Normalize(vmin=error_vmin, vmax=error_vmax)
    cbar_err = ColorbarBase(ax_err_cbar, cmap=cm.YlOrRd, norm=norm_err)
    cbar_err.set_label('Error (ns)', fontsize=16, fontweight='bold')
    cbar_err.ax.tick_params(labelsize=14)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved presentation grid to: {output_path}")
    plt.close()


def main():
    """Main function"""

    results_dir = r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation\single_exp_results'
    output_path = os.path.join(results_dir, 'presentation_grid_3samples.png')

    print("="*60)
    print("Creating Presentation Grid (3 Samples)")
    print("="*60)

    # Use first 3 samples
    create_presentation_grid(results_dir, output_path, sample_indices=[0, 1, 2])

    print("\n" + "="*60)
    print("Presentation grid creation completed!")
    print("="*60)


if __name__ == '__main__':
    main()
