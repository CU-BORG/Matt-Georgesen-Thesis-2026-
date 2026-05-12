#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a panel showing all high-res tau predictions side by side
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os

def create_prediction_panel(results_dir, output_path):
    """
    Create a panel with all high-res predictions

    Parameters:
    -----------
    results_dir : str
        Directory containing the saved numpy arrays
    output_path : str
        Path to save the panel figure
    """

    # Find all tau prediction files
    tau_files = sorted(glob.glob(os.path.join(results_dir, '*_tau.npy')))

    print(f"Found {len(tau_files)} samples")

    # Load all predictions
    predictions = []
    sample_names = []

    for tau_file in tau_files:
        sample_name = Path(tau_file).stem.replace('_tau', '')
        sample_names.append(sample_name)

        # Load prediction
        tau = np.load(tau_file)
        predictions.append(tau)

        print(f"  Loaded: {sample_name}")

    # Determine grid layout (2 rows x 5 columns for 10 samples)
    n_samples = len(predictions)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))

    # Flatten axes array for easy iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Find global min/max for consistent color scale (excluding zeros)
    all_nonzero_values = []
    for tau in predictions:
        nonzero_values = tau[tau > 0]
        if len(nonzero_values) > 0:
            all_nonzero_values.extend(nonzero_values.flatten())

    vmin = np.min(all_nonzero_values) if len(all_nonzero_values) > 0 else 0
    vmax = np.max(all_nonzero_values) if len(all_nonzero_values) > 0 else 1

    print(f"\nGlobal tau range (excluding zeros): [{vmin:.3f}, {vmax:.3f}] ns")

    # Plot each prediction
    for idx, (tau, sample_name) in enumerate(zip(predictions, sample_names)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Set zeros to NaN for visualization (so they appear as background)
        tau_display = np.where(tau > 0, tau, np.nan)

        # Plot
        im = ax.imshow(tau_display, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(sample_name.replace('Sample_', 'S').replace('_MultiExp_', '\n'),
                     fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Tau (ns)', rotation=270, labelpad=20, fontsize=12)

    # Add title
    fig.suptitle('High-Resolution Tau Predictions (Single-Exponential LLE)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])

    # Save figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved panel to: {output_path}")
    plt.close()


def main():
    """Main function"""

    results_dir = r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation\single_exp_results'
    output_path = os.path.join(results_dir, 'all_predictions_panel.png')

    print("="*60)
    print("Creating High-Res Prediction Panel")
    print("="*60)

    create_prediction_panel(results_dir, output_path)

    print("\n" + "="*60)
    print("Panel creation completed!")
    print("="*60)


if __name__ == '__main__':
    main()
