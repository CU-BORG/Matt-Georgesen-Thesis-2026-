#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package Real FLIM Data into .mat Files by Group

Converts real FLIM data from .asc ASCII and .npy files into .mat format
organized by experimental group for model inference and analysis.

Data Source: E:\RealdataSCC74A\2024-03-12 SCC74A Low O2
Groups: Control, F+-, F++, Rot (7 samples each, 28 total)

Output Format (compatible with model training pipeline):
- Hist: [H, W, 256] decay histograms
- tau_gt_components: [H, W, 2] lifetime components (tau1, tau2 in ns)
- f_gt_components: [H, W, 2] fraction components (f1, f2)
- photons: [H, W] total photon counts per pixel
- sample_info: metadata dictionary

@author: Package real data script
@date: 2026-03-01
"""

import os
import numpy as np
import scipy.io as io
import json
import glob
from datetime import datetime
from scipy.ndimage import zoom

# ============================================================================
# CONFIGURATION
# ============================================================================
SOURCE_DIR = r'E:\RealdataSCC74A\2024-03-12 SCC74A Low O2'
OUTPUT_BASE_DIR = r'E:\RealdataSCC74A\mat_files'
TARGET_SIZE = 256  # Standard size for model (upscale if needed)
TIME_BINS = 256

# Define group mappings
GROUPS = {
    'Control': 'Control_Group',
    'F+-': 'FPlusMinus_Group',
    'F++': 'FPlusPlus_Group',
    'Rot': 'Rot_Group'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_ascii_image(filepath):
    """
    Load a 2D image from ASCII file format (.asc)

    Parameters:
    -----------
    filepath : str
        Path to .asc file

    Returns:
    --------
    image : ndarray [H, W]
        2D image array
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if values:  # Skip empty lines
                data.append(values)

    return np.array(data)


def upscale_image(image, target_size=256, order=1):
    """
    Upscale image to target size using interpolation

    Parameters:
    -----------
    image : ndarray [H, W] or [H, W, D]
        Image to upscale
    target_size : int
        Target size (assumes square images)
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
    --------
    upscaled : ndarray [target_size, target_size] or [target_size, target_size, D]
        Upscaled image
    """
    if len(image.shape) == 2:
        H, W = image.shape
        zoom_factor_h = target_size / H
        zoom_factor_w = target_size / W
        return zoom(image, (zoom_factor_h, zoom_factor_w), order=order)
    elif len(image.shape) == 3:
        H, W, D = image.shape
        zoom_factors = (target_size / H, target_size / W, 1.0)  # Don't zoom time dimension
        return zoom(image, zoom_factors, order=order)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")


def load_and_package_sample(base_path, sample_name, upscale=True):
    """
    Load all data for a single sample and package into .mat format

    Parameters:
    -----------
    base_path : str
        Directory containing the data files
    sample_name : str
        Sample name prefix (e.g., "Control 1", "F+- 2")
    upscale : bool
        Whether to upscale to TARGET_SIZE

    Returns:
    --------
    data_dict : dict
        Dictionary ready for scipy.io.savemat with keys:
        - Hist: [H, W, 256]
        - tau_gt_components: [H, W, 2]
        - f_gt_components: [H, W, 2]
        - photons: [H, W]
        - sample_info: metadata dict

    Raises:
    -------
    FileNotFoundError if required files are missing
    """
    print(f"  Loading {sample_name}...")

    # Required files
    required_files = {
        'decay': f"{sample_name}_Ch2.npy",
        'tau1': f"{sample_name}_t1.asc",
        'tau2': f"{sample_name}_t2.asc",
        'a1': f"{sample_name}_a1.asc",
        'a2': f"{sample_name}_a2.asc"
    }

    # Optional files
    optional_files = {
        'photons': f"{sample_name}_photons.asc"
    }

    # Check required files exist
    for file_type, filename in required_files.items():
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")

    # Load decay histogram (NPY format)
    hist_path = os.path.join(base_path, required_files['decay'])
    Hist = np.load(hist_path)

    # Ensure correct shape [H, W, 256]
    if Hist.shape[0] == TIME_BINS:
        Hist = np.transpose(Hist, (1, 2, 0))  # [256, H, W] -> [H, W, 256]

    H_orig, W_orig, T = Hist.shape
    print(f"    Original size: {H_orig}×{W_orig}×{T}")

    # Load lifetime parameters (ASCII format, in picoseconds)
    tau1 = load_ascii_image(os.path.join(base_path, required_files['tau1']))
    tau2 = load_ascii_image(os.path.join(base_path, required_files['tau2']))

    # Convert from picoseconds to nanoseconds
    tau1 = tau1 / 1000.0
    tau2 = tau2 / 1000.0

    # Load amplitudes (ASCII format)
    a1 = load_ascii_image(os.path.join(base_path, required_files['a1']))
    a2 = load_ascii_image(os.path.join(base_path, required_files['a2']))

    # Normalize amplitudes to fractions that sum to 1
    total_amp = a1 + a2
    f1 = a1 / (total_amp + 1e-8)
    f2 = a2 / (total_amp + 1e-8)

    # Load photon counts if available
    photons_path = os.path.join(base_path, optional_files['photons'])
    if os.path.exists(photons_path):
        photons = load_ascii_image(photons_path)
    else:
        # Estimate from histogram if photon file not available
        photons = Hist.sum(axis=2)
        print(f"    Photons file not found, estimated from histogram")

    # Upscale if requested and needed
    if upscale and (H_orig != TARGET_SIZE or W_orig != TARGET_SIZE):
        print(f"    Upscaling from {H_orig}×{W_orig} to {TARGET_SIZE}×{TARGET_SIZE}...")
        Hist = upscale_image(Hist, TARGET_SIZE, order=1)
        tau1 = upscale_image(tau1, TARGET_SIZE, order=1)
        tau2 = upscale_image(tau2, TARGET_SIZE, order=1)
        f1 = upscale_image(f1, TARGET_SIZE, order=1)
        f2 = upscale_image(f2, TARGET_SIZE, order=1)
        photons = upscale_image(photons, TARGET_SIZE, order=1)
        H, W = TARGET_SIZE, TARGET_SIZE
    else:
        H, W = H_orig, W_orig

    # Package into .mat format structure
    # Stack tau components: [H, W, 2] with [:, :, 0] = tau1, [:, :, 1] = tau2
    tau_gt_components = np.stack([tau1, tau2], axis=2)

    # Stack fraction components: [H, W, 2] with [:, :, 0] = f1, [:, :, 1] = f2
    f_gt_components = np.stack([f1, f2], axis=2)

    # Create metadata
    sample_info = {
        'sample_name': sample_name,
        'original_size': [H_orig, W_orig],
        'final_size': [H, W],
        'time_bins': TIME_BINS,
        'upscaled': upscale and (H_orig != TARGET_SIZE or W_orig != TARGET_SIZE),
        'source_directory': os.path.basename(base_path),
        'conversion_date': datetime.now().isoformat(),
        'tau1_range_ns': [float(tau1.min()), float(tau1.max())],
        'tau2_range_ns': [float(tau2.min()), float(tau2.max())],
        'f1_range': [float(f1.min()), float(f1.max())],
        'f2_range': [float(f2.min()), float(f2.max())],
        'photon_range': [float(photons.min()), float(photons.max())],
        'photon_mean': float(photons.mean())
    }

    # Package all data
    data_dict = {
        'Hist': Hist,
        'tau_gt_components': tau_gt_components,
        'f_gt_components': f_gt_components,
        'photons': photons,
        'sample_info': sample_info
    }

    print(f"    [OK] Packaged successfully: {H}x{W}x{TIME_BINS}")
    print(f"      tau1: [{tau1.min():.3f}, {tau1.max():.3f}] ns")
    print(f"      tau2: [{tau2.min():.3f}, {tau2.max():.3f}] ns")
    print(f"      photons: mean={photons.mean():.0f}, range=[{photons.min():.0f}, {photons.max():.0f}]")

    return data_dict


def identify_sample_group(sample_name):
    """
    Identify which experimental group a sample belongs to

    Parameters:
    -----------
    sample_name : str
        Sample name (e.g., "Control 1", "F+- 2")

    Returns:
    --------
    group_key : str
        Group key ('Control', 'F+-', 'F++', 'Rot')
    group_name : str
        Group directory name
    """
    for group_key, group_name in GROUPS.items():
        if sample_name.startswith(group_key):
            return group_key, group_name

    raise ValueError(f"Cannot identify group for sample: {sample_name}")


# ============================================================================
# MAIN CONVERSION FUNCTION
# ============================================================================

def main():
    print("=" * 80)
    print(" PACKAGE REAL FLIM DATA INTO .MAT FILES BY GROUP")
    print("=" * 80)
    print(f"\nSource directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Target size: {TARGET_SIZE}×{TARGET_SIZE}")
    print(f"Time bins: {TIME_BINS}")

    # Create output base directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Find all sample names
    print(f"\nScanning for samples in {SOURCE_DIR}...")
    npy_files = glob.glob(os.path.join(SOURCE_DIR, '*_Ch2.npy'))
    sample_names = sorted([os.path.basename(f).replace('_Ch2.npy', '') for f in npy_files])

    print(f"Found {len(sample_names)} samples:")
    for name in sample_names:
        print(f"  - {name}")

    # Group samples by experimental group
    grouped_samples = {group_key: [] for group_key in GROUPS.keys()}

    for sample_name in sample_names:
        try:
            group_key, _ = identify_sample_group(sample_name)
            grouped_samples[group_key].append(sample_name)
        except ValueError as e:
            print(f"WARNING: {e}")
            continue

    print(f"\nSamples grouped:")
    for group_key, samples in grouped_samples.items():
        print(f"  {group_key}: {len(samples)} samples")

    # Process each group
    group_statistics = {}
    total_success = 0
    total_failed = 0

    for group_key, group_dir_name in GROUPS.items():
        samples = grouped_samples[group_key]

        if not samples:
            print(f"\nNo samples found for group: {group_key}")
            continue

        print("\n" + "=" * 80)
        print(f"PROCESSING GROUP: {group_key} ({len(samples)} samples)")
        print("=" * 80)

        # Create group output directory
        group_output_dir = os.path.join(OUTPUT_BASE_DIR, group_dir_name)
        os.makedirs(group_output_dir, exist_ok=True)

        group_stats = {
            'group_name': group_key,
            'total_samples': len(samples),
            'successful': 0,
            'failed': 0,
            'samples': []
        }

        # Process each sample in this group
        for sample_name in samples:
            try:
                # Load and package the data
                data_dict = load_and_package_sample(SOURCE_DIR, sample_name, upscale=True)

                # Generate output filename
                # Clean sample name for filename (replace special chars)
                clean_name = sample_name.replace(' ', '_').replace('+', 'Plus').replace('-', 'Minus')
                output_filename = f"{clean_name}_LowO2.mat"
                output_path = os.path.join(group_output_dir, output_filename)

                # Save to .mat file
                io.savemat(output_path, data_dict, do_compression=True)

                print(f"    [SAVED] {output_filename}")

                # Update statistics
                group_stats['successful'] += 1
                group_stats['samples'].append({
                    'sample_name': sample_name,
                    'output_file': output_filename,
                    'status': 'success',
                    'info': data_dict['sample_info']
                })
                total_success += 1

            except Exception as e:
                print(f"    [ERROR] processing {sample_name}: {e}")
                import traceback
                traceback.print_exc()

                group_stats['failed'] += 1
                group_stats['samples'].append({
                    'sample_name': sample_name,
                    'status': 'failed',
                    'error': str(e)
                })
                total_failed += 1

        # Save group statistics
        group_statistics[group_key] = group_stats

        print(f"\n{group_key} Summary:")
        print(f"  Successful: {group_stats['successful']}/{group_stats['total_samples']}")
        print(f"  Failed: {group_stats['failed']}/{group_stats['total_samples']}")
        print(f"  Output directory: {group_output_dir}")

    # Save overall manifest
    manifest = {
        'conversion_info': {
            'source_directory': SOURCE_DIR,
            'output_directory': OUTPUT_BASE_DIR,
            'conversion_date': datetime.now().isoformat(),
            'target_size': TARGET_SIZE,
            'time_bins': TIME_BINS,
            'total_samples': len(sample_names),
            'total_successful': total_success,
            'total_failed': total_failed
        },
        'groups': group_statistics
    }

    manifest_path = os.path.join(OUTPUT_BASE_DIR, 'conversion_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(" CONVERSION COMPLETE")
    print("=" * 80)
    print(f"\nTotal samples processed: {len(sample_names)}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
    print(f"Manifest saved to: {manifest_path}")

    # Print group summary
    print(f"\nGroup Summary:")
    for group_key, stats in group_statistics.items():
        print(f"  {group_key}: {stats['successful']}/{stats['total_samples']} samples")
        print(f"    -> {os.path.join(OUTPUT_BASE_DIR, GROUPS[group_key])}")

    print("\n" + "=" * 80)
    print()


if __name__ == '__main__':
    main()
