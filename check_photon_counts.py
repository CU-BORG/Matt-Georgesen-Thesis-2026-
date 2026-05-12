"""
Check if photon counts from _photons.asc files match the sums from _Ch2.npy files
"""

import os
import numpy as np

DATA_DIR = r'C:\Users\mcg11923\Thesis\FPFLI\Realdata\2024-02-29-SCC74A High O2'

# Sample names
samples = [f'Control {i}' for i in range(1, 8)]

def load_ascii_image(filepath):
    """Load 2D image from ASCII file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            data.append(values)
    return np.array(data)

print("=" * 100)
print("COMPARING PHOTON COUNTS: _photons.asc vs NPY histogram sums")
print("=" * 100)

for sample_name in samples:
    print(f"\n{sample_name}:")
    print("-" * 80)

    # Load photons from ASCII file
    photons_file = os.path.join(DATA_DIR, f"{sample_name}_photons.asc")
    if not os.path.exists(photons_file):
        print(f"  WARNING: {sample_name}_photons.asc not found")
        continue

    photons_asc = load_ascii_image(photons_file)

    # Load histogram from NPY file
    npy_file = os.path.join(DATA_DIR, f"{sample_name}_Ch2.npy")
    if not os.path.exists(npy_file):
        print(f"  WARNING: {sample_name}_Ch2.npy not found")
        continue

    hist = np.load(npy_file)

    # Ensure correct shape [H, W, 256]
    if hist.shape[0] == 256:
        hist = np.transpose(hist, (1, 2, 0))

    # Calculate photons from histogram (sum over time bins)
    photons_npy = np.sum(hist, axis=2)

    # Compare
    print(f"  photons.asc shape: {photons_asc.shape}")
    print(f"  NPY histogram shape: {hist.shape}")
    print(f"  NPY photon sum shape: {photons_npy.shape}")

    # Statistics
    print(f"\n  Photon count statistics:")
    print(f"    From .asc file:")
    print(f"      Min: {photons_asc.min():.1f}")
    print(f"      Max: {photons_asc.max():.1f}")
    print(f"      Mean: {photons_asc.mean():.1f}")
    print(f"      Median: {np.median(photons_asc):.1f}")

    print(f"    From .npy histogram sum:")
    print(f"      Min: {photons_npy.min():.1f}")
    print(f"      Max: {photons_npy.max():.1f}")
    print(f"      Mean: {photons_npy.mean():.1f}")
    print(f"      Median: {np.median(photons_npy):.1f}")

    # Check if they match
    difference = photons_asc - photons_npy
    abs_diff = np.abs(difference)

    print(f"\n  Difference (asc - npy):")
    print(f"    Min: {difference.min():.1f}")
    print(f"    Max: {difference.max():.1f}")
    print(f"    Mean: {difference.mean():.1f}")
    print(f"    Mean absolute: {abs_diff.mean():.1f}")
    print(f"    Max absolute: {abs_diff.max():.1f}")

    # Percentage difference
    # Avoid division by zero
    mask = photons_asc > 0
    if mask.sum() > 0:
        pct_diff = 100 * abs_diff[mask] / photons_asc[mask]
        print(f"    Mean % difference (non-zero pixels): {pct_diff.mean():.2f}%")
        print(f"    Max % difference (non-zero pixels): {pct_diff.max():.2f}%")

    # Check if identical
    if np.allclose(photons_asc, photons_npy, rtol=1e-5, atol=1e-5):
        print(f"\n  MATCH: photons.asc and NPY histogram sum are identical (within tolerance)")
    else:
        # Count pixels with differences
        diff_mask = abs_diff > 0.5  # More than 0.5 photon difference
        n_diff = diff_mask.sum()
        pct_diff_pixels = 100 * n_diff / diff_mask.size
        print(f"\n  DIFFERENCE: {n_diff} pixels ({pct_diff_pixels:.1f}%) differ by >0.5 photons")

        # Show some examples
        if n_diff > 0:
            print(f"\n  Example differences (first 5):")
            diff_coords = np.argwhere(diff_mask)[:5]
            for idx, (i, j) in enumerate(diff_coords):
                print(f"    Pixel ({i},{j}): asc={photons_asc[i,j]:.1f}, npy={photons_npy[i,j]:.1f}, diff={difference[i,j]:.1f}")

print("\n" + "=" * 100)
print("COMPARISON COMPLETE")
print("=" * 100)
