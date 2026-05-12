"""
Create Photon-Bin Color-Coded Intensity Images

For each real FLIM sample, creates visualizations showing:
1. Intensity image (grayscale, based on photon counts)
2. Color-coded map showing which photon bin each pixel belongs to
3. Combined overlay with statistics

This helps visualize data quality spatially - showing which regions have
sufficient signal for reliable lifetime measurements.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import scipy.io as io

# Configuration
DATA_DIR = r'E:\RealdataSCC74A\mat_files'
OUTPUT_DIR = r'C:\Users\mcg11923\Thesis\photon_bin_visualizations'

# Photon bins (from stratified analysis)
PHOTON_BINS = [
    (50, 200),
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1200)
]

# Color scheme: Red (poor) -> Green (excellent)
BIN_COLORS = [
    '#D32F2F',  # Red - 50-200 (low quality)
    '#FF9800',  # Orange - 200-400 (medium-low)
    '#FDD835',  # Yellow - 400-600 (medium)
    '#7CB342',  # Light green - 600-800 (good)
    '#2E7D32',  # Dark green - 800-1200 (excellent)
]

BIN_LABELS = [
    '50-200\n(Low)',
    '200-400\n(Med-Low)',
    '400-600\n(Medium)',
    '600-800\n(Good)',
    '800-1200\n(Excellent)'
]

GROUPS = ['Control_Group', 'FPlusMinus_Group', 'FPlusPlus_Group', 'Rot_Group']


def load_sample(mat_file):
    """Load photon counts from mat file"""
    data = io.loadmat(mat_file)
    photons = data['photons']  # [H, W]
    return photons


def assign_to_bins(photons, bins):
    """
    Assign each pixel to a photon bin.

    Returns:
        bin_map: [H, W] array with bin index (0-4) or -1 for out of range
    """
    H, W = photons.shape
    bin_map = np.full((H, W), -1, dtype=np.int8)

    for bin_idx, (bin_min, bin_max) in enumerate(bins):
        mask = (photons >= bin_min) & (photons < bin_max)
        bin_map[mask] = bin_idx

    return bin_map


def create_photon_bin_visualization(photons, bin_map, sample_name, save_path):
    """
    Create 3-panel visualization:
    - Left: Intensity image (photon counts)
    - Center: Color-coded bin map
    - Right: Statistics and histogram
    """
    fig = plt.figure(figsize=(20, 6))

    # Calculate statistics
    total_pixels = photons.size
    valid_pixels = (bin_map >= 0).sum()

    # Panel 1: Intensity Image (Photon Counts)
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(photons, cmap='gray', interpolation='nearest')
    ax1.set_title('Photon Count Intensity', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)', fontsize=11)
    ax1.set_ylabel('Y (pixels)', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Photon Count', fontsize=11)

    # Add text with stats
    stats_text = f'Min: {photons.min():.0f}\nMax: {photons.max():.0f}\nMean: {photons.mean():.1f}\nMedian: {np.median(photons):.0f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: Color-Coded Bin Map
    ax2 = plt.subplot(1, 3, 2)

    # Create custom colormap
    colors_with_gray = ['#808080'] + BIN_COLORS  # Gray for out-of-range
    cmap = ListedColormap(colors_with_gray)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # Bin boundaries
    norm = BoundaryNorm(bounds, cmap.N)

    im2 = ax2.imshow(bin_map, cmap=cmap, norm=norm, interpolation='nearest')
    ax2.set_title('Photon Bin Classification', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)', fontsize=11)
    ax2.set_ylabel('Y (pixels)', fontsize=11)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#808080', label='Out of range')]
    for i, (color, label) in enumerate(zip(BIN_COLORS, BIN_LABELS)):
        legend_elements.append(Patch(facecolor=color, label=label.replace('\n', ' ')))

    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize=10, title='Photon Bins', title_fontsize=11)

    # Panel 3: Statistics and Histogram
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')

    # Create histogram of photon distribution
    ax3_hist = fig.add_axes([0.69, 0.55, 0.25, 0.35])

    # Plot histogram with bin colors
    bin_edges = [0, 50, 200, 400, 600, 800, 1200, photons.max() + 1]
    hist_data, _ = np.histogram(photons.flatten(), bins=bin_edges)

    x_pos = np.arange(len(hist_data))
    bar_colors = ['#808080'] + BIN_COLORS + ['#808080']  # Gray for <50 and >1200

    bars = ax3_hist.bar(x_pos, hist_data, color=bar_colors[:len(hist_data)],
                        edgecolor='black', linewidth=0.5)
    ax3_hist.set_xlabel('Photon Bin', fontsize=10)
    ax3_hist.set_ylabel('Pixel Count', fontsize=10)
    ax3_hist.set_title('Photon Distribution', fontsize=11, fontweight='bold')
    ax3_hist.set_xticks(x_pos)
    ax3_hist.set_xticklabels(['<50', '50-200', '200-400', '400-600',
                              '600-800', '800-1200', '>1200'], rotation=45, ha='right', fontsize=8)
    ax3_hist.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3_hist.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=8)

    # Statistics table
    table_ax = fig.add_axes([0.69, 0.15, 0.25, 0.35])
    table_ax.axis('off')

    # Calculate bin statistics
    stats_data = []
    stats_data.append(['Total Pixels', f'{total_pixels:,}'])
    stats_data.append(['Valid Pixels', f'{valid_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)'])
    stats_data.append(['', ''])
    stats_data.append(['Bin', 'Count (%)'])

    for bin_idx, (bin_min, bin_max) in enumerate(PHOTON_BINS):
        count = (bin_map == bin_idx).sum()
        pct = 100 * count / total_pixels
        stats_data.append([f'{bin_min}-{bin_max}', f'{count:,} ({pct:.1f}%)'])

    # Create table
    table = table_ax.table(cellText=stats_data, cellLoc='left',
                          loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code the bin rows
    for i in range(4, 4 + len(PHOTON_BINS)):
        table[(i, 0)].set_facecolor(BIN_COLORS[i-4])
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_facecolor(BIN_COLORS[i-4])
        table[(i, 1)].set_alpha(0.3)

    # Bold header rows
    for i in [0, 1, 3]:
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_text_props(weight='bold')

    # Main title
    plt.suptitle(f'{sample_name}\nPhoton-Bin Quality Assessment',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return valid_pixels, total_pixels


def process_group(group_name, data_dir, output_dir):
    """Process all samples in a group"""
    group_dir = os.path.join(data_dir, group_name)
    mat_files = sorted(glob.glob(os.path.join(group_dir, '*.mat')))

    if len(mat_files) == 0:
        print(f"  No .mat files found in {group_name}")
        return

    print(f"\n{'='*80}")
    print(f"Processing {group_name}")
    print(f"{'='*80}")
    print(f"Found {len(mat_files)} samples")

    # Create output directory for this group
    group_output_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_output_dir, exist_ok=True)

    group_stats = []

    for mat_file in mat_files:
        sample_name = os.path.basename(mat_file).replace('.mat', '')
        print(f"\n  Processing: {sample_name}...")

        # Load data
        photons = load_sample(mat_file)

        # Assign to bins
        bin_map = assign_to_bins(photons, PHOTON_BINS)

        # Create visualization
        save_path = os.path.join(group_output_dir, f'{sample_name}_photon_bins.png')
        valid_pixels, total_pixels = create_photon_bin_visualization(
            photons, bin_map, sample_name, save_path
        )

        print(f"    Saved: {save_path}")
        print(f"    Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")

        group_stats.append({
            'sample': sample_name,
            'photons': photons,
            'bin_map': bin_map,
            'valid_pct': 100*valid_pixels/total_pixels
        })

    return group_stats


def create_summary_figure(all_group_stats, output_dir):
    """Create summary comparison figure showing one sample from each group"""
    print(f"\n{'='*80}")
    print("Creating summary comparison figure")
    print(f"{'='*80}")

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    for group_idx, (group_name, stats) in enumerate(all_group_stats.items()):
        if len(stats) == 0:
            continue

        # Pick first sample from each group
        sample = stats[0]
        photons = sample['photons']
        bin_map = sample['bin_map']

        # Column 1: Intensity
        ax = axes[group_idx, 0]
        im = ax.imshow(photons, cmap='gray', interpolation='nearest')
        ax.set_title(f'{group_name}\nIntensity', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{sample["sample"]}', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Column 2: Bin Map
        ax = axes[group_idx, 1]
        colors_with_gray = ['#808080'] + BIN_COLORS
        cmap = ListedColormap(colors_with_gray)
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(bin_map, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title('Bin Classification', fontsize=12, fontweight='bold')

        # Column 3: Histogram
        ax = axes[group_idx, 2]
        bin_edges = [0, 50, 200, 400, 600, 800, 1200, photons.max() + 1]
        hist_data, _ = np.histogram(photons.flatten(), bins=bin_edges)
        bar_colors = ['#808080'] + BIN_COLORS + ['#808080']
        x_pos = np.arange(len(hist_data))
        ax.bar(x_pos, hist_data, color=bar_colors[:len(hist_data)],
               edgecolor='black', linewidth=0.5)
        ax.set_title('Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['<50', '50-200', '200-400', '400-600',
                           '600-800', '800-1200', '>1200'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Pixel Count', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Photon-Bin Quality Assessment - All Groups Comparison\n(One representative sample per group)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'summary_all_groups.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary figure saved: {save_path}")


def main():
    """Main execution"""
    print("=" * 80)
    print("PHOTON-BIN VISUALIZATION GENERATOR")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nPhoton bins: {PHOTON_BINS}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each group
    all_group_stats = {}

    for group_name in GROUPS:
        stats = process_group(group_name, DATA_DIR, OUTPUT_DIR)
        if stats:
            all_group_stats[group_name] = stats

    # Create summary figure
    if len(all_group_stats) > 0:
        create_summary_figure(all_group_stats, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print(f"Total groups processed: {len(all_group_stats)}")
    total_samples = sum(len(stats) for stats in all_group_stats.values())
    print(f"Total samples: {total_samples}")


if __name__ == '__main__':
    main()
