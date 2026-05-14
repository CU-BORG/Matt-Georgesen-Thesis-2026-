"""
Plot the Instrument Response Function (IRF) from the MATLAB script
Generate_MultiExp_DecayStart_Bin50.m

The IRF is a Gaussian function used for convolution with exponential decays.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters from the MATLAB script
IRF_CENTER_BIN = 62  # IRF centered at bin 62
bin_width = 0.048828  # Time bin width in nanoseconds (12.5ns / 256 bins)
FWHM = 0.120  # Full Width Half Maximum in nanoseconds (120 ps)
n_bins = 256  # Number of time bins

# Generate IRF using the same formula as IRF_gaussian.m
t = np.arange(1, n_bins + 1)  # Bins 1 to 256 (MATLAB indexing)
sig0 = FWHM / 2.3548 / bin_width  # Standard deviation in bins
IRF = np.exp(-(t - IRF_CENTER_BIN)**2 / (2 * sig0**2))

# Normalize to unit amplitude (as done in MATLAB script)
IRF = IRF / np.max(IRF)

# Convert time bins to nanoseconds
time_ns = (t - 1) * bin_width  # Start from 0 ns (bin 1 = 0 ns)

# Create high-resolution version for smooth plotting (10000 points)
t_hires = np.linspace(1, n_bins, 10000)  # 10000 points for smooth curve
IRF_hires = np.exp(-(t_hires - IRF_CENTER_BIN)**2 / (2 * sig0**2))
IRF_hires = IRF_hires / np.max(IRF_hires)
time_ns_hires = (t_hires - 1) * bin_width

# Calculate center time and FWHM points
center_time = IRF_CENTER_BIN * bin_width
half_max_idx = np.where(IRF_hires >= 0.5)[0]
if len(half_max_idx) > 0:
    fwhm_left = time_ns_hires[half_max_idx[0]]
    fwhm_right = time_ns_hires[half_max_idx[-1]]
    fwhm_measured = fwhm_right - fwhm_left

# Create simple single plot
plt.figure(figsize=(8, 6))

plt.plot(time_ns_hires, IRF_hires, 'b-', linewidth=2.5)

# Mark FWHM with vertical lines and annotation
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
plt.axvline(x=fwhm_left, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
plt.axvline(x=fwhm_right, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Add double-headed arrow for FWHM at the half-max level
plt.annotate('', xy=(fwhm_right, 0.5), xytext=(fwhm_left, 0.5),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))

# Add FWHM label to the side (right of the peak)
plt.text(fwhm_right + 0.5, 0.5, f'FWHM = {FWHM*1000:.0f} ps',
         ha='left', va='center', fontsize=12, color='red', fontweight='bold')

plt.xlabel('Time (ns)', fontsize=14)
plt.ylabel('Normalized Intensity', fontsize=14)
plt.title('Instrument Response Function', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim([0, time_ns_hires[-1]])

plt.tight_layout()
plt.savefig('irf_plot.png', dpi=300, bbox_inches='tight')
print("IRF plot saved as 'irf_plot.png'")

# Print statistics
print("\n" + "="*60)
print("IRF Statistics:")
print("="*60)
print(f"Center bin: {IRF_CENTER_BIN}")
print(f"Center time: {center_time:.3f} ns")
print(f"Bin width: {bin_width:.6f} ns")
print(f"Total time range: 0 to {time_ns[-1]:.3f} ns")
print(f"FWHM: {FWHM} ns = {FWHM * 1000:.1f} ps")
print(f"Standard deviation (sigma): {sig0 * bin_width:.4f} ns")
print(f"Peak amplitude (normalized): {np.max(IRF):.4f}")
print(f"Number of bins: {n_bins}")
print("="*60)

plt.show()
