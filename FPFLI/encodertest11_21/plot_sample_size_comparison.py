#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot MAE results vs sample size for fully flat random models
Creates comprehensive comparison plots showing how performance improves with dataset size

@author: mg
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Results from inference runs
results = {
    'sizes': [1, 10, 100, 200, 500],
    'tau1_mae': [0.150310, 0.093500, 0.086395, 0.085903, 0.085646],
    'tau2_mae': [0.466966, 0.160728, 0.144087, 0.141274, 0.140893],
    'f_mae': [0.131419, 0.065284, 0.053103, 0.053017, 0.052108]
}

# Create output directory
output_dir = r'C:\Users\mcg11923\Thesis\sample_size_analysis'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("PLOTTING SAMPLE SIZE COMPARISON")
print("=" * 80)

sizes = np.array(results['sizes'])
tau1_mae = np.array(results['tau1_mae'])
tau2_mae = np.array(results['tau2_mae'])
f_mae = np.array(results['f_mae'])

# =============================================================================
# PLOT 1: Individual parameter plots (3x1 grid)
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('MAE vs Training Dataset Size - Fully Flat Random (tau1=[0.2,0.7])',
             fontsize=16, fontweight='bold')

# Tau1 plot
ax = axes[0]
ax.plot(sizes, tau1_mae, 'o-', linewidth=2.5, markersize=10, color='#2E86AB', label='tau1')
ax.fill_between(sizes, tau1_mae, alpha=0.3, color='#2E86AB')
ax.set_ylabel('tau1 MAE (ns)', fontsize=13, fontweight='bold')
ax.set_title('tau1: Short Lifetime Component [0.2, 0.7] ns', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)

# Add value labels
for i, (x, y) in enumerate(zip(sizes, tau1_mae)):
    ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
               xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

# Tau2 plot
ax = axes[1]
ax.plot(sizes, tau2_mae, 's-', linewidth=2.5, markersize=10, color='#A23B72', label='tau2')
ax.fill_between(sizes, tau2_mae, alpha=0.3, color='#A23B72')
ax.set_ylabel('tau2 MAE (ns)', fontsize=13, fontweight='bold')
ax.set_title('tau2: Long Lifetime Component [1.2, 4.5] ns', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)

# Add value labels
for i, (x, y) in enumerate(zip(sizes, tau2_mae)):
    ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
               xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

# F1 plot
ax = axes[2]
ax.plot(sizes, f_mae, '^-', linewidth=2.5, markersize=10, color='#F18F01', label='f')
ax.fill_between(sizes, f_mae, alpha=0.3, color='#F18F01')
ax.set_xlabel('Number of Training Images', fontsize=13, fontweight='bold')
ax.set_ylabel('f MAE (fraction)', fontsize=13, fontweight='bold')
ax.set_title('f: Bound Fraction [0.1, 0.9]', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)

# Add value labels
for i, (x, y) in enumerate(zip(sizes, f_mae)):
    ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
               xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'mae_vs_sample_size_individual.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"Saved: {plot1_path}")
plt.close()

# =============================================================================
# PLOT 2: All parameters together (single plot)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('MAE vs Training Dataset Size - All Parameters',
             fontsize=16, fontweight='bold')

ax.plot(sizes, tau1_mae, 'o-', linewidth=2.5, markersize=12,
        color='#2E86AB', label='tau1 (ns)', alpha=0.8)
ax.plot(sizes, tau2_mae, 's-', linewidth=2.5, markersize=12,
        color='#A23B72', label='tau2 (ns)', alpha=0.8)
ax.plot(sizes, f_mae, '^-', linewidth=2.5, markersize=12,
        color='#F18F01', label='f (fraction)', alpha=0.8)

ax.set_xlabel('Number of Training Images', fontsize=14, fontweight='bold')
ax.set_ylabel('MAE', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

# Add annotations for key points
# Highlight the "plateau" region
ax.axvspan(100, 500, alpha=0.1, color='green', label='Plateau region')
ax.text(200, 0.4, 'Performance plateau\n(diminishing returns)',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plot2_path = os.path.join(output_dir, 'mae_vs_sample_size_combined.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"Saved: {plot2_path}")
plt.close()

# =============================================================================
# PLOT 3: Log-log plot to show convergence behavior
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('MAE vs Training Dataset Size - Log-Log Scale',
             fontsize=16, fontweight='bold')

ax.loglog(sizes, tau1_mae, 'o-', linewidth=2.5, markersize=12,
          color='#2E86AB', label='tau1 (ns)', alpha=0.8)
ax.loglog(sizes, tau2_mae, 's-', linewidth=2.5, markersize=12,
          color='#A23B72', label='tau2 (ns)', alpha=0.8)
ax.loglog(sizes, f_mae, '^-', linewidth=2.5, markersize=12,
          color='#F18F01', label='f (fraction)', alpha=0.8)

ax.set_xlabel('Number of Training Images (log scale)', fontsize=14, fontweight='bold')
ax.set_ylabel('MAE (log scale)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', which='both')
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

plt.tight_layout()
plot3_path = os.path.join(output_dir, 'mae_vs_sample_size_loglog.png')
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"Saved: {plot3_path}")
plt.close()

# =============================================================================
# PLOT 4: Improvement relative to baseline (n=10)
# =============================================================================
baseline_idx = 1  # n=10
baseline_tau1 = tau1_mae[baseline_idx]
baseline_tau2 = tau2_mae[baseline_idx]
baseline_f = f_mae[baseline_idx]

improvement_tau1 = (baseline_tau1 - tau1_mae) / baseline_tau1 * 100
improvement_tau2 = (baseline_tau2 - tau2_mae) / baseline_tau2 * 100
improvement_f = (baseline_f - f_mae) / baseline_f * 100

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('Improvement Relative to Baseline (n=10)',
             fontsize=16, fontweight='bold')

ax.plot(sizes, improvement_tau1, 'o-', linewidth=2.5, markersize=12,
        color='#2E86AB', label='tau1', alpha=0.8)
ax.plot(sizes, improvement_tau2, 's-', linewidth=2.5, markersize=12,
        color='#A23B72', label='tau2', alpha=0.8)
ax.plot(sizes, improvement_f, '^-', linewidth=2.5, markersize=12,
        color='#F18F01', label='f', alpha=0.8)

ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Number of Training Images', fontsize=14, fontweight='bold')
ax.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_xticks(sizes)
ax.set_xticklabels(sizes)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)

# Add value labels
for i, (x, y1, y2, y3) in enumerate(zip(sizes, improvement_tau1, improvement_tau2, improvement_f)):
    if i > 0:  # Skip baseline
        ax.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8, color='#2E86AB')

plt.tight_layout()
plot4_path = os.path.join(output_dir, 'improvement_vs_sample_size.png')
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"Saved: {plot4_path}")
plt.close()

# =============================================================================
# PLOT 5: Bar chart comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle('MAE Comparison Across Sample Sizes - Bar Chart',
             fontsize=16, fontweight='bold')

x_pos = np.arange(len(sizes))
width = 0.25

bars1 = ax.bar(x_pos - width, tau1_mae, width, label='tau1 (ns)',
               color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x_pos, tau2_mae, width, label='tau2 (ns)',
               color='#A23B72', alpha=0.8)
bars3 = ax.bar(x_pos + width, f_mae, width, label='f (fraction)',
               color='#F18F01', alpha=0.8)

ax.set_xlabel('Number of Training Images', fontsize=14, fontweight='bold')
ax.set_ylabel('MAE', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'n={s}' for s in sizes])
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plot5_path = os.path.join(output_dir, 'mae_vs_sample_size_barchart.png')
plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
print(f"Saved: {plot5_path}")
plt.close()

# =============================================================================
# Print summary statistics
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nMAE by Sample Size:")
print(f"{'Size':<10}{'tau1':<15}{'tau2':<15}{'f':<15}")
print("-" * 55)
for i, size in enumerate(sizes):
    print(f"{size:<10}{tau1_mae[i]:<15.6f}{tau2_mae[i]:<15.6f}{f_mae[i]:<15.6f}")

print("\nImprovement from n=1 to n=500:")
print(f"  tau1: {(tau1_mae[0] - tau1_mae[-1]) / tau1_mae[0] * 100:.1f}%")
print(f"  tau2: {(tau2_mae[0] - tau2_mae[-1]) / tau2_mae[0] * 100:.1f}%")
print(f"  f:    {(f_mae[0] - f_mae[-1]) / f_mae[0] * 100:.1f}%")

print("\nImprovement from n=10 to n=500:")
print(f"  tau1: {(tau1_mae[1] - tau1_mae[-1]) / tau1_mae[1] * 100:.1f}%")
print(f"  tau2: {(tau2_mae[1] - tau2_mae[-1]) / tau2_mae[1] * 100:.1f}%")
print(f"  f:    {(f_mae[1] - f_mae[-1]) / f_mae[1] * 100:.1f}%")

print("\nImprovement from n=100 to n=500:")
print(f"  tau1: {(tau1_mae[2] - tau1_mae[-1]) / tau1_mae[2] * 100:.1f}%")
print(f"  tau2: {(tau2_mae[2] - tau2_mae[-1]) / tau2_mae[2] * 100:.1f}%")
print(f"  f:    {(f_mae[2] - f_mae[-1]) / f_mae[2] * 100:.1f}%")

print("\n" + "=" * 80)
print("All plots saved to:", output_dir)
print("=" * 80)
