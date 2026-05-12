#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze training losses from model checkpoint
Plot training and validation loss curves for all 3 stages

@author: mg
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Path to the model checkpoint
MODEL_PATH = r'C:\Users\mcg11923\Thesis\training_logs_fully_flatrandom_20260219_072138\model_final_fully_flatrandom.pth'
OUTPUT_DIR = r'C:\Users\mcg11923\Thesis\training_logs_fully_flatrandom_20260219_072138'

print("=" * 80)
print("TRAINING LOSS ANALYSIS - FULLY FLAT RANDOM MODEL")
print("=" * 80)
print(f"\nLoading checkpoint from: {MODEL_PATH}")

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

# Check what's in the checkpoint
print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

if 'all_records' not in checkpoint:
    print("\nERROR: No 'all_records' found in checkpoint!")
    print("Available keys:", list(checkpoint.keys()))
    sys.exit(1)

all_records = checkpoint['all_records']
print(f"\nStages found: {list(all_records.keys())}")

# Create figure with subplots for each stage
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Training Loss Analysis - Fully Flat Random Model', fontsize=16, fontweight='bold')

stages = ['stage1', 'stage2', 'stage3']
stage_names = ['Stage 1: Reconstruction', 'Stage 2: Parameter Learning', 'Stage 3: Joint Optimization']

# Top row: Individual stage losses
for idx, (stage, stage_name) in enumerate(zip(stages, stage_names)):
    ax = axes[0, idx]

    if stage in all_records and all_records[stage] is not None:
        record = all_records[stage]

        if 'Train_Loss' in record and 'Val_Loss' in record:
            train_loss = record['Train_Loss']
            val_loss = record['Val_Loss']

            epochs = range(1, len(train_loss) + 1)

            ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
            ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, alpha=0.7)

            # Mark best validation loss
            best_val_idx = np.argmin(val_loss)
            best_val = val_loss[best_val_idx]
            ax.plot(best_val_idx + 1, best_val, 'r*', markersize=15,
                   label=f'Best Val (epoch {best_val_idx+1})')

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(stage_name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Print statistics
            print(f"\n{stage_name}:")
            print(f"  Epochs trained: {len(train_loss)}")
            print(f"  Final train loss: {train_loss[-1]:.6f}")
            print(f"  Final val loss: {val_loss[-1]:.6f}")
            print(f"  Best val loss: {best_val:.6f} (epoch {best_val_idx+1})")
            print(f"  Train loss range: [{min(train_loss):.6f}, {max(train_loss):.6f}]")
            print(f"  Val loss range: [{min(val_loss):.6f}, {max(val_loss):.6f}]")

            # Check for overfitting
            if train_loss[-1] < val_loss[-1]:
                gap = val_loss[-1] - train_loss[-1]
                print(f"  WARNING: Overfitting detected: val-train gap = {gap:.6f}")
        else:
            ax.text(0.5, 0.5, 'No loss data available',
                   ha='center', va='center', transform=ax.transAxes)
            print(f"\n{stage_name}: No loss data available")
            print(f"  Available keys: {list(record.keys())}")
    else:
        ax.text(0.5, 0.5, f'{stage} not found',
               ha='center', va='center', transform=ax.transAxes)
        print(f"\n{stage_name}: Stage not found in records")

# Bottom row: Combined validation loss and detailed component losses

# Bottom left: All validation losses on same plot
ax = axes[1, 0]
colors = ['blue', 'green', 'red']
epoch_offset = 0

for idx, (stage, stage_name) in enumerate(zip(stages, stage_names)):
    if stage in all_records and all_records[stage] is not None:
        record = all_records[stage]
        if 'Val_Loss' in record:
            val_loss = record['Val_Loss']
            epochs = np.arange(epoch_offset, epoch_offset + len(val_loss))
            ax.plot(epochs, val_loss, color=colors[idx], linewidth=2,
                   label=f'{stage_name}', alpha=0.7)
            epoch_offset += len(val_loss)

ax.set_xlabel('Cumulative Epoch', fontsize=11)
ax.set_ylabel('Validation Loss', fontsize=11)
ax.set_title('Validation Loss Across All Stages', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom middle: Stage 2 component losses (tau1, tau2, f1)
ax = axes[1, 1]
if 'stage2' in all_records and all_records['stage2'] is not None:
    record = all_records['stage2']

    # Check for component validation losses
    component_keys = [k for k in record.keys() if 'Val' in k and k != 'Val_Loss']

    if component_keys:
        epochs = range(1, len(record[component_keys[0]]) + 1)

        for key in component_keys:
            values = record[key]
            ax.plot(epochs, values, linewidth=2, label=key, alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Component Loss', fontsize=11)
        ax.set_title('Stage 2: Parameter Component Losses', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        print(f"\nStage 2 Component Losses:")
        for key in component_keys:
            values = record[key]
            print(f"  {key}: final={values[-1]:.6f}, min={min(values):.6f}")
    else:
        ax.text(0.5, 0.5, 'No component losses available',
               ha='center', va='center', transform=ax.transAxes)
        print(f"\nStage 2: No component losses found")
        print(f"  Available keys: {list(record.keys())}")
else:
    ax.text(0.5, 0.5, 'Stage 2 not found',
           ha='center', va='center', transform=ax.transAxes)

# Bottom right: Loss statistics table
ax = axes[1, 2]
ax.axis('off')

# Create table data
table_data = []
table_data.append(['Stage', 'Epochs', 'Final Val', 'Best Val', 'Overfit?'])

for stage, stage_name in zip(stages, ['S1', 'S2', 'S3']):
    if stage in all_records and all_records[stage] is not None:
        record = all_records[stage]
        if 'Train_Loss' in record and 'Val_Loss' in record:
            train_loss = record['Train_Loss']
            val_loss = record['Val_Loss']

            n_epochs = len(train_loss)
            final_val = val_loss[-1]
            best_val = min(val_loss)

            # Check overfitting
            overfit = "Yes" if train_loss[-1] < val_loss[-1] * 0.95 else "No"

            table_data.append([
                stage_name,
                f'{n_epochs}',
                f'{final_val:.4f}',
                f'{best_val:.4f}',
                overfit
            ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Training Summary', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save figure
save_path = os.path.join(OUTPUT_DIR, 'loss_analysis.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n{'=' * 80}")
print(f"Loss analysis plot saved to: {save_path}")
print("=" * 80)

# Show plot
plt.show()

# Additional detailed analysis
print(f"\n{'=' * 80}")
print("DETAILED ANALYSIS")
print("=" * 80)

# Check for dataset info
if 'data_info' in checkpoint:
    print("\nDataset Information:")
    for key, value in checkpoint['data_info'].items():
        print(f"  {key}: {value}")

# Check hyperparameters
if 'hyperparameters' in checkpoint:
    print("\nModel Hyperparameters:")
    for key, value in checkpoint['hyperparameters'].items():
        print(f"  {key}: {value}")

# Analyze if there are issues with tau1 learning
if 'stage2' in all_records and all_records['stage2'] is not None:
    record = all_records['stage2']

    # Look for tau-specific losses
    tau_keys = [k for k in record.keys() if 'tau' in k.lower() or 'Tau' in k]
    if tau_keys:
        print("\nTau-related losses in Stage 2:")
        for key in tau_keys:
            if isinstance(record[key], list):
                values = record[key]
                print(f"  {key}:")
                print(f"    Initial: {values[0]:.6f}")
                print(f"    Final: {values[-1]:.6f}")
                print(f"    Min: {min(values):.6f}")
                print(f"    Reduction: {(values[0] - values[-1])/values[0]*100:.1f}%")

print("\n" + "=" * 80)
