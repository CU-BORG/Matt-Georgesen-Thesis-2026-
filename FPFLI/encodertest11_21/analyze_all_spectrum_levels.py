#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Analysis of All 10 Spectrum Level Models

Analyzes all 10 trained models across the spectrum progression and creates
comparative visualizations.

@author: mg
"""

# Fix OpenMP library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder
from analyze_trained_model import (
    plot_training_curves,
    analyze_architecture,
    count_parameters,
    test_on_images
)

# Configuration
STD_LEVELS = 10
TAU_1_STD_ARRAY = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05]
TAU_2_STD_ARRAY = [0.10, 0.30, 0.50, 0.80, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50]

BASE_DATA_DIR = r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite'
BASE_MODEL_DIR = r'C:\Users\mcg11923\Thesis'

NUM_TEST_SAMPLES = 3
SIGNAL_THRESHOLD = 50


def find_latest_model_dir(level_idx):
    """Find the most recent training directory for a given level"""
    pattern = os.path.join(BASE_MODEL_DIR, f'training_logs_spectrum_level_{level_idx:02d}_*')
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    # Return most recent (sorted by name which includes timestamp)
    return sorted(dirs)[-1]


def create_comparative_plots(all_results, output_dir):
    """Create comparative visualizations across all levels"""
    print("\n" + "="*80)
    print("Creating Comparative Visualizations")
    print("="*80)

    levels = []
    tau1_stds = []
    tau2_stds = []

    # Extract data for plotting
    stage1_final_losses = []
    stage2_final_losses = []
    stage3_final_losses = []

    tau1_maes = []
    tau2_maes = []
    f1_maes = []

    for result in all_results:
        levels.append(result['level'])
        tau1_stds.append(result['tau1_std'])
        tau2_stds.append(result['tau2_std'])

        # Get final losses from each stage
        stage1_final_losses.append(result['checkpoint']['all_records']['stage1']['Val_Loss'][-1])
        stage2_final_losses.append(result['checkpoint']['all_records']['stage2']['Val_Loss'][-1])
        stage3_final_losses.append(result['checkpoint']['all_records']['stage3']['Val_Loss'][-1])

        # Get MAE for parameters
        tau1_maes.append(result['test_metrics']['tau1_mae'])
        tau2_maes.append(result['test_metrics']['tau2_mae'])
        f1_maes.append(result['test_metrics']['f1_mae'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Standard deviations vs level
    ax = axes[0, 0]
    ax.plot(levels, tau1_stds, 'o-', label='Tau1 Std', linewidth=2, markersize=8)
    ax.plot(levels, tau2_stds, 's-', label='Tau2 Std', linewidth=2, markersize=8)
    ax.set_xlabel('Spectrum Level', fontsize=12)
    ax.set_ylabel('Standard Deviation (ns)', fontsize=12)
    ax.set_title('Spectral Variation Progression', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(levels)

    # Plot 2: Final validation losses across stages
    ax = axes[0, 1]
    ax.plot(levels, stage1_final_losses, 'o-', label='Stage 1 (Reconstruction)', linewidth=2)
    ax.plot(levels, stage2_final_losses, 's-', label='Stage 2 (Parameters)', linewidth=2)
    ax.plot(levels, stage3_final_losses, '^-', label='Stage 3 (Joint)', linewidth=2)
    ax.set_xlabel('Spectrum Level', fontsize=12)
    ax.set_ylabel('Final Validation Loss', fontsize=12)
    ax.set_title('Training Performance vs Spectral Variation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(levels)
    ax.set_yscale('log')

    # Plot 3: Parameter MAE vs spectral variation
    ax = axes[0, 2]
    ax.plot(levels, tau1_maes, 'o-', label='Tau1 MAE', linewidth=2, markersize=8)
    ax.plot(levels, tau2_maes, 's-', label='Tau2 MAE', linewidth=2, markersize=8)
    ax.plot(levels, f1_maes, '^-', label='F1 MAE', linewidth=2, markersize=8)
    ax.set_xlabel('Spectrum Level', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Test Error vs Spectral Variation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(levels)

    # Plot 4: Tau1 MAE vs Tau1 Std
    ax = axes[1, 0]
    ax.plot(tau1_stds, tau1_maes, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Tau1 Standard Deviation (ns)', fontsize=12)
    ax.set_ylabel('Tau1 MAE', fontsize=12)
    ax.set_title('Tau1 Prediction Error vs Variation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 5: Tau2 MAE vs Tau2 Std
    ax = axes[1, 1]
    ax.plot(tau2_stds, tau2_maes, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Tau2 Standard Deviation (ns)', fontsize=12)
    ax.set_ylabel('Tau2 MAE', fontsize=12)
    ax.set_title('Tau2 Prediction Error vs Variation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = "Summary Statistics\n\n"
    summary_text += f"Levels analyzed: {len(levels)}\n\n"
    summary_text += "Stage 3 Final Loss:\n"
    summary_text += f"  Best:  Level {levels[np.argmin(stage3_final_losses)]} ({min(stage3_final_losses):.6f})\n"
    summary_text += f"  Worst: Level {levels[np.argmax(stage3_final_losses)]} ({max(stage3_final_losses):.6f})\n\n"
    summary_text += "Tau1 MAE:\n"
    summary_text += f"  Best:  Level {levels[np.argmin(tau1_maes)]} ({min(tau1_maes):.6f})\n"
    summary_text += f"  Worst: Level {levels[np.argmax(tau1_maes)]} ({max(tau1_maes):.6f})\n\n"
    summary_text += "Tau2 MAE:\n"
    summary_text += f"  Best:  Level {levels[np.argmin(tau2_maes)]} ({min(tau2_maes):.6f})\n"
    summary_text += f"  Worst: Level {levels[np.argmax(tau2_maes)]} ({max(tau2_maes):.6f})"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Spectrum Progression Analysis: All 10 Levels', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'spectrum_progression_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved comparative plot to: {save_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print(" MULTI-LEVEL ANALYSIS - ALL SPECTRUM LEVELS")
    print("="*80)
    print(f"\nAnalyzing {STD_LEVELS} models across spectrum progression")
    print("="*80 + "\n")

    all_results = []

    # Loop through all levels
    for level_idx in range(1, STD_LEVELS + 1):
        tau1_std = TAU_1_STD_ARRAY[level_idx - 1]
        tau2_std = TAU_2_STD_ARRAY[level_idx - 1]

        print("\n" + "="*80)
        print(f"ANALYZING LEVEL {level_idx}/{STD_LEVELS}")
        print(f"  Tau1_std = {tau1_std:.2f} ns, Tau2_std = {tau2_std:.2f} ns")
        print("="*80)

        # Find model directory
        model_dir = find_latest_model_dir(level_idx)

        if model_dir is None or not os.path.exists(model_dir):
            print(f"WARNING: No model directory found for level {level_idx}, skipping...")
            continue

        print(f"Model directory: {model_dir}")

        # Find model file
        model_path = os.path.join(model_dir, f'model_final_spectrum_level_{level_idx:02d}.pth')

        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found: {model_path}, skipping...")
            continue

        # Data directory
        data_dir = os.path.join(BASE_DATA_DIR, f'spectrum_level_{level_idx:02d}_std_{tau1_std:.2f}_{tau2_std:.2f}')

        if not os.path.exists(data_dir):
            print(f"WARNING: Data directory not found: {data_dir}, skipping...")
            continue

        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        model = LLE_BiExp_Autoencoder(
            dim=checkpoint['hyperparameters']['dim'],
            depth=checkpoint['hyperparameters']['depth'],
            kernel_size=checkpoint['hyperparameters']['kernel_size'],
            patch_size=checkpoint['hyperparameters']['patch_size']
        )
        model.load_state_dict(checkpoint['model'])
        model.eval()

        print("[OK] Model loaded successfully")

        # Run analysis tasks
        plot_training_curves(checkpoint, model_dir)

        if level_idx == 1:
            analyze_architecture(model, model_dir)
            count_parameters(model, model_dir)

        test_on_images(model, data_dir, model_dir, NUM_TEST_SAMPLES)

        # Extract key metrics for comparison
        # Read test summary to get MAE values
        test_summary_path = os.path.join(model_dir, 'test_summary.txt')
        tau1_mae = tau2_mae = f1_mae = None

        if os.path.exists(test_summary_path):
            with open(test_summary_path, 'r') as f:
                content = f.read()
                # Parse MAE values from first sample
                lines = content.split('\n')
                for line in lines:
                    if 'tau1 - MAE:' in line:
                        tau1_mae = float(line.split('MAE:')[1].split(',')[0].strip())
                    elif 'tau2 - MAE:' in line:
                        tau2_mae = float(line.split('MAE:')[1].split(',')[0].strip())
                    elif 'f1   - MAE:' in line:
                        f1_mae = float(line.split('MAE:')[1].split(',')[0].strip())
                    if tau1_mae and tau2_mae and f1_mae:
                        break

        all_results.append({
            'level': level_idx,
            'tau1_std': tau1_std,
            'tau2_std': tau2_std,
            'model_dir': model_dir,
            'checkpoint': checkpoint,
            'test_metrics': {
                'tau1_mae': tau1_mae if tau1_mae else 0.0,
                'tau2_mae': tau2_mae if tau2_mae else 0.0,
                'f1_mae': f1_mae if f1_mae else 0.0
            }
        })

        print(f"\n[OK] Level {level_idx} analysis complete")

    # Create comparative visualizations
    if len(all_results) > 0:
        print("\n" + "="*80)
        print(f"Creating comparative analysis for {len(all_results)} levels")
        print("="*80)

        output_dir = os.path.join(BASE_MODEL_DIR, 'spectrum_progression_analysis')
        os.makedirs(output_dir, exist_ok=True)

        create_comparative_plots(all_results, output_dir)

        # Save summary report
        report_path = os.path.join(output_dir, 'spectrum_progression_summary.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SPECTRUM PROGRESSION ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analyzed {len(all_results)} spectrum levels\n\n")

            f.write("Level | Tau1_std | Tau2_std | Stage3_Loss | Tau1_MAE | Tau2_MAE | F1_MAE\n")
            f.write("-"*80 + "\n")
            for r in all_results:
                stage3_loss = r['checkpoint']['all_records']['stage3']['Val_Loss'][-1]
                f.write(f"  {r['level']:2d}  |   {r['tau1_std']:5.2f}  |   {r['tau2_std']:5.2f}  |   {stage3_loss:9.6f} | {r['test_metrics']['tau1_mae']:8.6f} | {r['test_metrics']['tau2_mae']:8.6f} | {r['test_metrics']['f1_mae']:6.4f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Key Findings:\n")
            f.write("="*80 + "\n\n")

            # Find trends
            stage3_losses = [r['checkpoint']['all_records']['stage3']['Val_Loss'][-1] for r in all_results]
            tau1_maes = [r['test_metrics']['tau1_mae'] for r in all_results]
            tau2_maes = [r['test_metrics']['tau2_mae'] for r in all_results]

            best_level = all_results[np.argmin(stage3_losses)]['level']
            worst_level = all_results[np.argmax(stage3_losses)]['level']

            f.write(f"Best performing level (Stage 3 Loss): Level {best_level}\n")
            f.write(f"Worst performing level (Stage 3 Loss): Level {worst_level}\n\n")

            f.write(f"Tau1 MAE range: [{min(tau1_maes):.6f}, {max(tau1_maes):.6f}]\n")
            f.write(f"Tau2 MAE range: [{min(tau2_maes):.6f}, {max(tau2_maes):.6f}]\n\n")

            f.write("Conclusion:\n")
            if stage3_losses[-1] > stage3_losses[0]:
                f.write("  - Performance degrades with increasing spectral overlap\n")
            else:
                f.write("  - Performance improves or remains stable with increasing spectral overlap\n")

        print(f"[OK] Saved summary report to: {report_path}")

        print("\n" + "="*80)
        print(" MULTI-LEVEL ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  - spectrum_progression_comparison.png")
        print("  - spectrum_progression_summary.txt")
        print()
    else:
        print("\nWARNING: No models were found or analyzed!")
        print("Please ensure models have been trained before running analysis.")
