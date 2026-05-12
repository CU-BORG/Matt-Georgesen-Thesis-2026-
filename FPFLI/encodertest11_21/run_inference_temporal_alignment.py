#!/usr/bin/env python3
"""
Temporal Alignment Sensitivity Analysis

Run inference on decay bin 50 trained model against multiple test datasets
with different IRF timing offsets to assess temporal alignment sensitivity.

Tests 7 datasets:
  - bin 50: Self-inference baseline (trained on this)
  - bins 49, 51: Small offset (±1 bin = ±0.039 ns)
  - bins 45, 55: Medium offset (±5 bins = ±0.195 ns)
  - bins 35, 65: Large offset (±15 bins = ±0.585 ns)

@author: mg
"""

import sys, os, time
import numpy as np
import torch
import h5py
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = r'C:\Users\mcg11923\Thesis\training_logs_decay_bin50_20260220_150855\model_final_decay_bin50.pth'
OUT_DIR = r'C:\Users\mcg11923\Thesis\training_logs_decay_bin50_20260220_150855\temporal_sensitivity_analysis'

# Datasets to test - order by bin number for plotting
DATASETS = {
    'bin35': {
        'path': r'E:\decay_start_bin35_test',
        'irf_bin': 35,
        'offset_bins': -15,
        'offset_ns': -0.585
    },
    'bin45': {
        'path': r'E:\decay_start_bin45_test',
        'irf_bin': 45,
        'offset_bins': -5,
        'offset_ns': -0.195
    },
    'bin49': {
        'path': r'E:\decay_start_bin49_test',
        'irf_bin': 49,
        'offset_bins': -1,
        'offset_ns': -0.039
    },
    'bin50': {
        'path': r'E:\decay_start_bin50_train',
        'irf_bin': 50,
        'offset_bins': 0,
        'offset_ns': 0.0
    },
    'bin51': {
        'path': r'E:\decay_start_bin51_test',
        'irf_bin': 51,
        'offset_bins': 1,
        'offset_ns': 0.039
    },
    'bin55': {
        'path': r'E:\decay_start_bin55_test',
        'irf_bin': 55,
        'offset_bins': 5,
        'offset_ns': 0.195
    },
    'bin65': {
        'path': r'E:\decay_start_bin65_test',
        'irf_bin': 65,
        'offset_bins': 15,
        'offset_ns': 0.585
    }
}

N_IMAGES = 5  # Use first 5 images from each dataset
SAMPLE_STEP = 1  # Use every pixel

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
print("\n" + "=" * 80)
print(" TEMPORAL ALIGNMENT SENSITIVITY ANALYSIS")
print(" Model: decay_bin50_20260220_150855")
print("=" * 80 + "\n")

print(f"Loading model from:\n  {MODEL_PATH}\n")
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
hp = ckpt['hyperparameters']

model = LLE_BiExp_Autoencoder(
    dim=hp['dim'], depth=hp['depth'],
    kernel_size=hp['kernel_size'], patch_size=hp['patch_size'],
    signal_length=hp['signal_length'], latent_dim=hp['latent_dim']
)
model.load_state_dict(ckpt['model'])
model.eval()

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    model.cuda()
    print("Running on GPU\n")
else:
    print("Running on CPU\n")

# =============================================================================
# INFERENCE FUNCTION
# =============================================================================
def run_inference_on_dataset(data_dir, dataset_name, irf_bin):
    """Run inference on first N_IMAGES from a dataset."""

    print(f"\n{'=' * 80}")
    print(f" {dataset_name.upper()}: IRF bin {irf_bin} (offset {DATASETS[dataset_name]['offset_bins']:+d} bins)")
    print(f" Path: {data_dir}")
    print(f"{'=' * 80}")

    # Get .mat files
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])[:N_IMAGES]

    if len(mat_files) == 0:
        print(f"  ERROR: No .mat files found!")
        return None

    print(f"  Processing {len(mat_files)} images...\n")

    # Accumulators for all pixels across all images
    all_gt_tau1, all_pred_tau1 = [], []
    all_gt_tau2, all_pred_tau2 = [], []
    all_gt_f1, all_pred_f1 = [], []

    for file_idx, mat_name in enumerate(mat_files):
        mat_path = os.path.join(data_dir, mat_name)
        print(f"  [{file_idx+1}/{len(mat_files)}] {mat_name}...", end=' ')

        # Load data
        with h5py.File(mat_path, 'r') as f:
            Hist = np.array(f['Hist'])  # [256, 256, 256] = [T, H, W]
            tau_gt = np.array(f['tau_gt_components'])  # [2, H, W]
            f_gt = np.array(f['f_gt_components'])  # [2, H, W]

        T, H, W = Hist.shape

        # Subsample pixels
        ys = range(0, H, SAMPLE_STEP)
        xs = range(0, W, SAMPLE_STEP)

        # Build IRF from bright-corner average
        irf = Hist[:, :10, :10].mean(axis=(1, 2))
        irf = irf / (irf.max() + 1e-8)

        # Intensity per pixel
        Int_full = Hist.sum(axis=0).astype(np.float32)
        Int_max = Int_full.max()
        if Int_max > 0:
            Int_full /= Int_max
        INTENSITY_THRESHOLD = 0.05

        # Collect valid pixels
        decays_list, gt_tau1_list, gt_tau2_list, gt_f1_list = [], [], [], []

        for y in ys:
            for x in xs:
                if Int_full[y, x] < INTENSITY_THRESHOLD:
                    continue
                decays_list.append(Hist[:, y, x])
                gt_tau1_list.append(tau_gt[0, y, x])
                gt_tau2_list.append(tau_gt[1, y, x])
                gt_f1_list.append(f_gt[0, y, x])

        n_valid = len(decays_list)
        decays = np.array(decays_list, dtype=np.float32)
        gt_tau1 = np.array(gt_tau1_list, dtype=np.float32)
        gt_tau2 = np.array(gt_tau2_list, dtype=np.float32)
        gt_f1 = np.array(gt_f1_list, dtype=np.float32)

        # Normalize decays
        maxvals = decays.max(axis=1, keepdims=True)
        maxvals[maxvals == 0] = 1.0
        decays = decays / maxvals

        # Tile IRF
        irfs = np.tile(irf, (n_valid, 1)).astype(np.float32)

        # Convert to tensors
        decay_tensor = torch.from_numpy(decays).unsqueeze(1)
        irf_tensor = torch.from_numpy(irfs).unsqueeze(1)

        # Run inference in batches
        BATCH = 4096
        pred_tau1_list, pred_tau2_list, pred_f1_list = [], [], []

        with torch.no_grad():
            for start in range(0, n_valid, BATCH):
                end = min(start + BATCH, n_valid)
                d = decay_tensor[start:end]
                i = irf_tensor[start:end]
                if USE_GPU:
                    d, i = d.cuda(), i.cuda()
                params = model.extract_parameters(d, i)
                pred_tau1_list.append(params['tau1'].cpu().numpy())
                pred_tau2_list.append(params['tau2'].cpu().numpy())
                pred_f1_list.append(params['f1'].cpu().numpy())

        pred_tau1 = np.concatenate(pred_tau1_list)
        pred_tau2 = np.concatenate(pred_tau2_list)
        pred_f1 = np.concatenate(pred_f1_list)

        # Accumulate
        all_gt_tau1.append(gt_tau1)
        all_pred_tau1.append(pred_tau1)
        all_gt_tau2.append(gt_tau2)
        all_pred_tau2.append(pred_tau2)
        all_gt_f1.append(gt_f1)
        all_pred_f1.append(pred_f1)

        print(f"{n_valid} pixels")

    # Concatenate all images
    all_gt_tau1 = np.concatenate(all_gt_tau1)
    all_pred_tau1 = np.concatenate(all_pred_tau1)
    all_gt_tau2 = np.concatenate(all_gt_tau2)
    all_pred_tau2 = np.concatenate(all_pred_tau2)
    all_gt_f1 = np.concatenate(all_gt_f1)
    all_pred_f1 = np.concatenate(all_pred_f1)

    # Calculate metrics
    tau1_mae = np.abs(all_pred_tau1 - all_gt_tau1).mean()
    tau2_mae = np.abs(all_pred_tau2 - all_gt_tau2).mean()
    f1_mae = np.abs(all_pred_f1 - all_gt_f1).mean()

    tau1_rmse = np.sqrt(((all_pred_tau1 - all_gt_tau1) ** 2).mean())
    tau2_rmse = np.sqrt(((all_pred_tau2 - all_gt_tau2) ** 2).mean())
    f1_rmse = np.sqrt(((all_pred_f1 - all_gt_f1) ** 2).mean())

    print(f"\n  Results ({len(all_gt_tau1):,} total pixels):")
    print(f"    tau1: MAE={tau1_mae:.6f}, RMSE={tau1_rmse:.6f}")
    print(f"    tau2: MAE={tau2_mae:.6f}, RMSE={tau2_rmse:.6f}")
    print(f"    f1:   MAE={f1_mae:.6f}, RMSE={f1_rmse:.6f}")

    return {
        'tau1_mae': float(tau1_mae),
        'tau2_mae': float(tau2_mae),
        'f1_mae': float(f1_mae),
        'tau1_rmse': float(tau1_rmse),
        'tau2_rmse': float(tau2_rmse),
        'f1_rmse': float(f1_rmse),
        'n_pixels': len(all_gt_tau1),
        'n_images': len(mat_files)
    }

# =============================================================================
# RUN INFERENCE ON ALL DATASETS
# =============================================================================
results = {}

for dataset_name in ['bin35', 'bin45', 'bin49', 'bin50', 'bin51', 'bin55', 'bin65']:
    dataset_info = DATASETS[dataset_name]
    data_dir = dataset_info['path']

    if not os.path.exists(data_dir):
        print(f"\nWARNING: Dataset not found: {data_dir}")
        print(f"  Skipping {dataset_name}...")
        continue

    metrics = run_inference_on_dataset(data_dir, dataset_name, dataset_info['irf_bin'])

    if metrics is not None:
        results[dataset_name] = {
            'irf_bin': dataset_info['irf_bin'],
            'offset_bins': dataset_info['offset_bins'],
            'offset_ns': dataset_info['offset_ns'],
            'metrics': metrics
        }

# =============================================================================
# SAVE RESULTS
# =============================================================================
if len(results) > 0:
    # Save JSON
    json_path = os.path.join(OUT_DIR, 'temporal_alignment_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {json_path}")

    # Save text summary
    summary_path = os.path.join(OUT_DIR, 'temporal_alignment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TEMPORAL ALIGNMENT SENSITIVITY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Model: decay_bin50_20260220_150855\n")
        f.write(f"Trained on: IRF bin 50 (t = 1.95 ns)\n")
        f.write(f"Bin width: 0.039 ns\n\n")
        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Dataset':<10} {'IRF Bin':<10} {'Offset':<12} {'tau1 MAE':<12} {'tau2 MAE':<12} {'f1 MAE':<12}\n")
        f.write("-" * 80 + "\n")

        for dataset_name in ['bin35', 'bin45', 'bin49', 'bin50', 'bin51', 'bin55', 'bin65']:
            if dataset_name in results:
                r = results[dataset_name]
                offset_str = f"{r['offset_bins']:+d} bins"
                f.write(f"{dataset_name:<10} "
                       f"{r['irf_bin']:<10} "
                       f"{offset_str:<12} "
                       f"{r['metrics']['tau1_mae']:<12.6f} "
                       f"{r['metrics']['tau2_mae']:<12.6f} "
                       f"{r['metrics']['f1_mae']:<12.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("=" * 80 + "\n")
        f.write("bin 50: Baseline (trained on this data)\n")
        f.write("bins 49, 51: Small temporal offset (±1 bin = ±0.039 ns)\n")
        f.write("bins 45, 55: Medium temporal offset (±5 bins = ±0.195 ns)\n")
        f.write("bins 35, 65: Large temporal offset (±15 bins = ±0.585 ns)\n\n")
        f.write("Expected: MAE increases with larger temporal misalignment\n")

    print(f"Summary saved to: {summary_path}")
    print("=" * 80 + "\n")

    # Print summary to console
    print("\nSUMMARY TABLE:")
    print("-" * 80)
    print(f"{'Dataset':<10} {'IRF Bin':<10} {'Offset':<12} {'tau1 MAE':<12} {'tau2 MAE':<12} {'f1 MAE':<12}")
    print("-" * 80)
    for dataset_name in ['bin35', 'bin45', 'bin49', 'bin50', 'bin51', 'bin55', 'bin65']:
        if dataset_name in results:
            r = results[dataset_name]
            offset_str = f"{r['offset_bins']:+d} bins"
            print(f"{dataset_name:<10} "
                  f"{r['irf_bin']:<10} "
                  f"{offset_str:<12} "
                  f"{r['metrics']['tau1_mae']:<12.6f} "
                  f"{r['metrics']['tau2_mae']:<12.6f} "
                  f"{r['metrics']['f1_mae']:<12.6f}")
    print("-" * 80 + "\n")

    print("Next step: Run plot_temporal_alignment_sensitivity.py to visualize results\n")
else:
    print("\nERROR: No results generated. Check dataset paths.")
