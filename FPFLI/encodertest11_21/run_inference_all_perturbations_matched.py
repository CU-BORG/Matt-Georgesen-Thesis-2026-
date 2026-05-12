#!/usr/bin/env python3
"""
Run inference on parameter-matched perturbations.
All perturbation samples use IDENTICAL ground truth parameters as baseline_normal.
Enables pixel-wise paired comparison and error difference analysis.

@author: mg
"""

import sys, os, time
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
MODEL_PATH = (r'C:\Users\mcg11923\Thesis'
              r'\training_logs_decay_bin62_12.5ns_20260306_112041'
              r'\model_final_decay_bin62_12.5ns.pth')

DATA_BASE  = r'E:\perturbations_bin62_12.5ns_matched'

OUT_BASE   = (r'C:\Users\mcg11923\Thesis'
              r'\training_logs_decay_bin62_12.5ns_20260306_112041'
              r'\inference_results_perturbations_matched')

GROUPS = [
    'baseline_normal',
    'perturb_delayed_decay',
    'perturb_tri_exponential',
    'perturb_flat_background',
    'perturb_50pct_photons',
    'perturb_10pct_photons',
    'perturb_1pct_photons',
]

N_IMAGES    = 10
SAMPLE_STEP = 1
INTENSITY_THRESHOLD = 0.05
BATCH       = 4096

# ---------------------------------------------------------------------------
# load model once
# ---------------------------------------------------------------------------
print("Loading model ...")
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
hp   = ckpt['hyperparameters']
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
    print("  Running on GPU\n")
else:
    print("  Running on CPU\n")

# ---------------------------------------------------------------------------
# inference helpers  (extracted from run_inference.py)
# ---------------------------------------------------------------------------
def run_one_image(mat_path, forced_valid_coords=None):
    """Load one .mat, run model, return (gt, pred) flat arrays + valid_coords.

    Args:
        mat_path: Path to .mat file
        forced_valid_coords: Optional list of (y,x) coords to use instead of threshold filtering
    """
    with h5py.File(mat_path, 'r') as f:
        Hist   = np.array(f['Hist'])                  # [T, H, W]
        tau_gt = np.array(f['tau_gt_components'])     # [2, H, W]
        f_gt   = np.array(f['f_gt_components'])       # [2, H, W]

    T, H, W = Hist.shape

    
    irf = Hist[:, :10, :10].mean(axis=(1, 2))
    irf = irf / (irf.max() + 1e-8)

    # collect valid pixels (use forced coords if provided)
    decays_list, gt_tau1_list, gt_tau2_list, gt_f1_list = [], [], [], []

    if forced_valid_coords is not None:
        valid_coords = forced_valid_coords
    else:
        # Compute from intensity threshold (original behavior)
        Int_full = Hist.sum(axis=0).astype(np.float32)
        Int_max  = Int_full.max()
        if Int_max > 0:
            Int_full /= Int_max

        valid_coords = []
        for y in range(0, H, SAMPLE_STEP):
            for x in range(0, W, SAMPLE_STEP):
                if Int_full[y, x] >= INTENSITY_THRESHOLD:
                    valid_coords.append((y, x))

    n_valid = len(valid_coords)
    if n_valid == 0:
        return None   # skip empty images

    # Collect data from valid pixels
    for (y, x) in valid_coords:
        decays_list.append(Hist[:, y, x])
        gt_tau1_list.append(tau_gt[0, y, x])
        gt_tau2_list.append(tau_gt[1, y, x])
        gt_f1_list.append(f_gt[0, y, x])

    decays  = np.array(decays_list,  dtype=np.float32)
    gt_tau1 = np.array(gt_tau1_list, dtype=np.float32)
    gt_tau2 = np.array(gt_tau2_list, dtype=np.float32)
    gt_f1   = np.array(gt_f1_list,   dtype=np.float32)

    # normalise decays
    maxvals = decays.max(axis=1, keepdims=True)
    maxvals[maxvals == 0] = 1.0
    decays /= maxvals

    irfs = np.tile(irf, (n_valid, 1)).astype(np.float32)

    decay_tensor = torch.from_numpy(decays).unsqueeze(1)
    irf_tensor   = torch.from_numpy(irfs).unsqueeze(1)

    # batched inference
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
    pred_f1   = np.concatenate(pred_f1_list)

    return {
        'gt_tau1': gt_tau1, 'gt_tau2': gt_tau2, 'gt_f1': gt_f1,
        'pred_tau1': pred_tau1, 'pred_tau2': pred_tau2, 'pred_f1': pred_f1,
        'valid_coords': valid_coords, 'H': H, 'W': W
    }


def compute_stats(pred, gt):
    mae  = np.abs(pred - gt).mean()
    rmse = np.sqrt(((pred - gt)**2).mean())
    return {'mae': mae, 'rmse': rmse}


# ---------------------------------------------------------------------------
# Load baseline intensity maps for consistent pixel filtering
# ---------------------------------------------------------------------------
print("Loading baseline intensity maps for consistent pixel filtering...")
baseline_dir = os.path.join(DATA_BASE, 'baseline_normal')
baseline_files = sorted([fn for fn in os.listdir(baseline_dir) if fn.endswith('.mat')])[:N_IMAGES]

baseline_intensity_maps = []
baseline_valid_coords_list = []

for baseline_file in baseline_files:
    with h5py.File(os.path.join(baseline_dir, baseline_file), 'r') as f:
        Hist = np.array(f['Hist'])  # [T, H, W]

    T, H, W = Hist.shape
    Int_full = Hist.sum(axis=0).astype(np.float32)
    Int_max = Int_full.max()
    if Int_max > 0:
        Int_full /= Int_max

    # Store valid pixel coordinates based on baseline intensity
    valid_coords = []
    for y in range(0, H, SAMPLE_STEP):
        for x in range(0, W, SAMPLE_STEP):
            if Int_full[y, x] >= INTENSITY_THRESHOLD:
                valid_coords.append((y, x))

    baseline_intensity_maps.append(Int_full)
    baseline_valid_coords_list.append(valid_coords)

print(f"  Loaded {len(baseline_intensity_maps)} baseline intensity maps\n")

# ---------------------------------------------------------------------------
# main loop - save predictions to .mat files for paired analysis
# ---------------------------------------------------------------------------
os.makedirs(OUT_BASE, exist_ok=True)
summary = {}
total_t0 = time.time()

for group_name in GROUPS:
    data_dir = os.path.join(DATA_BASE, group_name)
    out_dir  = os.path.join(OUT_BASE, group_name)
    os.makedirs(out_dir, exist_ok=True)

    mat_files = sorted([fn for fn in os.listdir(data_dir) if fn.endswith('.mat')])[:N_IMAGES]

    print(f"\n{'='*70}")
    print(f"  GROUP: {group_name}  ({len(mat_files)} images)")
    print(f"{'='*70}")

    # accumulators for MAE calculation
    mae_per_image = {'tau1': [], 'tau2': [], 'f1': []}

    for file_idx, mat_name in enumerate(mat_files):
        t0 = time.time()
        mat_path = os.path.join(data_dir, mat_name)
        print(f"  [{file_idx+1}/{len(mat_files)}] {mat_name} ... ", end='', flush=True)

        # Use baseline valid coords for all groups to ensure pixel-wise matching
        forced_coords = baseline_valid_coords_list[file_idx]
        res = run_one_image(mat_path, forced_valid_coords=forced_coords)
        if res is None:
            print("(no valid pixels — skipped)")
            continue

        n_valid = len(res['valid_coords'])

        # Compute stats
        st1 = compute_stats(res['pred_tau1'], res['gt_tau1'])
        st2 = compute_stats(res['pred_tau2'], res['gt_tau2'])
        sf  = compute_stats(res['pred_f1'],   res['gt_f1'])

        mae_per_image['tau1'].append(st1['mae'])
        mae_per_image['tau2'].append(st2['mae'])
        mae_per_image['f1'].append(sf['mae'])

        # Save predictions and ground truth to .mat file for analysis
        sample_num = file_idx + 1
        output_filename = f'Sample_{sample_num:03d}_{group_name}_results.mat'
        output_path = os.path.join(out_dir, output_filename)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('tau1_pred', data=res['pred_tau1'])
            f.create_dataset('tau2_pred', data=res['pred_tau2'])
            f.create_dataset('f1_pred', data=res['pred_f1'])
            f.create_dataset('tau1_gt', data=res['gt_tau1'])
            f.create_dataset('tau2_gt', data=res['gt_tau2'])
            f.create_dataset('f1_gt', data=res['gt_f1'])

        print(f"tau1={st1['mae']:.4f}, tau2={st2['mae']:.4f}, f1={sf['mae']:.4f} ({time.time()-t0:.1f}s)")

    # Record mean MAE for summary
    if mae_per_image['tau1']:
        summary[group_name] = {
            'tau1': np.mean(mae_per_image['tau1']),
            'tau2': np.mean(mae_per_image['tau2']),
            'f1':   np.mean(mae_per_image['f1']),
        }

        print(f"\n  {group_name} MEAN MAE:  "
              f"tau1={summary[group_name]['tau1']:.4f}  "
              f"tau2={summary[group_name]['tau2']:.4f}  "
              f"f1={summary[group_name]['f1']:.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  PARAMETER-MATCHED PERTURBATIONS SUMMARY  (mean MAE)")
print(f"{'='*70}")
print(f"  {'Group':<30s}  {'tau1':>8s}  {'tau2':>8s}  {'f1':>8s}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
for g in GROUPS:
    if g in summary:
        s = summary[g]
        print(f"  {g:<30s}  {s['tau1']:8.4f}  {s['tau2']:8.4f}  {s['f1']:8.4f}")

total_elapsed = time.time() - total_t0
print(f"\nTotal runtime: {total_elapsed:.0f} s  ({total_elapsed/60:.1f} min)")
print(f"Results saved to: {OUT_BASE}")
print("\nNext step: Run analyze_perturbation_error_differences.py for paired comparison")
