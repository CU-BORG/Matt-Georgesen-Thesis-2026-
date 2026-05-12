#!/usr/bin/env python3
"""
Self-inference: Run the realdata_stats model on its own training data.
Tests how well the model performs on the data it was trained on.
Loads samples from realdata_stats, extracts tau1/tau2/f1 per-pixel,
compares to ground truth, and saves comparison plots.

@author: mg
"""

# Fix OpenMP library conflict - must be first
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys, time, copy
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LLE_BiExp_Autoencoder_11_21 import LLE_BiExp_Autoencoder

# ---------------------------------------------------------------------------
# PATHS - SELF-INFERENCE ON REALDATA_STATS
# ---------------------------------------------------------------------------
MODEL_PATH = r'C:\Users\mcg11923\Thesis\training_logs_realdata_stats_20260203_232837\model_final_realdata_stats.pth'
DATA_DIR   = r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation\generate_multiexp_spectrumless\generate_spectrum_lite\realdata_stats'
OUT_DIR    = r'C:\Users\mcg11923\Thesis\training_logs_realdata_stats_20260203_232837\inference_results_self'
N_IMAGES   = 10         # how many .mat files to run
SAMPLE_STEP = 1         # subsample pixels (1 = every pixel; 4 = every 4th)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------
print("=" * 70)
print("  SELF-INFERENCE: REALDATA_STATS MODEL")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Data : {DATA_DIR}")
print()

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
    print("Running on GPU")
else:
    print("Running on CPU")

# ---------------------------------------------------------------------------
# COLLECT .MAT FILES
# ---------------------------------------------------------------------------
mat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])[:N_IMAGES]
print(f"\nRunning {len(mat_files)} images from:\n  {DATA_DIR}\n")

# ---------------------------------------------------------------------------
# PER-IMAGE INFERENCE
# ---------------------------------------------------------------------------
all_errors = {k: [] for k in ('tau1', 'tau2', 'f1')}

# accumulators for the combined predicted-vs-gt scatter at the end
all_gt_tau1,   all_pred_tau1   = [], []
all_gt_tau2,   all_pred_tau2   = [], []
all_gt_f1,     all_pred_f1     = [], []

for file_idx, mat_name in enumerate(mat_files):
    t0 = time.time()
    mat_path = os.path.join(DATA_DIR, mat_name)
    print(f"--- [{file_idx+1}/{len(mat_files)}] {mat_name} ---")

    # load data
    with h5py.File(mat_path, 'r') as f:
        Hist = np.array(f['Hist'])                  # [256, 256, 256] = [T, H, W]
        tau_gt = np.array(f['tau_gt_components'])   # [2, H, W]
        f_gt   = np.array(f['f_gt_components'])     # [2, H, W]

    T, H, W = Hist.shape

    # subsample pixels
    ys = range(0, H, SAMPLE_STEP)
    xs = range(0, W, SAMPLE_STEP)
    n_pixels = len(ys) * len(xs)

    # build IRF from bright-corner average (same approach as training loader)
    irf = Hist[:, :10, :10].mean(axis=(1, 2))
    irf = irf / (irf.max() + 1e-8)

    # intensity per pixel — normalised to [0,1]
    Int_full = Hist.sum(axis=0).astype(np.float32)   # [H, W]
    Int_max  = Int_full.max()
    if Int_max > 0:
        Int_full /= Int_max
    INTENSITY_THRESHOLD = 0.05

    # collect only pixels that pass the threshold — skip the rest entirely
    decays_list, gt_tau1_list, gt_tau2_list, gt_f1_list = [], [], [], []
    valid_coords = []          # (y, x) for each kept pixel

    for y in ys:
        for x in xs:
            if Int_full[y, x] < INTENSITY_THRESHOLD:
                continue
            decays_list.append(Hist[:, y, x])
            gt_tau1_list.append(tau_gt[0, y, x])
            gt_tau2_list.append(tau_gt[1, y, x])
            gt_f1_list.append(f_gt[0, y, x])
            valid_coords.append((y, x))

    n_valid  = len(valid_coords)
    decays   = np.array(decays_list,  dtype=np.float32)   # [n_valid, T]
    gt_tau1  = np.array(gt_tau1_list, dtype=np.float32)
    gt_tau2  = np.array(gt_tau2_list, dtype=np.float32)
    gt_f1    = np.array(gt_f1_list,   dtype=np.float32)

    # normalise each decay to [0,1]
    maxvals = decays.max(axis=1, keepdims=True)
    maxvals[maxvals == 0] = 1.0
    decays = decays / maxvals

    # tile IRF
    irfs = np.tile(irf, (n_valid, 1)).astype(np.float32)

    # reshape -> [N, 1, 256]
    decay_tensor = torch.from_numpy(decays).unsqueeze(1)
    irf_tensor   = torch.from_numpy(irfs).unsqueeze(1)

    # run model in batches — only on valid pixels
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
    pred_f1   = np.concatenate(pred_f1_list)

    # mask is just "all True" for valid pixels (dark ones were never collected)
    mask = np.ones(n_valid, dtype=bool)

    # ---------------------------------------------------------------------------
    # stats  — returns dict so we can reuse on the plots
    # ---------------------------------------------------------------------------
    def stats(name, pred, gt, m):
        p, g = pred[m], gt[m]
        mae  = np.abs(p - g).mean()
        rmse = np.sqrt(((p - g)**2).mean())
        print(f"  {name:5s}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"gt=[{g.min():.3f},{g.max():.3f}]  pred=[{p.min():.3f},{p.max():.3f}]")
        return {'mae': mae, 'rmse': rmse}

    print(f"  valid pixels: {mask.sum()}/{mask.size}")
    stats_tau1 = stats('tau1', pred_tau1, gt_tau1, mask)
    stats_tau2 = stats('tau2', pred_tau2, gt_tau2, mask)
    stats_f1   = stats('f1',   pred_f1,   gt_f1,   mask)
    all_errors['tau1'].append(stats_tau1['mae'])
    all_errors['tau2'].append(stats_tau2['mae'])
    all_errors['f1'].append(stats_f1['mae'])

    # accumulate for combined scatter
    all_gt_tau1.append(gt_tau1);     all_pred_tau1.append(pred_tau1)
    all_gt_tau2.append(gt_tau2);     all_pred_tau2.append(pred_tau2)
    all_gt_f1.append(gt_f1);         all_pred_f1.append(pred_f1)

    # ---------------------------------------------------------------------------
    # scatter valid pixels back onto full-size NaN images
    # ---------------------------------------------------------------------------
    nH = len(ys)
    nW = len(xs)

    def to_img(arr):
        """Place 1-D valid-pixel array onto [nH, nW] grid; unvisited = NaN."""
        img = np.full((nH, nW), np.nan, dtype=np.float32)
        for k, (y, x) in enumerate(valid_coords):
            img[y, x] = arr[k]
        return img

    gt_tau1_img   = to_img(gt_tau1)
    gt_tau2_img   = to_img(gt_tau2)
    gt_f1_img     = to_img(gt_f1)
    pred_tau1_img = to_img(pred_tau1)
    pred_tau2_img = to_img(pred_tau2)
    pred_f1_img   = to_img(pred_f1)
    err_tau1_img  = to_img(pred_tau1 - gt_tau1)
    err_tau2_img  = to_img(pred_tau2 - gt_tau2)
    err_f1_img    = to_img(pred_f1   - gt_f1)

    # ---------------------------------------------------------------------------
    # plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f'{mat_name}  (realdata_stats, self-inference)', fontsize=14, y=0.98)

    col_titles = ['Ground Truth', 'Predicted', 'Error (pred - gt)']
    rows = [
        (r'$\tau_1$' + ' (ns)',            gt_tau1_img, pred_tau1_img, err_tau1_img, stats_tau1),
        (r'$\tau_2$' + ' (ns)',            gt_tau2_img, pred_tau2_img, err_tau2_img, stats_tau2),
        ('Bound Fraction' + r' ($f_1$)',   gt_f1_img,   pred_f1_img,   err_f1_img,   stats_f1),
    ]

    for r, (label, gt_img, pr_img, er_img, st) in enumerate(rows):
        vmin, vmax = np.nanmin(gt_img), np.nanmax(gt_img)
        elim = max(np.nanmax(np.abs(er_img)), 1e-6)

        # colormaps with NaN rendered white
        cmap_val = copy.copy(plt.cm.viridis);  cmap_val.set_bad('white')
        cmap_err = copy.copy(plt.cm.RdBu_r);   cmap_err.set_bad('white')

        # --- ground truth ---
        im0 = axes[r, 0].imshow(gt_img, cmap=cmap_val, vmin=vmin, vmax=vmax)
        axes[r, 0].set_ylabel(label, fontsize=13, fontweight='bold')
        axes[r, 0].axis('off')
        fig.colorbar(im0, ax=axes[r, 0], shrink=0.82, pad=0.02, label=label)

        # --- predicted (same clim as gt) ---
        im1 = axes[r, 1].imshow(pr_img, cmap=cmap_val, vmin=vmin, vmax=vmax)
        axes[r, 1].axis('off')
        fig.colorbar(im1, ax=axes[r, 1], shrink=0.82, pad=0.02, label=label)

        # --- error with stats text box ---
        im2 = axes[r, 2].imshow(er_img, cmap=cmap_err, vmin=-elim, vmax=elim)
        axes[r, 2].axis('off')
        fig.colorbar(im2, ax=axes[r, 2], shrink=0.82, pad=0.02, label=label + ' error')

        txt = (f"MAE  {st['mae']:.4f}\n"
               f"RMSE {st['rmse']:.4f}")
        axes[r, 2].text(0.02, 0.97, txt, transform=axes[r, 2].transAxes,
                        fontsize=9, va='top', ha='left',
                        bbox=dict(fc='white', ec='grey', alpha=0.85, pad=3),
                        family='monospace')

    # column titles on top row — include the parameter name for clarity
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight='bold', pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(OUT_DIR, f'inference_self_{file_idx+1:02d}_{mat_name.replace(".mat","")}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved -> {save_path}")
    print(f"  time: {time.time()-t0:.1f}s\n")

# ---------------------------------------------------------------------------
# summary across all images
# ---------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY (mean MAE across images)")
print("=" * 60)
for k in ('tau1', 'tau2', 'f1'):
    print(f"  {k:5s}  mean MAE = {np.mean(all_errors[k]):.4f}")

# ---------------------------------------------------------------------------
# combined predicted vs ground-truth scatter (all images pooled)
# ---------------------------------------------------------------------------
gt_tau1_all   = np.concatenate(all_gt_tau1)
pred_tau1_all = np.concatenate(all_pred_tau1)
gt_tau2_all   = np.concatenate(all_gt_tau2)
pred_tau2_all = np.concatenate(all_pred_tau2)
gt_f1_all     = np.concatenate(all_gt_f1)
pred_f1_all   = np.concatenate(all_pred_f1)

total_pixels = len(gt_tau1_all)
print(f"\nCombined scatter: {total_pixels:,} pixels across {len(mat_files)} images")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Predicted vs Ground Truth  —  all images combined  (realdata_stats, self-inference)',
             fontsize=13, y=1.02)

panels = [
    (axes[0], gt_tau1_all,  pred_tau1_all, r'$\tau_1$' + ' (ns)'),
    (axes[1], gt_tau2_all,  pred_tau2_all, r'$\tau_2$' + ' (ns)'),
    (axes[2], gt_f1_all,    pred_f1_all,   r'Bound Fraction ($f_1$)'),
]

for ax, gt, pred, label in panels:
    mae  = np.abs(pred - gt).mean()
    rmse = np.sqrt(((pred - gt)**2).mean())

    # identity line limits from the data
    lo = min(gt.min(), pred.min())
    hi = max(gt.max(), pred.max())
    pad = (hi - lo) * 0.03
    lo -= pad; hi += pad

    ax.scatter(gt, pred, s=0.8, alpha=0.35, c='steelblue', edgecolors='none')
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.2, label='Identity')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Ground Truth', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    # stats box
    txt = (f"$n$ = {len(gt):,}\n"
           f"MAE  = {mae:.4f}\n"
           f"RMSE = {rmse:.4f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(fc='white', ec='grey', alpha=0.85, pad=3),
            family='monospace')
    ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
scatter_path = os.path.join(OUT_DIR, 'combined_predicted_vs_gt_self.png')
plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Combined scatter saved -> {scatter_path}")
print(f"\nAll plots saved to: {OUT_DIR}")
