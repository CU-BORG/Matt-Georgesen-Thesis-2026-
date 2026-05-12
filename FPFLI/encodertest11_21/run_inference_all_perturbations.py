#!/usr/bin/env python3
"""
Run the level-04 fraction-weighted model on every perturbation group.
Produces per-image GT/Pred/Error maps, a combined scatter per group,
and a final cross-group summary bar-chart.

@author: mg
"""

import sys, os, time, copy
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

DATA_BASE  = r'E:\perturbations_bin62_12.5ns'

OUT_BASE   = (r'C:\Users\mcg11923\Thesis'
              r'\training_logs_decay_bin62_12.5ns_20260306_112041'
              r'\inference_results_perturbations')

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
    print("  Running on GPU")
else:
    print("  Running on CPU")

# ---------------------------------------------------------------------------
# inference helpers  (extracted from run_inference.py)
# ---------------------------------------------------------------------------
def run_one_image(mat_path):
    """Load one .mat, run model, return (gt, pred) flat arrays + valid_coords."""
    with h5py.File(mat_path, 'r') as f:
        Hist   = np.array(f['Hist'])                  # [T, H, W]
        tau_gt = np.array(f['tau_gt_components'])     # [2, H, W]
        f_gt   = np.array(f['f_gt_components'])       # [2, H, W]

    T, H, W = Hist.shape

    
    irf = Hist[:, :10, :10].mean(axis=(1, 2))
    irf = irf / (irf.max() + 1e-8)

    # intensity map normalised to [0,1]
    Int_full = Hist.sum(axis=0).astype(np.float32)
    Int_max  = Int_full.max()
    if Int_max > 0:
        Int_full /= Int_max

    # collect valid pixels
    decays_list, gt_tau1_list, gt_tau2_list, gt_f1_list = [], [], [], []
    valid_coords = []

    for y in range(0, H, SAMPLE_STEP):
        for x in range(0, W, SAMPLE_STEP):
            if Int_full[y, x] < INTENSITY_THRESHOLD:
                continue
            decays_list.append(Hist[:, y, x])
            gt_tau1_list.append(tau_gt[0, y, x])
            gt_tau2_list.append(tau_gt[1, y, x])
            gt_f1_list.append(f_gt[0, y, x])
            valid_coords.append((y, x))

    n_valid = len(valid_coords)
    if n_valid == 0:
        return None   # skip empty images

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


def to_img(arr, valid_coords, H, W):
    img = np.full((H, W), np.nan, dtype=np.float32)
    for k, (y, x) in enumerate(valid_coords):
        img[y, x] = arr[k]
    return img


def save_perimage_plot(res, mat_name, group_name, out_dir, file_idx):
    """3x3 GT / Pred / Error map for one image."""
    vc   = res['valid_coords']
    H, W = res['H'], res['W']

    gt_tau1_img   = to_img(res['gt_tau1'],   vc, H, W)
    gt_tau2_img   = to_img(res['gt_tau2'],   vc, H, W)
    gt_f1_img     = to_img(res['gt_f1'],     vc, H, W)
    pred_tau1_img = to_img(res['pred_tau1'], vc, H, W)
    pred_tau2_img = to_img(res['pred_tau2'], vc, H, W)
    pred_f1_img   = to_img(res['pred_f1'],   vc, H, W)
    err_tau1_img  = to_img(res['pred_tau1'] - res['gt_tau1'], vc, H, W)
    err_tau2_img  = to_img(res['pred_tau2'] - res['gt_tau2'], vc, H, W)
    err_f1_img    = to_img(res['pred_f1']   - res['gt_f1'],   vc, H, W)

    st_tau1 = compute_stats(res['pred_tau1'], res['gt_tau1'])
    st_tau2 = compute_stats(res['pred_tau2'], res['gt_tau2'])
    st_f1   = compute_stats(res['pred_f1'],   res['gt_f1'])

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f'{mat_name}  ({group_name}, level-04 fraction-weighted)',
                 fontsize=14, y=0.98)

    col_titles = ['Ground Truth', 'Predicted', 'Error (pred - gt)']
    rows = [
        (r'$\tau_1$' + ' (ns)',           gt_tau1_img, pred_tau1_img, err_tau1_img, st_tau1),
        (r'$\tau_2$' + ' (ns)',           gt_tau2_img, pred_tau2_img, err_tau2_img, st_tau2),
        ('Bound Fraction' + r' ($f_1$)',  gt_f1_img,   pred_f1_img,   err_f1_img,   st_f1),
    ]

    for r, (label, gt_img, pr_img, er_img, st) in enumerate(rows):
        vmin, vmax = np.nanmin(gt_img), np.nanmax(gt_img)
        elim = max(np.nanmax(np.abs(er_img)), 1e-6)

        cmap_val = copy.copy(plt.cm.viridis);  cmap_val.set_bad('white')
        cmap_err = copy.copy(plt.cm.RdBu_r);   cmap_err.set_bad('white')

        im0 = axes[r, 0].imshow(gt_img, cmap=cmap_val, vmin=vmin, vmax=vmax)
        axes[r, 0].set_ylabel(label, fontsize=13, fontweight='bold')
        axes[r, 0].axis('off')
        fig.colorbar(im0, ax=axes[r, 0], shrink=0.82, pad=0.02, label=label)

        im1 = axes[r, 1].imshow(pr_img, cmap=cmap_val, vmin=vmin, vmax=vmax)
        axes[r, 1].axis('off')
        fig.colorbar(im1, ax=axes[r, 1], shrink=0.82, pad=0.02, label=label)

        im2 = axes[r, 2].imshow(er_img, cmap=cmap_err, vmin=-elim, vmax=elim)
        axes[r, 2].axis('off')
        fig.colorbar(im2, ax=axes[r, 2], shrink=0.82, pad=0.02, label=label + ' error')

        txt = (f"MAE  {st['mae']:.4f}\n"
               f"RMSE {st['rmse']:.4f}")
        axes[r, 2].text(0.02, 0.97, txt, transform=axes[r, 2].transAxes,
                        fontsize=9, va='top', ha='left',
                        bbox=dict(fc='white', ec='grey', alpha=0.85, pad=3),
                        family='monospace')

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight='bold', pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(out_dir, f'inference_{file_idx+1:02d}_{mat_name.replace(".mat","")}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_group_scatter(all_gt, all_pred, group_name, out_dir):
    """Combined predicted-vs-gt scatter for one group."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Predicted vs Ground Truth  —  {group_name}  (level 04, fraction-weighted)',
                 fontsize=13, y=1.02)

    panels = [
        (axes[0], all_gt['tau1'],  all_pred['tau1'],  r'$\tau_1$' + ' (ns)'),
        (axes[1], all_gt['tau2'],  all_pred['tau2'],  r'$\tau_2$' + ' (ns)'),
        (axes[2], all_gt['f1'],    all_pred['f1'],    r'Bound Fraction ($f_1$)'),
    ]

    for ax, gt, pred, label in panels:
        mae  = np.abs(pred - gt).mean()
        rmse = np.sqrt(((pred - gt)**2).mean())

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

        txt = (f"$n$ = {len(gt):,}\n"
               f"MAE  = {mae:.4f}\n"
               f"RMSE = {rmse:.4f}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(fc='white', ec='grey', alpha=0.85, pad=3),
                family='monospace')
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f'combined_predicted_vs_gt_{group_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined scatter -> {path}")


# ---------------------------------------------------------------------------
# cross-group summary bar chart
# ---------------------------------------------------------------------------
def save_summary_bar(summary, out_dir):
    """
    summary: dict  group_name -> {'tau1': mae, 'tau2': mae, 'f1': mae}
    """
    groups  = list(summary.keys())
    n       = len(groups)
    tau1_mae = [summary[g]['tau1'] for g in groups]
    tau2_mae = [summary[g]['tau2'] for g in groups]
    f1_mae   = [summary[g]['f1']   for g in groups]

    # Extract model name from OUT_BASE path
    # OUT_BASE looks like: .../training_logs_decay_bin62_12.5ns_20260306_112041/inference_results_perturbations
    model_dir = os.path.basename(os.path.dirname(out_dir))

    # short labels for x-axis
    short = {
        'baseline_normal':           'Baseline\n(normal)',
        'perturb_delayed_decay':     'Delayed\nDecay',
        'perturb_tri_exponential':   'Tri-Exp',
        'perturb_flat_background':   'Flat\nBackground',
        'perturb_50pct_photons':     '50 %\nPhotons',
        'perturb_10pct_photons':     '10 %\nPhotons',
        'perturb_1pct_photons':      '1 %\nPhotons',
    }
    labels = [short.get(g, g) for g in groups]

    x    = np.arange(n)
    w    = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Mean MAE by Perturbation Group  —  {model_dir}',
                 fontsize=14, fontweight='bold', y=1.02)

    datasets = [
        (axes[0], tau1_mae, r'$\tau_1$' + ' (ns)',  'tab:blue'),
        (axes[1], tau2_mae, r'$\tau_2$' + ' (ns)',  'tab:orange'),
        (axes[2], f1_mae,   r'Bound Fraction ($f_1$)', 'tab:green'),
    ]

    for ax, vals, title, color in datasets:
        bars = ax.bar(x, vals, width=0.5, color=color, edgecolor='black', linewidth=0.8)
        # value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Mean MAE', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'summary_mae_by_group.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSummary bar chart -> {path}")


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
os.makedirs(OUT_BASE, exist_ok=True)
summary = {}          # group -> mean MAE per param
total_t0 = time.time()

for group_name in GROUPS:
    data_dir = os.path.join(DATA_BASE, group_name)
    out_dir  = os.path.join(OUT_BASE, group_name)
    os.makedirs(out_dir, exist_ok=True)

    mat_files = sorted([fn for fn in os.listdir(data_dir) if fn.endswith('.mat')])[:N_IMAGES]

    print(f"\n{'='*70}")
    print(f"  GROUP: {group_name}  ({len(mat_files)} images)")
    print(f"{'='*70}")

    # accumulators
    all_gt   = {'tau1': [], 'tau2': [], 'f1': []}
    all_pred = {'tau1': [], 'tau2': [], 'f1': []}
    mae_per_image = {'tau1': [], 'tau2': [], 'f1': []}

    for file_idx, mat_name in enumerate(mat_files):
        t0 = time.time()
        mat_path = os.path.join(data_dir, mat_name)
        print(f"  --- [{file_idx+1}/{len(mat_files)}] {mat_name} ---")

        res = run_one_image(mat_path)
        if res is None:
            print("    (no valid pixels — skipped)")
            continue

        n_valid = len(res['valid_coords'])

        # stats
        st1 = compute_stats(res['pred_tau1'], res['gt_tau1'])
        st2 = compute_stats(res['pred_tau2'], res['gt_tau2'])
        sf  = compute_stats(res['pred_f1'],   res['gt_f1'])
        print(f"    valid={n_valid}  "
              f"tau1 MAE={st1['mae']:.4f}  "
              f"tau2 MAE={st2['mae']:.4f}  "
              f"f1   MAE={sf['mae']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        mae_per_image['tau1'].append(st1['mae'])
        mae_per_image['tau2'].append(st2['mae'])
        mae_per_image['f1'].append(sf['mae'])

        # accumulate for scatter
        all_gt['tau1'].append(res['gt_tau1']);    all_pred['tau1'].append(res['pred_tau1'])
        all_gt['tau2'].append(res['gt_tau2']);    all_pred['tau2'].append(res['pred_tau2'])
        all_gt['f1'].append(res['gt_f1']);        all_pred['f1'].append(res['pred_f1'])

        # per-image map plot
        save_perimage_plot(res, mat_name, group_name, out_dir, file_idx)

    # concatenate accumulators
    for k in all_gt:
        all_gt[k]   = np.concatenate(all_gt[k])
        all_pred[k] = np.concatenate(all_pred[k])

    # group scatter
    save_group_scatter(all_gt, all_pred, group_name, out_dir)

    # record mean MAE for summary
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
# cross-group summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  CROSS-GROUP SUMMARY  (mean MAE)")
print(f"{'='*70}")
print(f"  {'Group':<30s}  {'tau1':>8s}  {'tau2':>8s}  {'f1':>8s}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
for g in GROUPS:
    s = summary[g]
    print(f"  {g:<30s}  {s['tau1']:8.4f}  {s['tau2']:8.4f}  {s['f1']:8.4f}")

save_summary_bar(summary, OUT_BASE)

total_elapsed = time.time() - total_t0
print(f"\nTotal runtime: {total_elapsed:.0f} s  ({total_elapsed/60:.1f} min)")
print(f"All outputs in: {OUT_BASE}")
