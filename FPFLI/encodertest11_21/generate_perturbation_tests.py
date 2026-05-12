#!/usr/bin/env python3
"""
Generate 6 perturbation test-image groups (10 images each) for robustness testing.

Each group uses the same baseline physics as the real-data statistics training data
(tau1=0.41 ns, tau2=2.51 ns, std 0.08/0.23, photons 517-1266) but with one
controlled perturbation:

    1  perturb_delayed_decay      decay shifted 30-100 bins (range across 10 images)
    2  perturb_tri_exponential    extra short-lifetime (0.3 ns) component added
    3  perturb_flat_background    uniform photon floor added across all bins
    4  perturb_50pct_photons      photon counts scaled to 50 %
    5  perturb_10pct_photons      photon counts scaled to 10 %
    6  perturb_1pct_photons       photon counts scaled to  1 %

Output .mat files are HDF5 v7.3 with the exact same variables the inference
pipeline already reads:
    Hist                [T=256, H=256, W=256]   (h5py axis order)
    tau_gt_components   [2, 256, 256]           Component 0=SHORT, 1=LONG
    f_gt_components     [2, 256, 256]           Component 0=(1-f), 1=f (bound)
    tau_gt_avg          [256, 256]
    Int                 [256, 256]
    max_components      scalar (2)

Note: f (Component 1) represents bound fraction for LONG lifetime (tau2).
      (1-f) (Component 0) represents free fraction for SHORT lifetime (tau1).

@author: mg
"""

import os, time, glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import h5py

# ---------------------------------------------------------------------------
# paths & config  (match RealDataStats parameters)
# ---------------------------------------------------------------------------
TRAIN_IMG_DIR = r'C:\Users\mcg11923\Thesis\train'
IRF_MAT_PATH  = r'C:\Users\mcg11923\Thesis\FPFLI\Evaluation\sample_data\irf_measure.mat'
OUT_BASE      = (r'C:\Users\mcg11923\Thesis\FPFLI\Synthetic data preparation'
                 r'\generate_multiexp_spectrumless\generate_spectrum_lite')

N_IMAGES      = 10
IMG_SIZE      = 256          # output spatial size
T_BINS        = 256          # number of time bins
BIN_WIDTH     = 0.039        # ns per bin

# REAL DATA STATISTICS parameters (from real_data_statistics.json)
TAU_CENTER    = np.array([0.4095, 2.5131])  # ns (means from real data)
TAU_STD       = np.array([0.0777, 0.2316])  # ns (std devs from real data)
TAU_MIN       = np.array([0.124,  1.528])   # ns (observed min from real data)
TAU_MAX       = np.array([0.784,  4.293])   # ns (observed max from real data)
PHOTON_MIN    = 517                         # p5 from real data
PHOTON_MAX    = 1266                        # p95 from real data
INTENSITY_THR = 0.05

# spatial correlation lengths (pixels) - from real data
TAU_CORR_LEN  = np.array([3.0, 2.29])       # per component
F_CORR_LEN    = 3.0
INTENSITY_MOD_TAU = 0.15                     # from real data config
INTENSITY_MOD_F   = 0.15
F_RANGE       = (0.516, 0.903)               # observed range for f (bound fraction, LONG tau2)

# ---------------------------------------------------------------------------
# group definitions  — each entry: (folder_name, perturbation_tag)
# ---------------------------------------------------------------------------
GROUPS = [
    'baseline_normal',              # no perturbation — clean reference
    'perturb_delayed_decay',
    'perturb_tri_exponential',
    'perturb_flat_background',
    'perturb_50pct_photons',
    'perturb_10pct_photons',
    'perturb_1pct_photons',
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_irf():
    """Load measured IRF or fall back to Gaussian."""
    if os.path.exists(IRF_MAT_PATH):
        print(f"  Loading IRF from {IRF_MAT_PATH}")
        with h5py.File(IRF_MAT_PATH, 'r') as f:
            irf = np.array(f['irf']).flatten()
    else:
        print("  IRF file not found — using Gaussian fallback")
        t = np.arange(1, T_BINS + 1)
        t0 = 14
        FWHM = 0.1673
        sig0 = FWHM / 2.3548 / BIN_WIDTH
        irf = np.exp(-(t - t0)**2 / (2 * sig0**2))
    irf = irf / irf.max()
    return irf


def load_intensity_images(n):
    """Load n grayscale 256x256 intensity images from TRAIN_IMG_DIR."""
    all_pngs = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, '*.png')))
    if len(all_pngs) < n:
        raise FileNotFoundError(
            f"Need {n} PNGs in {TRAIN_IMG_DIR}, found {len(all_pngs)}")
    # pick evenly-spaced images for variety
    indices = np.linspace(0, len(all_pngs) - 1, n, dtype=int)
    images = []
    for idx in indices:
        img = Image.open(all_pngs[idx]).convert('L')   # grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        images.append(np.array(img, dtype=np.float64) / 255.0)
    return images


def make_tau_map(intensity_norm, rng):
    """Generate spatially-correlated tau map [2, 256, 256] using real data parameters."""
    tau_map = np.zeros((2, IMG_SIZE, IMG_SIZE))
    for comp in range(2):
        noise = rng.standard_normal((IMG_SIZE, IMG_SIZE))
        smooth = gaussian_filter(noise, sigma=TAU_CORR_LEN[comp])
        smooth /= (smooth.std() + 1e-10)
        # intensity-guided modulation
        int_effect = INTENSITY_MOD_TAU * (intensity_norm - 0.5) * 2
        raw = TAU_CENTER[comp] + TAU_STD[comp] * (smooth + int_effect)
        # clamp to observed real-data range
        tau_map[comp] = np.clip(raw, TAU_MIN[comp], TAU_MAX[comp])

    # Ensure tau1 < tau2 everywhere (swap if needed)
    swap_mask = tau_map[0] >= tau_map[1]
    if np.any(swap_mask):
        temp = tau_map[0].copy()
        tau_map[0] = np.where(swap_mask, tau_map[1], tau_map[0])
        tau_map[1] = np.where(swap_mask, temp, tau_map[1])

    return tau_map


def make_f_map(intensity_norm, rng):
    """
    Generate per-pixel Dirichlet fractions, then spatially smooth (matching real data).

    Uses Dirichlet(alpha=[110.49, 45.71]) to match real data statistics,
    which creates a tight distribution around f ≈ 0.707 for the LONG lifetime (tau2).

    Note: f (Component 2) represents the bound fraction for LONG tau2.
          (1-f) (Component 1) represents the free fraction for SHORT tau1.
    """
    # Dirichlet alpha parameters from real data
    DIRICHLET_ALPHA = np.array([110.49, 45.71])

    # Marsaglia & Tsang method for Gamma sampling (no toolbox needed)
    n_px = IMG_SIZE * IMG_SIZE
    gamma_samp = np.zeros((n_px, 2))

    for k in range(2):
        a = DIRICHLET_ALPHA[k]
        d = a - 1.0/3.0
        c = 1.0 / np.sqrt(9.0 * d)
        g = np.zeros(n_px)
        filled = 0

        while filled < n_px:
            batch = n_px - filled
            x = rng.standard_normal(batch)
            v = (1.0 + c * x) ** 3
            keep = v > 0
            x = x[keep]
            v = v[keep]
            u = rng.random(len(x))

            # Acceptance criterion
            accept = (u < 1.0 - 0.0331 * (x**2)**2) | \
                     (np.log(u) < 0.5 * x**2 + d * (1.0 - v + np.log(v)))
            accepted_v = d * v[accept]
            n_new = min(len(accepted_v), n_px - filled)
            g[filled:filled + n_new] = accepted_v[:n_new]
            filled += n_new

        gamma_samp[:, k] = g

    # Normalize to get Dirichlet samples
    dir_samp = gamma_samp / gamma_samp.sum(axis=1, keepdims=True)
    f_raw = dir_samp[:, 0].reshape(IMG_SIZE, IMG_SIZE)

    # Smooth to create spatial structure while preserving the mean
    f_mean = f_raw.mean()
    f_smooth = gaussian_filter(f_raw, sigma=F_CORR_LEN)
    # Re-centre so the image-wide mean matches the Dirichlet mean
    f_smooth = f_smooth - f_smooth.mean() + f_mean

    # Clip to the observed real-data range for f (bound fraction, LONG tau2)
    f_smooth = np.clip(f_smooth, F_RANGE[0], F_RANGE[1])

    f_map = np.zeros((2, IMG_SIZE, IMG_SIZE))
    # f (Component 2) = bound fraction for LONG lifetime (tau2)
    f_map[1] = f_smooth
    # (1-f) (Component 1) = free fraction for SHORT lifetime (tau1)
    f_map[0] = 1.0 - f_smooth
    return f_map


def decay_curve(tau_num, tau, f, irf):
    """
    Compute the noiseless convolved decay (no Poisson sampling).
    Mirrors MATLAB Fluorescence_multi_decay_nonhomopp exactly.
    tau, f: 1-D arrays length tau_num
    Returns: lambda array [T_BINS]
    """
    t = np.arange(1, T_BINS + 1, dtype=np.float64)
    yo = np.zeros(T_BINS)
    for i in range(tau_num):
        yo += f[i] * np.exp(-t / (tau[i] / BIN_WIDTH))
    # convolve with IRF
    C = np.convolve(irf, yo)
    lam = C[:T_BINS]
    return lam


def poisson_sample(lam, n_photons, rng):
    """
    Acceptance-rejection sampling matching the MATLAB implementation.
    Returns integer histogram [T_BINS].
    """
    lam_bound = lam.max() * 1.1
    counts = np.zeros(T_BINS, dtype=np.int64)
    collected = 0
    # vectorised rejection sampling in chunks for speed
    while collected < n_photons:
        needed = n_photons - collected
        # generate more candidates than needed (accept rate ≈ mean/max ≈ 0.3-0.5)
        n_cand = int(needed * 3.5)
        u1 = rng.integers(0, T_BINS, size=n_cand)       # bin indices 0-based
        u2 = rng.random(n_cand)
        accept = u2 <= lam[u1] / lam_bound
        accepted_bins = u1[accept]
        take = min(len(accepted_bins), needed)
        for b in accepted_bins[:take]:
            counts[b] += 1
        collected += take
    return counts


def generate_pixel_histogram(tau, f, n_photons, irf, rng,
                             delay_bins=0, background_counts=0):
    """
    Full per-pixel pipeline: decay → convolve → sample → optional perturbations.

    delay_bins : shift the lambda curve right by this many bins
    background_counts : flat photons added uniformly after sampling
    """
    lam = decay_curve(len(tau), tau, f, irf)

    if delay_bins > 0:
        # circular shift — photons that "fall off" the end are lost (realistic)
        lam = np.roll(lam, delay_bins)
        lam[:delay_bins] = 0.0   # zero out the leading bins

    if lam.max() < 1e-12:
        return np.zeros(T_BINS, dtype=np.int64)

    hist = poisson_sample(lam, n_photons, rng)

    if background_counts > 0:
        # distribute background_counts uniformly across all bins
        bg_per_bin = background_counts // T_BINS
        remainder  = background_counts  % T_BINS
        hist += bg_per_bin
        # sprinkle the remainder into random bins
        extra_bins = rng.integers(0, T_BINS, size=remainder)
        for b in extra_bins:
            hist[b] += 1

    return hist


# ---------------------------------------------------------------------------
# per-group image generation
# ---------------------------------------------------------------------------
def generate_group(group_name, intensity_images, irf, master_rng):
    """Generate 10 .mat files for one perturbation group."""
    out_dir = os.path.join(OUT_BASE, group_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  GROUP: {group_name}")
    print(f"{'='*70}")

    for img_idx in range(N_IMAGES):
        t0 = time.time()
        rng = np.random.default_rng(master_rng.integers(0, 2**31))

        Int = intensity_images[img_idx]
        intensity_norm = (Int - Int.min()) / (Int.max() - Int.min() + 1e-10)

        tau_map = make_tau_map(intensity_norm, rng)   # [2, H, W]
        f_map   = make_f_map(intensity_norm, rng)     # [2, H, W]

        # --- group-specific overrides ------------------------------------------
        # photon scale
        ph_min, ph_max = PHOTON_MIN, PHOTON_MAX
        if group_name == 'perturb_50pct_photons':
            ph_min, ph_max = int(PHOTON_MIN * 0.5), int(PHOTON_MAX * 0.5)
        elif group_name == 'perturb_10pct_photons':
            ph_min, ph_max = int(PHOTON_MIN * 0.1), int(PHOTON_MAX * 0.1)
        elif group_name == 'perturb_1pct_photons':
            ph_min, ph_max = max(1, int(PHOTON_MIN * 0.01)), max(1, int(PHOTON_MAX * 0.01))

        # delay - linearly spaced from 30 to 100 bins across 10 images
        if group_name == 'perturb_delayed_decay':
            # Image 0 -> 30 bins, Image 9 -> 100 bins
            delay_bins = int(round(30 + (100 - 30) * img_idx / (N_IMAGES - 1)))
        else:
            delay_bins = 0

        # tri-exponential: build a 3-component tau/f for decay generation
        use_tri = (group_name == 'perturb_tri_exponential')
        # ---------------------------------------------------------------------------

        # tau_gt / f_gt stored are always 2-component (what the model expects)
        tau_gt_components = tau_map.copy()      # [2, H, W]
        f_gt_components   = f_map.copy()        # [2, H, W]
        tau_gt_avg        = (tau_map[0] * f_map[0] +
                             tau_map[1] * f_map[1])

        # --- generate histogram cube -------------------------------------------
        Hist = np.zeros((T_BINS, IMG_SIZE, IMG_SIZE), dtype=np.float64)

        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                intensity = Int[y, x]
                if intensity < INTENSITY_THR:
                    continue

                n_photons = int(round(ph_min + intensity * (ph_max - ph_min)))
                if n_photons < 1:
                    continue

                tau_px = tau_map[:, y, x]   # [2]
                f_px   = f_map[:, y, x]     # [2]

                if use_tri:
                    # add τ3 = 0.3 ns; steal fraction from both components
                    tau3 = 0.3
                    # draw f3 from uniform [0.05, 0.25] per pixel
                    f3   = rng.uniform(0.05, 0.25)
                    # rescale f1, f2 so they still sum with f3 to 1
                    scale = (1.0 - f3) / (f_px[0] + f_px[1] + 1e-12)
                    tau_px_3 = np.array([tau_px[0], tau_px[1], tau3])
                    f_px_3   = np.array([f_px[0] * scale, f_px[1] * scale, f3])
                    tau_use, f_use = tau_px_3, f_px_3
                else:
                    tau_use, f_use = tau_px, f_px

                # background photon count for flat-background group
                bg = 0
                if group_name == 'perturb_flat_background':
                    # 5 % of n_photons spread uniformly
                    bg = max(1, int(round(n_photons * 0.05)))
                    n_photons_signal = n_photons - bg   # keep total similar
                    if n_photons_signal < 1:
                        n_photons_signal = 1
                else:
                    n_photons_signal = n_photons

                hist = generate_pixel_histogram(
                    tau_use, f_use, n_photons_signal, irf, rng,
                    delay_bins=delay_bins,
                    background_counts=bg
                )
                Hist[:, y, x] = hist

        # --- save --------------------------------------------------------------
        fname = os.path.join(out_dir, f'Sample_{img_idx+1:03d}_{group_name}.mat')
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('Hist',               data=Hist)
            hf.create_dataset('tau_gt_components',  data=tau_gt_components)
            hf.create_dataset('f_gt_components',    data=f_gt_components)
            hf.create_dataset('tau_gt_avg',         data=tau_gt_avg)
            hf.create_dataset('Int',                data=Int)
            hf.create_dataset('max_components',     data=np.array([[2.0]]))

        elapsed = time.time() - t0
        delay_info = f"  delay={delay_bins}" if delay_bins > 0 else ""
        print(f"  [{img_idx+1:2d}/{N_IMAGES}] {os.path.basename(fname)}  "
              f"({elapsed:.1f} s)  photons={n_photons}{delay_info}")

    print(f"  -> saved to {out_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    total_t0 = time.time()
    master_rng = np.random.default_rng(42)

    print("Loading IRF ...")
    irf = load_irf()
    print(f"  IRF shape: {irf.shape}, peak at bin {np.argmax(irf)}")

    print("\nLoading intensity images ...")
    intensity_images = load_intensity_images(N_IMAGES)
    print(f"  Loaded {len(intensity_images)} images, shape {intensity_images[0].shape}")

    for group in GROUPS:
        generate_group(group, intensity_images, irf, master_rng)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  ALL DONE  —  {len(GROUPS)} groups x {N_IMAGES} images")
    print(f"  Total time: {total_elapsed:.0f} s  ({total_elapsed/60:.1f} min)")
    print(f"  Output base: {OUT_BASE}")
    print(f"{'='*70}")
