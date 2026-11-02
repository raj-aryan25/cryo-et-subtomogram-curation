"""Phase 1c: match augmentation strength to OBSERVED test feature shift, then
augment-train at matched strength. Steps:
 1) on 400 train vols: radial-bin means under blur(sigma)/wedge(tilt) grid +
    test bin means -> pick sigma*, tilt* minimizing z-distance on shifted bins.
 2) featurize full train at {clean, blur*, wedge*, blur*+wedge*} -> cache_aug2.
 3) OOF + pseudo-CD eval of augmented training.
Run: python research/phase1_match.py (CPU, ~20 min)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from features_m2 import pervol_norm, radial_profile
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
KX = np.fft.fftfreq(32) * 32
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)
BINS = np.linspace(0.5, 16.0, 15)
BIN_IDX = np.clip(np.digitize(R_FREQ.ravel(), BINS) - 1, 0, 13)
torch.manual_seed(42)
np.random.seed(42)


def wedge_mask(tilt_deg, az_deg):
    phi = np.deg2rad(az_deg)
    kn = -KXm * np.sin(phi) + KYm * np.cos(phi)
    edge = np.tan(np.deg2rad(tilt_deg)) * np.abs(kn)
    return ((np.abs(KZm) <= edge) | (R_FREQ < 1.5)).astype(np.float32)


def radprof_of(v):
    vn = pervol_norm(np.asarray(v, dtype=np.float32))
    return radial_profile(vn)


def main():
    train, _ = load_meta()
    tr = np.load(ROOT.parent / "data" / "train_volumes.npy", mmap_mode="r")
    te = np.load(ROOT.parent / "data" / "test_volumes.npy", mmap_mode="r")
    az_tr = train["tilt_axis_azimuth_deg"].values
    rng = np.random.default_rng(1)
    sub = rng.choice(len(tr), 400, replace=False)
    te_sub = rng.choice(len(te), 400, replace=False)

    print("test radial-bin means...")
    te_profs = np.stack([radprof_of(te[i]) for i in te_sub])
    te_mean = te_profs.mean(0)
    tr_profs = np.stack([radprof_of(tr[i]) for i in sub])
    tr_mean = tr_profs.mean(0)
    tr_sd = tr_profs.std(0) + 1e-9
    # shifted bins from DIAG0
    FOCUS = [3, 4, 5, 7, 8]
    print(f"clean dist on focus bins: "
          f"{float(np.abs(tr_mean[FOCUS] - te_mean[FOCUS]).mean()):.4f}")

    print("grid search blur x wedge...")
    vols = [np.asarray(tr[i], dtype=np.float32) for i in sub]
    azs = az_tr[sub]
    results = {}
    for sig in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]:
        bv = [gaussian_filter(v, sig) if sig > 0 else v for v in vols]
        for tilt in [None, 44.0, 47.0, 49.0, 52.0]:
            if tilt is None:
                pv = bv
                tag = f"blur{sig}"
            else:
                pv = []
                for v, a in zip(bv, azs):
                    F = np.fft.fftn(v)
                    pv.append(np.real(np.fft.ifftn(F * wedge_mask(tilt, a))).astype(np.float32))
                tag = f"blur{sig}_wedge{tilt}"
            pm = np.stack([radprof_of(v) for v in pv]).mean(0)
            d = float(np.abs(pm[FOCUS] - te_mean[FOCUS]).mean())
            dall = float(np.abs(pm - te_mean).mean())
            results[tag] = (d, dall)
            print(f"  {tag:<18} focus-dist={d:.4f} all-dist={dall:.4f}", flush=True)
    best = min(results, key=lambda k: results[k][0])
    print("BEST:", best, results[best])
    with open(ROOT / "diag0_match.json", "w") as f:
        json.dump({"best": best, "all": {k: list(v) for k, v in results.items()},
                   "te_mean": te_mean.tolist()}, f)


if __name__ == "__main__":
    main()
