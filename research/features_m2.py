"""M2 features: gain-invariant, mostly rotation-invariant handcrafted features.

Cache: research/cache_m2.npz (X_train, X_test, feature_names).
Run: python research/features_m2.py  (~2-4 min local CPU, ~4000 FFTs of 32^3)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
N_RADIAL = 14

# ---- static grids (32^3) ----
_ax = np.arange(32) - 15.5
XX, YY, ZZ = np.meshgrid(_ax, _ax, _ax, indexing="ij")
R_SPAT = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
KX = np.fft.fftfreq(32) * 32
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)
# radial bin edges in frequency: 0.5..16
BINS = np.linspace(0.5, 16.0, N_RADIAL + 1)
BIN_IDX = np.clip(np.digitize(R_FREQ.ravel(), BINS) - 1, 0, N_RADIAL - 1)
C_SPHERE = (R_SPAT <= 6.0)
MID_SHELL = (R_SPAT > 6.0) & (R_SPAT <= 11.0)
OUT_SHELL = (R_SPAT > 11.0)


def pervol_norm(v):
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    if s < 1e-6:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def radial_profile(vn):
    P = np.abs(np.fft.fftn(vn)) ** 2
    prof = np.bincount(BIN_IDX, weights=P.ravel(), minlength=N_RADIAL)
    prof = np.log1p(prof)
    s = prof.sum()
    return prof / s if s > 0 else prof  # gain-invariant shape


def spatial_feats(vn):
    f = {}
    f["skew"] = float(((vn ** 3).mean()))
    f["kurt"] = float((vn ** 4).mean())
    f["frac_gt2"] = float((vn > 2).mean())
    f["frac_lt_2"] = float((vn < -2).mean())
    f["p99"] = float(np.percentile(vn, 99))
    f["p1"] = float(np.percentile(vn, 1))
    c, m, o = vn[C_SPHERE].mean(), vn[MID_SHELL].mean(), vn[OUT_SHELL].mean()
    f["center_mean"] = float(c)
    f["mid_mean"] = float(m)
    f["outer_mean"] = float(o)
    f["center_vs_outer"] = float(c - o)
    # z-elongation (wedge signature): variance of z-means vs x/y-means
    f["z_elong"] = float(vn.mean(axis=(0, 1)).var() /
                         (vn.mean(axis=(0, 2)).var() + vn.mean(axis=(1, 2)).var() + 1e-9))
    # asymmetry: 1 - corr with 90-deg z rotation (isotropic shapes ~1)
    r = np.rot90(vn, 1, axes=(0, 1)).ravel()
    v = vn.ravel()
    f["rot90_corr"] = float(np.corrcoef(v, r)[0, 1])
    # 180-deg flip correlations (filament/membrane strongly anisotropic)
    for axn, nm in [(0, "fx"), (1, "fy"), (2, "fz")]:
        w = np.flip(vn, axis=axn).ravel()
        f[f"flip_{nm}_corr"] = float(np.corrcoef(v, w)[0, 1])
    return f


FEAT_NAMES = ([f"rad{i:02d}" for i in range(N_RADIAL)]
              + ["skew", "kurt", "frac_gt2", "frac_lt_2", "p99", "p1",
                 "center_mean", "mid_mean", "outer_mean", "center_vs_outer",
                 "z_elong", "rot90_corr", "flip_fx_corr", "flip_fy_corr",
                 "flip_fz_corr",
                 "defocus_um", "tilt_max_deg", "az_sin", "az_cos",
                 "ice_thickness_nm"])


def featurize(vols, meta):
    n = len(vols)
    X = np.zeros((n, len(FEAT_NAMES)), dtype=np.float32)
    for i in range(n):
        vn = pervol_norm(np.asarray(vols[i], dtype=np.float32))
        prof = radial_profile(vn)
        s = spatial_feats(vn)
        row = list(prof) + [s[k] for k in FEAT_NAMES[N_RADIAL:N_RADIAL + 15]]
        az = np.deg2rad(meta["tilt_axis_azimuth_deg"].iloc[i])
        row += [meta["defocus_um"].iloc[i], meta["tilt_max_deg"].iloc[i],
                np.sin(2 * az), np.cos(2 * az), meta["ice_thickness_nm"].iloc[i]]
        X[i] = row
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    return X


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    tr = np.load(DATA / "train_volumes.npy", mmap_mode="r")
    te = np.load(DATA / "test_volumes.npy", mmap_mode="r")
    print("featurizing train...")
    Xtr = featurize(tr, train)
    print("featurizing test...")
    Xte = featurize(te, test)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(ROOT / "cache_m2.npz", X_train=Xtr, X_test=Xte,
                        names=np.array(FEAT_NAMES))
    print("saved", ROOT / "cache_m2.npz", Xtr.shape, Xte.shape)


if __name__ == "__main__":
    main()
