"""Phase 1: wedge-robust wide feature matrix.

Per volume (uses released defocus/tilt/azimuth ONLY — all allowed):
  raw radial (14, v2) +
  phase-flipped radial on common-wedge W(44deg, tomo-azimuth) (14) +
  spatial stats on flipped+masked spectrum, full box (15, same keys as v2) +
  central-24^3 moments (8: skew,kurt,frac_gt2,frac_lt_2,p99,p1,center,mid) +
  confidence-only wedge descriptors (2: fillfrac, inout_ratio)
= 53 cols. Cache: research/cache_v3.npz (X_train, X_test, names).
Run: python research/phase1_feats.py (CPU, ~8 min for 3993 vols)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
N_RADIAL = 14

_ax = np.arange(32) - 15.5
XX, YY, ZZ = np.meshgrid(_ax, _ax, _ax, indexing="ij")
R_SPAT = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
KX = np.fft.fftfreq(32) * 32
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)
BINS = np.linspace(0.5, 16.0, N_RADIAL + 1)
BIN_IDX = np.clip(np.digitize(R_FREQ.ravel(), BINS) - 1, 0, N_RADIAL - 1)
C_SPHERE = (R_SPAT <= 6.0)
MID_SHELL = (R_SPAT > 6.0) & (R_SPAT <= 11.0)
OUT_SHELL = (R_SPAT > 11.0)
C24 = np.zeros_like(R_SPAT, dtype=bool)
C24[4:28, 4:28, 4:28] = True  # central 24^3 (crowding enters via faces)

# CTF constants: 300 keV electrons, Cs 2.7 mm, amplitude contrast 7%.
LAM = 0.0197   # Angstrom
CS = 2.7e7     # Angstrom
AMP_C = 0.07
VOX_A = 10.0
S_FREQ = R_FREQ / (32.0 * VOX_A)  # 1/Angstrom
S2 = S_FREQ ** 2
CENTRAL_BALL = (R_FREQ < 1.5)


def ctf_sign(defocus_um):
    df = defocus_um * 1e4  # um -> A
    chi = np.pi * LAM * S2 * (df - 0.5 * LAM ** 2 * S2 * CS)
    ctf = -(np.sqrt(1 - AMP_C ** 2) * np.sin(chi) + AMP_C * np.cos(chi))
    s = np.sign(ctf)
    s[s == 0] = 1.0
    return s


def wedge_mask(tilt_deg, az_deg):
    phi = np.deg2rad(az_deg)
    kn = -KXm * np.sin(phi) + KYm * np.cos(phi)
    edge = np.tan(np.deg2rad(tilt_deg)) * np.abs(kn)
    return ((np.abs(KZm) <= edge) | CENTRAL_BALL)


SPAT_KEYS = ["skew", "kurt", "frac_gt2", "frac_lt_2", "p99", "p1",
             "center_mean", "mid_mean", "outer_mean", "center_vs_outer",
             "z_elong", "rot90_corr", "flip_fx_corr", "flip_fy_corr",
             "flip_fz_corr"]
CENT_KEYS = ["skew", "kurt", "frac_gt2", "frac_lt_2", "p99", "p1",
             "center_mean", "mid_mean"]
NAMES = ([f"rad{i:02d}" for i in range(N_RADIAL)]
         + [f"pflip_rad{i:02d}" for i in range(N_RADIAL)]
         + [f"w_{k}" for k in SPAT_KEYS]
         + [f"c24_{k}" for k in CENT_KEYS]
         + ["wedge_fillfrac", "wedge_inout"])


def spatial_dict(vn):
    v = vn.ravel()
    r = np.rot90(vn, 1, axes=(0, 1)).ravel()
    d = {
        "skew": float((vn ** 3).mean()), "kurt": float((vn ** 4).mean()),
        "frac_gt2": float((vn > 2).mean()), "frac_lt_2": float((vn < -2).mean()),
        "p99": float(np.percentile(vn, 99)), "p1": float(np.percentile(vn, 1)),
        "center_mean": float(vn[C_SPHERE].mean()),
        "mid_mean": float(vn[MID_SHELL].mean()),
        "outer_mean": float(vn[OUT_SHELL].mean()),
        "center_vs_outer": float(vn[C_SPHERE].mean() - vn[OUT_SHELL].mean()),
        "z_elong": float(vn.mean(axis=(0, 1)).var() / (
            vn.mean(axis=(0, 2)).var() + vn.mean(axis=(1, 2)).var() + 1e-9)),
        "rot90_corr": float(np.corrcoef(v, r)[0, 1])}
    for axn, nm in [(0, "fx"), (1, "fy"), (2, "fz")]:
        w = np.flip(vn, axis=axn).ravel()
        d[f"flip_{nm}_corr"] = float(np.corrcoef(v, w)[0, 1])
    return d


def central_dict(vn):
    c = vn[C24]
    v = c.ravel()
    return {
        "skew": float((c ** 3).mean()), "kurt": float((c ** 4).mean()),
        "frac_gt2": float((c > 2).mean()), "frac_lt_2": float((c < -2).mean()),
        "p99": float(np.percentile(c, 99)), "p1": float(np.percentile(c, 1)),
        "center_mean": float(vn[C_SPHERE].mean()),
        "mid_mean": float(vn[MID_SHELL].mean())}


def pervol_norm(v):
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    if s < 1e-6:
        return np.zeros_like(v)
    return (v - v.mean()) / s


def featurize(vols, meta, ctf_cache, wedge_cache):
    n = len(vols)
    X = np.zeros((n, len(NAMES)), dtype=np.float32)
    for i in range(n):
        v = np.asarray(vols[i], dtype=np.float32)
        F = np.fft.fftn(v)
        df = float(meta["defocus_um"].iloc[i])
        az = float(meta["tilt_axis_azimuth_deg"].iloc[i])
        key = (round(df, 4), round(az, 4))
        if key not in ctf_cache:
            ctf_cache[key] = ctf_sign(df)
            wedge_cache[key] = wedge_mask(44.0, az)
        S = ctf_cache[key]
        W = wedge_cache[key]
        P_all = np.abs(F * S) ** 2
        prof = np.bincount(BIN_IDX, weights=P_all.ravel(), minlength=N_RADIAL)
        prof = np.log1p(prof)
        prof = prof / prof.sum()
        P_in = np.abs(F * S * W) ** 2
        pfl = np.bincount(BIN_IDX, weights=P_in.ravel(), minlength=N_RADIAL)
        pfl = np.log1p(pfl)
        pfl = pfl / pfl.sum()
        # image domain from flipped + common-wedge spectrum
        vw = np.real(np.fft.ifftn(F * S * W)).astype(np.float32)
        vn = pervol_norm(vw)
        sd = spatial_dict(vn)
        cd = central_dict(vn)
        fill = float(W.mean())
        inout = float(np.log1p(P_in.sum()) - np.log1p((P_all * (1 - W)).sum()))
        row = (list(prof) + list(pfl)
               + [sd[k] for k in SPAT_KEYS] + [cd[k] for k in CENT_KEYS]
               + [fill, inout])
        X[i] = np.nan_to_num(np.array(row, dtype=np.float32))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    return X


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    tr = np.load(DATA / "train_volumes.npy", mmap_mode="r")
    te = np.load(DATA / "test_volumes.npy", mmap_mode="r")
    cc, wc = {}, {}
    print("featurizing train (wide)...")
    Xtr = featurize(tr, train, cc, wc)
    print("featurizing test (wide)...")
    Xte = featurize(te, test, cc, wc)
    np.savez_compressed(ROOT / "cache_v3.npz", X_train=Xtr, X_test=Xte,
                        names=np.array(NAMES))
    print("saved", ROOT / "cache_v3.npz", Xtr.shape, Xte.shape)


if __name__ == "__main__":
    main()
