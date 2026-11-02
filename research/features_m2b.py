"""M2b: append finer shell + per-bin z-anisotropy features to v1 cache.
Run: python research/features_m2b.py  (~4 min) then OOF eval inline.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

_ax = np.arange(32) - 15.5
XX, YY, ZZ = np.meshgrid(_ax, _ax, _ax, indexing="ij")
R_SPAT = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
KX = np.fft.fftfreq(32) * 32
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)
K_NORM = R_FREQ.copy()
K_NORM[R_FREQ == 0] = 1.0
ZCOS = np.abs(KZm) / K_NORM  # 1 = along beam
BINS = np.linspace(0.5, 16.0, 15)
BIN_IDX = np.clip(np.digitize(R_FREQ.ravel(), BINS) - 1, 0, 13)
ZCOS_F = ZCOS.ravel()
RINGS = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, 16)]
RING_MASKS = [((R_SPAT >= a) & (R_SPAT < b)) for a, b in RINGS]


def pervol_norm(v):
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    return (v - v.mean()) / (s + 1e-6)


def extra_feats(vn):
    out = []
    for m in RING_MASKS:  # shell means: open/closed differ at r 4-6
        out.append(vn[m].mean())
    P = np.abs(np.fft.fftn(vn)) ** 2
    Pf = P.ravel()
    for b in range(14):  # per-bin beam-vs-lateral power (wedge-aware shape)
        sel = BIN_IDX == b
        cone = sel & (ZCOS_F > 0.7)
        lat = sel & (ZCOS_F <= 0.7)
        c, l = Pf[cone].sum(), Pf[lat].sum()
        out.append(np.log1p(c) - np.log1p(l))
    return np.array(out, dtype=np.float32)


def featurize(vols):
    n = len(vols)
    X = np.zeros((n, 20), dtype=np.float32)
    for i in range(n):
        vn = pervol_norm(np.asarray(vols[i], dtype=np.float32))
        # NOTE: keep float64 intermediates inside helpers; cast at end
        X[i] = extra_feats(vn).astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    return X


def main():
    tr = np.load(DATA / "train_volumes.npy", mmap_mode="r")
    te = np.load(DATA / "test_volumes.npy", mmap_mode="r")
    print("extra train...")
    Etr = featurize(tr)
    print("extra test...")
    Ete = featurize(te)
    z = np.load(ROOT / "cache_m2.npz")
    Xtr = np.concatenate([z["X_train"], np.nan_to_num(Etr)], axis=1)
    Xte = np.concatenate([z["X_test"], np.nan_to_num(Ete)], axis=1)
    names = list(z["names"]) + [f"ring{i}" for i in range(6)] + [f"zan{i:02d}" for i in range(14)]
    np.savez_compressed(ROOT / "cache_m2b.npz", X_train=Xtr, X_test=Xte,
                        names=np.array(names))
    print("saved cache_m2b.npz", Xtr.shape, Xte.shape)


if __name__ == "__main__":
    main()
