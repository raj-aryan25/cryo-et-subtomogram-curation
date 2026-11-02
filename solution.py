"""Subtomogram classification under a missing wedge.

Usage:
    python3 solution.py [public_dir] [submission_out]

Defaults to ./data and ./submissions/submission.csv.

Inputs (read-only, from <public_dir>): train.csv, test.csv, train_volumes.npy,
test_volumes.npy and species.json. Only the released columns of train.csv /
test.csv are used (volume_index, particle_id, tomogram_id, session,
defocus_um, tilt_max_deg, tilt_axis_azimuth_deg, ice_thickness_nm,
pixel_size_a, plus the training labels); class names and conformer pairs come
from species.json. No other data sources, no downloads, no network access;
runs fully offline with numpy / pandas / scikit-learn / scipy / torch.

Method:
  1. Per-volume normalization (1-99 percentile clip, then z-score), because
     every tomogram has its own gain and offset so raw voxels are not
     comparable between tomograms.
  2. Rotation-invariant image features per volume (74): log radial Fourier
     power profile (14 shells), spatial moments, shell masses and
     self-correlation shape cues (29), plus spherical-harmonic shell power
     for l=0..4 over six radial shells (30), within-shell std/skew (12) and
     inertia-tensor eigenvalue ratios (3). All are computed after the
     per-volume normalization so they are gain-invariant.
  3. A three-seed ensemble of small feed-forward networks (74 -> 64 -> 10)
     trained with class-weighted cross-entropy on the training split, on CUDA
     when available, otherwise CPU; plus one binary network (74 -> 32 -> 1)
     per conformer pair trained only on that pair's training rows. A pair
     network overrides the 10-way label only on rows the 10-way network
     already assigns inside the pair.
  4. Confidence is the ensemble maximum probability scaled by a fixed
     per-predicted-class calibration constant (ranking only) with a tiny
     row-order tie-break. particle_id values are copied verbatim from
     test.csv into the output and are never used as model inputs.

Output: CSV with exactly the columns particle_id, label, confidence, one row
per test particle in test.csv order.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import sph_harm_y
from sklearn.preprocessing import StandardScaler

N_RADIAL = 14
SEEDS = (7, 42, 123)
EPOCHS_10WAY = 200
EPOCHS_PAIR = 150
# Fixed per-class confidence calibration constants (fit once by greedy
# maximization of selective accuracy on cross-validated training
# predictions; bounded to [0.7, 1.3] so no class is silenced or dominant).
CONF_W = {
    "ribosome_60s": 1.15, "proteasome_20s": 1.0,
    "chaperonin_open": 0.85, "chaperonin_closed": 0.85,
    "atpase_open": 0.7, "atpase_closed": 0.7,
    "ferritin_shell": 1.15, "filament_seg": 1.3,
    "membrane_frag": 1.0, "unassigned_density": 0.85,
}

# Static 32^3 grids (built once).
_AX = np.arange(32) - 15.5
_XX, _YY, _ZZ = np.meshgrid(_AX, _AX, _AX, indexing="ij")
_R_SPAT = np.sqrt(_XX ** 2 + _YY ** 2 + _ZZ ** 2)
_RR = _R_SPAT.ravel()
_THETA = np.arccos(np.clip(_ZZ.ravel() / np.maximum(_RR, 1e-9), -1, 1))
_PHI = np.arctan2(_YY.ravel(), _XX.ravel())
_K = np.fft.fftfreq(32) * 32
_KXm, _KYm, _KZm = np.meshgrid(_K, _K, _K, indexing="ij")
_R_FREQ = np.sqrt(_KXm ** 2 + _KYm ** 2 + _KZm ** 2)
_BINS = np.linspace(0.5, 16.0, N_RADIAL + 1)
_BIN_IDX = np.clip(np.digitize(_R_FREQ.ravel(), _BINS) - 1, 0, N_RADIAL - 1)
_C_SPHERE = (_R_SPAT <= 6.0)
_MID_SHELL = (_R_SPAT > 6.0) & (_R_SPAT <= 11.0)
_OUT_SHELL = (_R_SPAT > 11.0)
_SHELLS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)]
_LMAX = 4


def _sh_basis():
    basis = []
    for (lo, hi) in _SHELLS:
        m = (_RR >= lo) & (_RR < hi)
        cols = []
        for l in range(_LMAX + 1):
            Y = sph_harm_y(l, np.arange(-l, l + 1)[:, None],
                           _THETA[m][None, :], _PHI[m][None, :])
            cols.append(Y[l].real)
            for mm in range(1, l + 1):
                cols.append(np.sqrt(2) * Y[l + mm].real)
                cols.append(np.sqrt(2) * Y[l - mm].imag)
        Q, _ = np.linalg.qr(np.stack(cols, axis=1))
        basis.append((m, Q.astype(np.float32)))
    return basis


_BASIS = _sh_basis()
_LSLICES = []
_s = 0
for _l in range(_LMAX + 1):
    _LSLICES.append(slice(_s, _s + 2 * _l + 1))
    _s += 2 * _l + 1


def load_taxonomy(public_dir):
    """Class names and conformer pairs exactly as listed in species.json."""
    with open(public_dir / "species.json") as f:
        spec = json.load(f)
    return list(spec["classes"]), [tuple(p) for p in spec["conformer_pairs"]]


def per_volume_norm(v):
    """Remove per-tomogram gain/offset: clip outliers, then z-score."""
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    return (v - v.mean()) / (s + 1e-6)


def featurize_one(vn):
    """74 image-only features for one normalized 32^3 volume."""
    P = np.abs(np.fft.fftn(vn)) ** 2
    prof = np.bincount(_BIN_IDX, weights=P.ravel(), minlength=N_RADIAL)
    prof = np.log1p(prof)
    prof = prof / prof.sum()
    v = vn.ravel()
    skew = float((vn ** 3).mean())
    kurt = float((vn ** 4).mean())
    f = [skew, kurt, float((vn > 2).mean()), float((vn < -2).mean()),
         float(np.percentile(vn, 99)), float(np.percentile(vn, 1)),
         float(vn[_C_SPHERE].mean()), float(vn[_MID_SHELL].mean()),
         float(vn[_OUT_SHELL].mean()),
         float(vn[_C_SPHERE].mean() - vn[_OUT_SHELL].mean())]
    f.append(float(vn.mean(axis=(0, 1)).var() / (
        vn.mean(axis=(0, 2)).var() + vn.mean(axis=(1, 2)).var() + 1e-9)))
    r = np.rot90(vn, 1, axes=(0, 1)).ravel()
    f.append(float(np.corrcoef(v, r)[0, 1]))
    for ax in (0, 1, 2):
        w = np.flip(vn, axis=ax).ravel()
        f.append(float(np.corrcoef(v, w)[0, 1]))
    out = list(prof) + f
    # Spherical-harmonic shell power + shell moments + inertia ratios.
    vd = np.asarray(vn, dtype=np.float64).ravel()
    for (m, Q) in _BASIS:
        x = vd[m]
        c = Q.T @ x
        tot = float(c @ c) + 1e-12
        for l in range(_LMAX + 1):
            sl = _LSLICES[l]
            out.append(float((c[sl] @ c[sl]) / tot))
        mu = x.mean()
        sd = x.std() + 1e-9
        out.append(float(x.std()))
        out.append(float((((x - mu) / sd) ** 3).mean()))
    w = np.clip(vd, 1.0, None) - 1.0
    tot = w.sum() + 1e-12
    C = np.stack([_XX.ravel(), _YY.ravel(), _ZZ.ravel()], axis=1)
    mu = (w[:, None] * C).sum(0) / tot
    D = C - mu[None, :]
    I = (w[:, None, None] * D[:, :, None] * D[:, None, :]).sum(0) / tot
    ev = np.linalg.eigvalsh(I)[::-1] + 1e-9
    out += [float(np.log(ev[0] / ev[1])), float(np.log(ev[1] / ev[2])),
            float(np.log(ev[0] / ev[2]))]
    return np.array(out, dtype=np.float64)


def featurize(vols):
    n = len(vols)
    X = np.zeros((n, N_RADIAL + 15 + 45), dtype=np.float32)
    for i in range(n):
        vn = per_volume_norm(np.asarray(vols[i], dtype=np.float32))
        X[i] = featurize_one(vn).astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f"  features {i + 1}/{n}", flush=True)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_predict(Xtr, ytr, Xte, ncls, d_hid, epochs, seeds, device):
    """Seed-ensemble of small networks with class-weighted loss. Trains on
    CUDA when available, else CPU. Returns mean test-class probabilities."""
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr).astype(np.float32)
    B = sc.transform(Xte).astype(np.float32)
    freq = np.bincount(ytr, minlength=ncls).astype(np.float64)
    cw = torch.tensor(len(ytr) / (ncls * freq), dtype=torch.float32,
                      device=device)
    XA = torch.tensor(A, device=device)
    YA = torch.tensor(ytr, dtype=torch.long, device=device)
    XB = torch.tensor(B, device=device)
    P = 0
    for sd in seeds:
        torch.manual_seed(sd)
        net = FeedForward(A.shape[1], d_hid, ncls).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(XA), YA, weight=cw)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            P = P + torch.softmax(net(XB), 1).float().cpu().numpy()
    return P / len(seeds)


def main():
    args = sys.argv[1:]
    public_dir = Path(args[0]) if len(args) > 0 else Path("./data")
    submission_out = Path(args[1]) if len(args) > 1 else Path("./submissions/submission.csv")
    print("public_dir:", public_dir, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    SPECIES, PAIRS = load_taxonomy(public_dir)
    CLS = {c: i for i, c in enumerate(SPECIES)}
    cw = np.array([CONF_W[c] for c in SPECIES])

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    tr_vols = np.load(public_dir / "train_volumes.npy", mmap_mode="r")
    te_vols = np.load(public_dir / "test_volumes.npy", mmap_mode="r")
    assert len(train) == len(tr_vols) and len(test) == len(te_vols)

    print("featurizing train...", flush=True)
    Xtr = featurize(tr_vols)
    print("featurizing test...", flush=True)
    Xte = featurize(te_vols)
    assert Xtr.shape[1] == 74 and Xte.shape[1] == 74
    y = train["label"].map(CLS).values

    # 10-way head: three-seed ensemble.
    print("training 10-way ensemble + specialists...", flush=True)
    P10 = fit_predict(Xtr, y, Xte, len(SPECIES), 64, EPOCHS_10WAY, SEEDS,
                      device)
    yp = np.array(SPECIES)[P10.argmax(1)].copy()

    # Conformer specialists, trained only on their pair's training rows;
    # they override the 10-way label only inside the pair.
    yn = train["label"].values
    for open_c, closed_c in PAIRS:
        m = np.isin(yn, [open_c, closed_c])
        s = fit_predict(Xtr[m], (yn[m] == open_c).astype(int), Xte, 2, 32,
                        EPOCHS_PAIR, SEEDS, device)[:, 1]
        use = np.isin(yp, [open_c, closed_c])
        yp[use] = np.where(s[use] > 0.5, open_c, closed_c)

    # Guard against a degenerate prediction set; fall back to the 10-way head.
    dist = Counter(yp)
    missing = [c for c in SPECIES if dist.get(c, 0) == 0]
    dominant = [c for c, n in dist.items() if n > 0.5 * len(yp)]
    if missing or dominant:
        print(f"fallback to 10-way: missing={missing} dominant={dominant}")
        yp = np.array(SPECIES)[P10.argmax(1)]
    print("pred distribution:", dict(sorted(Counter(yp).items())), flush=True)

    # Ranking-only confidence: calibrated maximum probability plus a
    # row-order tie-break.
    pi = np.array([CLS[c] for c in yp])
    conf = np.clip(P10.max(1) * cw[pi], 1e-3, 1 - 1e-3)
    conf = conf + np.arange(len(yp), dtype=np.float64) / max(1, len(yp)) * 1e-6

    sub = pd.DataFrame({"particle_id": test["particle_id"].values,
                        "label": yp, "confidence": conf})
    assert len(sub) == len(test) and sub["particle_id"].is_unique
    assert (sub["particle_id"].values == test["particle_id"].values).all()
    assert set(sub["label"]).issubset(set(SPECIES))
    assert sub["confidence"].between(0, 1).all() and sub["confidence"].notna().all()
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_out, index=False)
    print("wrote", submission_out, flush=True)


if __name__ == "__main__":
    main()
