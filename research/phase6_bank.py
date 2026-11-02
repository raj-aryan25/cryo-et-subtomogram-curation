"""Workstream A: richer rotation-invariant bank (no tabular, gain-invariant).
New per volume (post per-vol norm):
  SHELL_L (6 shells x l=0..4 power = 30): rotation-invariant angular spectrum
  SHELLS (6 shells x {std, skew} = 12): within-shell shape
  INERTIA (3): log eigenvalue ratios of thresholded-mass inertia tensor
Cache: research/cache_bank.npz (new feats only).
Run: python research/phase6_bank.py (CPU, ~10 min)
"""
import numpy as np
from pathlib import Path
from scipy.special import sph_harm_y

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

N = 32
_ax = np.arange(N) - 15.5
XX, YY, ZZ = np.meshgrid(_ax, _ax, _ax, indexing="ij")
R_SPAT = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
RR = R_SPAT.ravel()
THETA = np.arccos(np.clip(ZZ.ravel() / np.maximum(RR, 1e-9), -1, 1))
PHI = np.arctan2(YY.ravel(), XX.ravel())
SHELLS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)]
LMAX = 4

# Precompute real-SH basis per shell: list over shells of (n_vox, n_basis)
print("building SH basis...", flush=True)
BASIS = []
for (lo, hi) in SHELLS:
    m = (RR >= lo) & (RR < hi)
    cols = []
    for l in range(LMAX + 1):
        Y = sph_harm_y(l, np.arange(-l, l + 1)[:, None],
                       THETA[m][None, :], PHI[m][None, :])  # (2l+1, n)
        # real combinations: m=0 real; m>0: sqrt2*Re, sqrt2*Im
        cols.append(Y[l].real)
        for mm in range(1, l + 1):
            cols.append(np.sqrt(2) * Y[l + mm].real)
            cols.append(np.sqrt(2) * Y[l - mm].imag)
    B = np.stack(cols, axis=1)  # (n_vox, 25)
    # orthonormalize within shell for stable power split
    Q, _ = np.linalg.qr(B)
    BASIS.append((m, Q.astype(np.float32)))
NB = sum(2 * l + 1 for l in range(LMAX + 1))
LSLICES = []
s = 0
for l in range(LMAX + 1):
    LSLICES.append(slice(s, s + 2 * l + 1))
    s += 2 * l + 1

NAMES = ([f"sh_s{r}_l{l}" for r in range(len(SHELLS)) for l in range(LMAX + 1)]
         + [f"shs_s{r}_{k}" for r in range(len(SHELLS)) for k in ("std", "skew")]
         + ["in_e12", "in_e23", "in_e13"])


def pervol_norm(v):
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    return (v - v.mean()) / (s + 1e-6)


def bank_one(vn):
    v = np.asarray(vn, dtype=np.float64).ravel()
    out = []
    for (m, Q) in BASIS:
        x = v[m]
        c = Q.T @ x
        tot = float(c @ c) + 1e-12
        for l in range(LMAX + 1):
            sl = LSLICES[l]
            out.append(float((c[sl] @ c[sl]) / tot))
        xs = x
        out.append(float(xs.std()))
        mu = xs.mean()
        sd = xs.std() + 1e-9
        out.append(float((((xs - mu) / sd) ** 3).mean()))
    # inertia tensor of positive mass
    w = np.clip(vn.astype(np.float64), 1.0, None) - 1.0
    W = w.ravel()
    tot = W.sum() + 1e-12
    C = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    mu = (W[:, None] * C).sum(0) / tot
    D = C - mu[None, :]
    I = (W[:, None, None] * D[:, :, None] * D[:, None, :]).sum(0) / tot
    ev = np.linalg.eigvalsh(I)[::-1] + 1e-9  # desc
    out += [float(np.log(ev[0] / ev[1])), float(np.log(ev[1] / ev[2])),
            float(np.log(ev[0] / ev[2]))]
    return np.array(out, dtype=np.float32)


def featurize(vols):
    n = len(vols)
    X = np.zeros((n, len(NAMES)), dtype=np.float32)
    for i in range(n):
        vn = pervol_norm(np.asarray(vols[i], dtype=np.float32))
        X[i] = bank_one(vn)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    tr = np.load(DATA / "train_volumes.npy", mmap_mode="r")
    te = np.load(DATA / "test_volumes.npy", mmap_mode="r")
    print("bank train...", flush=True)
    Xtr = featurize(tr)
    print("bank test...", flush=True)
    Xte = featurize(te)
    np.savez_compressed(ROOT / "cache_bank.npz", X_train=Xtr, X_test=Xte,
                        names=np.array(NAMES))
    print("saved", ROOT / "cache_bank.npz", Xtr.shape, Xte.shape)


if __name__ == "__main__":
    main()
