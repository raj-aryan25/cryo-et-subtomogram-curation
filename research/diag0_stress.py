"""Phase 0b: synthetic stress slopes + TTA disagreement (train labels only for
stress; test used unlabeled for disagreement rates).

Writes: research/diag0_stress.json. Run: python research/diag0_stress.py (CPU).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta
from features_m2 import pervol_norm, radial_profile, spatial_feats, FEAT_NAMES
from metrics import SPECIES

ROOT = Path(__file__).resolve().parent
TABULAR = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
torch.manual_seed(42)
np.random.seed(42)

KX = np.fft.fftfreq(32) * 32
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)


def wedge_mask(tilt_deg, az_deg):
    """1 = observed. Double wedge of half-angle 90-tilt around z, extruded
    along tilt axis t=(cosφ,sinφ,0); central ball always observed."""
    phi = np.deg2rad(az_deg)
    kn = -KXm * np.sin(phi) + KYm * np.cos(phi)
    edge = np.tan(np.deg2rad(tilt_deg)) * np.abs(kn)
    return ((np.abs(KZm) <= edge) | (R_FREQ < 1.5)).astype(np.float32)


def apply_wedge(v, tilt_deg, az_deg):
    F = np.fft.fftn(v.astype(np.float32))
    v2 = np.real(np.fft.ifftn(F * wedge_mask(tilt_deg, az_deg)))
    return v2.astype(np.float32)


def paste_neighbour(v, donor, rng):
    v = v.copy()
    s = 10
    face = rng.integers(0, 6)
    d = donor - donor.mean()
    patch = d[:s, :s, :s] if face == 0 else d[-s:, -s:, -s:]
    if face == 0:
        v[:s, :s, :s] += 0.7 * patch
    elif face == 1:
        v[-s:, -s:, -s:] += 0.7 * patch
    elif face == 2:
        v[:s, -s:, :s] += 0.7 * patch
    elif face == 3:
        v[-s:, :s, :s] += 0.7 * patch
    elif face == 4:
        v[:s, :s, -s:] += 0.7 * patch
    else:
        v[-s:, -s:, :s] += 0.7 * patch
    return v


def feats_of(vols):
    n = len(vols)
    X = np.zeros((n, 29), dtype=np.float32)
    for i, vv in enumerate(vols):
        vn = pervol_norm(np.asarray(vv, dtype=np.float32))
        prof = radial_profile(vn)
        s = spatial_feats(vn)
        row = list(prof) + [s[k] for k in FEAT_NAMES[14:29]]
        X[i] = np.nan_to_num(np.array(row, dtype=np.float32))
    return X


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def train_full_mlp(Xtr, y, d_hid=64, epochs=120):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr).astype(np.float32)
    freq = np.bincount(y, minlength=10).astype(np.float64)
    cw = torch.tensor(len(y) / (10 * freq), dtype=torch.float32)
    net = MLP(A.shape[1], d_hid, 10)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    XA = torch.tensor(A)
    YA = torch.tensor(y, dtype=torch.long)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(XA), YA, weight=cw)
        loss.backward()
        opt.step()
    return net.eval(), sc


@torch.no_grad()
def predict(net, sc, X):
    P = torch.softmax(net(torch.tensor(sc.transform(X).astype(np.float32))), 1)
    return P.numpy()


def main():
    train, test = load_meta()
    tr = np.load(ROOT.parent / "data" / "train_volumes.npy", mmap_mode="r")
    te = np.load(ROOT.parent / "data" / "test_volumes.npy", mmap_mode="r")
    y_all = np.array([SPECIES.index(c) for c in train["label"].values])
    print("featurizing full train for model fit...")
    Xtr_full = feats_of([tr[i] for i in range(len(tr))])
    net, sc = train_full_mlp(Xtr_full, y_all)
    особый_acc = (predict(net, sc, Xtr_full).argmax(1) == y_all).mean()
    print(f"in-sample train acc: {особый_acc:.3f} (optimistic reference)")

    # stratified subset for stress
    rng = np.random.default_rng(0)
    sub_idx = []
    for c in range(10):
        idx = np.where(y_all == c)[0]
        sub_idx += list(rng.choice(idx, 24, replace=False))
    sub_idx = np.array(sub_idx)
    y_sub = y_all[sub_idx]
    az = train["tilt_axis_azimuth_deg"].values[sub_idx]
    vols = [np.asarray(tr[i], dtype=np.float32) for i in sub_idx]
    donors = [np.asarray(tr[i], dtype=np.float32)
              for i in rng.choice(len(tr), len(vols))]

    variants = {"clean": vols,
                "wedge49": [apply_wedge(v, 49.0, a) for v, a in zip(vols, az)],
                "blur05": [gaussian_filter(v, 0.5) for v in vols],
                "blur10": [gaussian_filter(v, 1.0) for v in vols],
                "crowd1": [paste_neighbour(v, d, rng) for v, d in zip(vols, donors)]}
    res = {}
    for name, vv in variants.items():
        X = feats_of(vv)
        acc = float((predict(net, sc, X).argmax(1) == y_sub).mean())
        res[name] = acc
        print(f"  {name:<8} acc={acc:.3f}  (drop {особый_acc - acc:+.3f})")

    # TTA disagreement: 4 axis-flip views; train-subset vs test-subset
    def tta_disagree(vol_list):
        views = []
        for v in vol_list:
            vv = np.asarray(v, dtype=np.float32)
            views.append([vv, np.flip(vv, 0).copy(), np.flip(vv, 1).copy(),
                          np.flip(vv, 2).copy()])
        P = np.stack([predict(net, sc, feats_of([w[i] for w in views]))
                      for i in range(4)])
        pred = P.argmax(2)  # (4, n)
        return float((pred != pred[0]).any(0).mean()), P

    d_tr, _ = tta_disagree(vols[:150])
    te_idx = np.arange(0, len(te), max(1, len(te) // 150))[:150]
    d_te, P_te = tta_disagree([te[i] for i in te_idx])
    print(f"TTA disagreement: train-subset={d_tr:.3f} test-subset={d_te:.3f}")
    res["tta_dis_train"] = d_tr
    res["tta_dis_test"] = d_te
    # mean maxprob on the same subsets
    Ptr = predict(net, sc, feats_of(vols[:150]))
    Pte = P_te.mean(0)
    print(f"mean maxprob: train-subset={Ptr.max(1).mean():.3f} "
          f"test-subset={Pte.max(1).mean():.3f}")
    res["maxprob_train_sub"] = float(Ptr.max(1).mean())
    res["maxprob_test_sub"] = float(Pte.max(1).mean())
    with open(ROOT / "diag0_stress.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote research/diag0_stress.json")


if __name__ == "__main__":
    main()
