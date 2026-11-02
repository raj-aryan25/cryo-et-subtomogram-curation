"""Phase 1b: bin ablation + spectral augmentation (no CTF assumptions).
Builds cache_aug.npz: image-29 feats of train vols under {clean, blur05, wedge49}.
Then OOF/pseudo-CD for: baseline / drop-fragile-bins / augmented-train / both.
Run: python research/phase1_aug.py (CPU, ~15 min)
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
from features_m2 import pervol_norm, radial_profile, spatial_feats, FEAT_NAMES
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
TABULAR = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
COLS29 = [i for i, n in enumerate(FEAT_NAMES) if n not in TABULAR]
KX = np.fft.fftfreq(32) * 32
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)
torch.manual_seed(42)
np.random.seed(42)


def wedge49_mask(az_deg):
    phi = np.deg2rad(az_deg)
    kn = -KXm * np.sin(phi) + KYm * np.cos(phi)
    edge = np.tan(np.deg2rad(49.0)) * np.abs(kn)
    return ((np.abs(KZm) <= edge) | (R_FREQ < 1.5)).astype(np.float32)


def feats29(vols):
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


def fit_mlp(Xtr, ytr, Xte, ncls, d_hid=64, epochs=100):
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr).astype(np.float32)
    B = sc.transform(Xte).astype(np.float32)
    freq = np.bincount(ytr, minlength=ncls).astype(np.float64)
    cw = torch.tensor(len(ytr) / (ncls * freq), dtype=torch.float32)
    net = MLP(A.shape[1], d_hid, ncls)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    XA, YA = torch.tensor(A), torch.tensor(ytr, dtype=torch.long)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(XA), YA, weight=cw)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.tensor(B)), 1).numpy()


def eval_split(y, yn, pid, tri, vai, drop_bins, augment, Xa_dict):
    cols = [c for c in range(29) if c not in drop_bins]
    if augment is None:
        Ptr = Xa_dict["clean"][tri][:, cols]
        ytr = y[tri]
    else:
        Ptr = np.concatenate([Xa_dict[k][tri][:, cols] for k in augment])
        ytr = np.concatenate([y[tri] for _ in augment])
    Pva = fit_mlp(Ptr, ytr, Xa_dict["clean"][vai][:, cols], 10)
    yp = np.array(SPECIES)[Pva.argmax(1)]
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        mtr = np.isin(yn[tri], [oc, cc])
        mva = np.isin(yn[vai], [oc, cc])
        if mtr.sum() == 0 or mva.sum() == 0:
            continue
        if augment is None:
            Str, yb = Ptr[mtr], (yn[tri][mtr] == oc).astype(int)
        else:
            Str = np.concatenate([Xa_dict[k][tri][mtr][:, cols] for k in augment])
            yb = np.concatenate([(yn[tri][mtr] == oc).astype(int) for _ in augment])
        s = fit_mlp(Str, yb, Xa_dict["clean"][vai][mva][:, cols], 2,
                    d_hid=32, epochs=150)[:, 1]
        vv = np.where(mva)[0]
        use = np.isin(yp, [oc, cc])
        for k, pos in enumerate(vv):
            if use[pos]:
                yp[pos] = oc if s[k] > 0.5 else cc
    return evaluate(yn[vai], yp, Pva.max(1), pid[vai])


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    tr = np.load(ROOT.parent / "data" / "train_volumes.npy", mmap_mode="r")
    az = train["tilt_axis_azimuth_deg"].values
    rng = np.random.default_rng(0)

    cache = ROOT / "cache_aug.npz"
    if cache.exists():
        z = np.load(cache)
        Xa = {k: z[k] for k in ["clean", "blur05", "wedge49"]}
        print("loaded cache_aug.npz")
    else:
        print("featurizing clean...")
        Xa = {"clean": feats29([tr[i] for i in range(len(tr))])}
        print("featurizing blur05...")
        Xa["blur05"] = feats29([gaussian_filter(np.asarray(tr[i], dtype=np.float32), 0.5)
                                for i in range(len(tr))])
        print("featurizing wedge49...")
        vols_w = []
        for i in range(len(tr)):
            v = np.asarray(tr[i], dtype=np.float32)
            F = np.fft.fftn(v)
            vols_w.append(np.real(np.fft.ifftn(F * wedge49_mask(az[i]))).astype(np.float32))
            if (i + 1) % 1000 == 0:
                print(f"  wedge {i + 1}/{len(tr)}", flush=True)
        Xa["wedge49"] = feats29(vols_w)
        np.savez_compressed(cache, **Xa)
        print("saved cache_aug.npz")

    # fragile bins from DIAG0: rad04,rad05,rad07,rad08 (+rad03,rad06 milder)
    variants = {
        "E1_base": ([], None),
        "E2_drop4678": ([4, 6, 7, 8], None),
        "E3_aug3x": ([], ["clean", "blur05", "wedge49"]),
        "E4_drop_aug": ([4, 6, 7, 8], ["clean", "blur05", "wedge49"]),
    }
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    for name, (drop, aug) in variants.items():
        tots = []
        for tri, vai in GroupKFold(3).split(Xa["clean"], y, groups):
            m = eval_split(y, yn, pid, tri, vai, drop, aug, Xa)
            tots.append(m["total"])
        mp = eval_split(y, yn, pid, base_idx, pseudo_idx, drop, aug, Xa)
        # recompute component means for the OOF folds
        print(f"{name:<12} OOF total={np.mean(tots):.4f} | "
              f"pseudo-CD total={mp['total']:.4f} (macro={mp['macro_f1']:.3f} "
              f"sel={mp['selective_accuracy']:.3f})", flush=True)


if __name__ == "__main__":
    main()
