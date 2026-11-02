"""Ensemble + TTA-mean OOF test (R1 recipe). Caches train shift views.
Compares: single-seed | 3-seed ensemble | 3-seed + TTA-mean(probs).
Run: python research/phase5_ens.py (CPU, ~25 min incl. shift featurization)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta
from features_m2 import pervol_norm, radial_profile, spatial_feats, FEAT_NAMES
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
TABULAR = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
EPOCHS = 200


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, ncls, d_hid, epochs, seed):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr).astype(np.float32)
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
    return net.eval(), sc


@torch.no_grad()
def probs(net, sc, X):
    return torch.softmax(net(torch.tensor(
        sc.transform(X).astype(np.float32))), 1).numpy()


def shift_view(v, dx, dy, dz):
    p = np.pad(np.asarray(v, dtype=np.float32), 2)
    return p[2 + dx:34 + dx, 2 + dy:34 + dy, 2 + dz:34 + dz].copy()


def feats29(vols):
    n = len(vols)
    X = np.zeros((n, 29), dtype=np.float32)
    for i, vv in enumerate(vols):
        vn = pervol_norm(np.asarray(vv, dtype=np.float32))
        prof = radial_profile(vn)
        s = spatial_feats(vn)
        row = list(prof) + [s[k] for k in FEAT_NAMES[14:29]]
        X[i] = np.nan_to_num(np.array(row, dtype=np.float32))
        if (i + 1) % 1000 == 0:
            print(f"  feat {i + 1}/{n}", flush=True)
    return X


def train_predict(Xtr_f, ytr, Xte_views, seeds, d_hid=64, epochs=EPOCHS,
                  ncls=10, tta_mean=False):
    """Xte_views: list of feature matrices (same rows). Returns mean probs."""
    P = 0
    for sd in seeds:
        m, sc = fit_mlp(Xtr_f, ytr, ncls, d_hid, epochs, sd)
        Pv = np.stack([probs(m, sc, Xv) for Xv in Xte_views])
        P = P + (Pv.mean(0) if tta_mean else Pv[0])
    return P / len(seeds)


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    tr = np.load(ROOT.parent / "data" / "train_volumes.npy", mmap_mode="r")
    cache = ROOT / "cache_ttav.npz"
    if cache.exists():
        z = np.load(cache)
        Xshx, Xshz = z["Xshx"], z["Xshz"]
        print("loaded shift cache")
    else:
        print("shift view +x...")
        Xshx = feats29([shift_view(tr[i], 2, 0, 0) for i in range(len(tr))])
        print("shift view -z...")
        Xshz = feats29([shift_view(tr[i], 0, 0, -2) for i in range(len(tr))])
        np.savez_compressed(cache, Xshx=Xshx, Xshz=Xshz)
    za = np.load(ROOT / "cache_aug.npz")
    print("views ready")
    views_full = {"clean": za["clean"], "wedge49": za["wedge49"],
                  "blur05": za["blur05"], "shx": Xshx, "shz": Xshz}
    SEEDS = [7, 42, 123]
    res = {}
    # Note: override uses single-seed specialists for speed in this screen;
    # final adopts ensemble everywhere if ensemble wins.
    for name, seeds, tta in [("single", [42], False), ("ens3", SEEDS, False),
                             ("ens3_tta", SEEDS, True)]:
        YP = np.empty(len(y), dtype=object)
        PP = np.zeros((len(y), 10))
        for tri, vai in GroupKFold(3).split(views_full["clean"], y, groups):
            Pv_list = [views_full[k][vai] for k in
                       (["clean", "wedge49", "blur05", "shx", "shz"] if tta else ["clean"])]
            P = train_predict(views_full["clean"][tri], y[tri], Pv_list,
                              seeds, tta_mean=tta)
            PP[vai] = P
            yp = np.array(SPECIES)[P.argmax(1)].copy()
            for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                           ("atpase_open", "atpase_closed")]:
                mtr = np.isin(yn[tri], [oc, cc])
                mva = np.isin(yn[vai], [oc, cc])
                Sm = train_predict(views_full["clean"][tri][mtr],
                                   (yn[tri][mtr] == oc).astype(int),
                                   [views_full["clean"][vai][mva]], [42],
                                   d_hid=32, epochs=150, ncls=2)
                s = Sm[:, 1]
                vv = np.where(mva)[0]
                use = np.isin(yp, [oc, cc])
                for k, pos in enumerate(vv):
                    if use[pos]:
                        yp[pos] = oc if s[k] > 0.5 else cc
            YP[vai] = yp
        m = evaluate(yn, YP, PP.max(1), pid)
        res[name] = m["total"]
        print(f"{name:<10} OOF total={m['total']:.4f} macro={m['macro_f1']:.3f} "
              f"conf={m['conformer_acc']:.3f} min={m['min_class_f1']:.3f} "
              f"sel={m['selective_accuracy']:.4f}", flush=True)
    print(res)


if __name__ == "__main__":
    main()
