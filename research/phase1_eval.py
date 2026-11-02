"""Phase 1 gate: candidate feature sets x {OOF 3-fold, pseudo-CD} totals.
Uses shipped MLP head (64 hidden, 120 epochs) + plain specialist override.
Run: python research/phase1_eval.py (CPU, ~10 min)
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
from splits import load_meta, group_folds, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
torch.manual_seed(42)
np.random.seed(42)

SETS = {
    "S0_v2base": ("cache_m2", None),
    "S1_rawrad_wspat": ("cache_v3", list(range(0, 14)) + list(range(28, 43))),
    "S2_pflip_wspat": ("cache_v3", list(range(14, 28)) + list(range(28, 43))),
    "S3_S2_central": ("cache_v3", list(range(14, 28)) + list(range(28, 51))),
    "S4_all_noconf": ("cache_v3", list(range(0, 51))),
}


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, Xte, ncls, d_hid=64, epochs=120):
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
        P = torch.softmax(net(torch.tensor(B)), 1).numpy()
    return P


def run_split(X, y, yn, pid, tri, vai, epochs=120):
    P = fit_mlp(X[tri], y[tri], X[vai], 10, epochs=epochs)
    yp = np.array(SPECIES)[P.argmax(1)]
    # plain specialist override (binary MLPs, C->torch hid32/150ep)
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        mtr = np.isin(yn[tri], [oc, cc])
        mva = np.isin(yn[vai], [oc, cc])
        if mtr.sum() == 0 or mva.sum() == 0:
            continue
        s = fit_mlp(X[tri][mtr], (yn[tri][mtr] == oc).astype(int),
                    X[vai][mva], 2, d_hid=32, epochs=150)[:, 1]
        full = np.empty(mva.sum(), dtype=object)
        full[:] = np.where(s > 0.5, oc, cc)
        use = np.isin(yp, [oc, cc])
        # map vai-positions: only rows both predicted-in-pair AND truly in pair
        vv = np.where(mva)[0]
        for k, pos in enumerate(vv):
            if use[pos]:
                yp[pos] = full[k]
    m = evaluate(yn[vai], yp, P.max(1), pid[vai])
    return m


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    caches = {}
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    print("pseudo-CD holdout tomos:", len(hold))
    for name, (cache, cols) in SETS.items():
        if cache not in caches:
            z = np.load(ROOT / f"{cache}.npz")
            caches[cache] = z["X_train"]
        X = caches[cache] if cols is None else caches[cache][:, cols]
        if cols is None:  # v2: image-only 29 like shipped solution
            zn = np.load(ROOT / "cache_m2.npz")
            nnames = list(zn["names"])
            tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos",
                   "ice_thickness_nm"]
            cc = [i for i, n in enumerate(nnames) if n not in tab]
            X = zn["X_train"][:, cc]
        # OOF
        tots, macros, confs, mins, sels = [], [], [], [], []
        for tri, vai in GroupKFold(3).split(X, y, groups):
            m = run_split(X, y, yn, pid, tri, vai, epochs=100)
            tots.append(m["total"])
            macros.append(m["macro_f1"])
            confs.append(m["conformer_acc"])
            mins.append(m["min_class_f1"])
            sels.append(m["selective_accuracy"])
        # pseudo-CD
        mp = run_split(X, y, yn, pid, base_idx, pseudo_idx, epochs=100)
        print(f"{name:<16} OOF total={np.mean(tots):.4f} "
              f"(macro={np.mean(macros):.3f} conf={np.mean(confs):.3f} "
              f"min={np.mean(mins):.3f} sel={np.mean(sels):.3f}) | "
              f"pseudo-CD total={mp['total']:.4f} (macro={mp['macro_f1']:.3f} "
              f"sel={mp['selective_accuracy']:.3f})", flush=True)


if __name__ == "__main__":
    main()
