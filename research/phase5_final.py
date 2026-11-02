"""Decisive head-to-head at locked epochs=400 (identical code path):
 R1 clean/no-offsets | R2 E3/offsets0.15 | R3 clean/offsets0.15 | R4 E3/no-offsets.
Full OOF 3-fold + pseudo-CD. Winner becomes the shipped recipe.
Run: python research/phase5_final.py (CPU, ~25 min)
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
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
torch.manual_seed(42)
np.random.seed(42)
EPOCHS = 400
CLIP = 0.15


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, ncls, d_hid, epochs):
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


def run_recipe(Xa, y, yn, pid, tri, vai, use_aug, use_off):
    keys = ["clean", "blur05", "wedge49"] if use_aug else ["clean"]
    Xb = np.concatenate([Xa[k][tri] for k in keys])
    yb = np.concatenate([y[tri]] * len(keys))
    m10, sc10 = fit_mlp(Xb, yb, 10, 64, EPOCHS)
    P = probs(m10, sc10, Xa["clean"][vai])
    if use_off:
        prior = np.bincount(y[tri], minlength=10) / len(tri)
        off = np.clip(np.log((0.1 + 1e-9) / (prior + 1e-9)), -CLIP, CLIP)
        P = P * np.exp(off)[None, :]
        P /= P.sum(1, keepdims=True)
    yp = np.array(SPECIES)[P.argmax(1)].copy()
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        mtr = np.isin(yn[tri], [oc, cc])
        mva = np.isin(yn[vai], [oc, cc])
        Sb = np.concatenate([Xa[k][tri][mtr] for k in keys])
        y2 = np.concatenate([(yn[tri][mtr] == oc).astype(int)] * len(keys))
        ms, scs = fit_mlp(Sb, y2, 2, 32, 150)
        s = probs(ms, scs, Xa["clean"][vai][mva])[:, 1]
        vv = np.where(mva)[0]
        use = np.isin(yp, [oc, cc])
        for k, pos in enumerate(vv):
            if use[pos]:
                yp[pos] = oc if s[k] > 0.5 else cc
    return evaluate(yn[vai], yp, P.max(1), pid[vai])


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    za = np.load(ROOT / "cache_aug.npz")
    Xa = {k: za[k] for k in ["clean", "blur05", "wedge49"]}
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    recipes = {"R1_clean": (False, False), "R2_E3_off": (True, True),
               "R3_clean_off": (False, True), "R4_E3": (True, False)}
    for name, (aug, off) in recipes.items():
        tots = []
        comp = []
        for tri, vai in GroupKFold(3).split(Xa["clean"], y, groups):
            m = run_recipe(Xa, y, yn, pid, tri, vai, aug, off)
            tots.append(m["total"])
            comp.append((m["macro_f1"], m["conformer_acc"], m["min_class_f1"],
                         m["selective_accuracy"]))
        comp = np.mean(comp, axis=0)
        mp = run_recipe(Xa, y, yn, pid, base_idx, pseudo_idx, aug, off)
        print(f"{name:<12} OOF total={np.mean(tots):.4f} "
              f"(macro={comp[0]:.3f} conf={comp[1]:.3f} min={comp[2]:.3f} "
              f"sel={comp[3]:.3f}) | pseudo total={mp['total']:.4f} "
              f"(macro={mp['macro_f1']:.3f} sel={mp['selective_accuracy']:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
