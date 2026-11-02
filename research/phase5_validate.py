"""Final-recipe validation (OOF 3-fold + pseudo-CD): E3-aug MLP + override +
clipped uniform offsets (0.15) + optional extreme-tomo tempering.
Tempering: w = prod of logistics centered just beyond train range, so it is
~1 in-distribution and discounts only CD-extreme tomos.
Run: python research/phase5_validate.py (CPU, ~15 min)
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
EPOCHS = 100
CLIP = 0.15


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, ncls, d_hid=64, epochs=EPOCHS):
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


def run_fold(Xa, y, yn, pid, tri, vai, temper):
    # E3 10-way
    Xb = np.concatenate([Xa[k][tri] for k in ["clean", "blur05", "wedge49"]])
    yb = np.concatenate([y[tri]] * 3)
    m10, sc10 = fit_mlp(Xb, yb, 10, 64, EPOCHS)
    P = probs(m10, sc10, Xa["clean"][vai])
    # uniform offsets from train-fold prior
    prior = np.bincount(y[tri], minlength=10) / len(tri)
    off = np.clip(np.log((0.1 + 1e-9) / (prior + 1e-9)), -CLIP, CLIP)
    P = P * np.exp(off)[None, :]
    P /= P.sum(1, keepdims=True)
    yp = np.array(SPECIES)[P.argmax(1)].copy()
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        mtr = np.isin(yn[tri], [oc, cc])
        mva = np.isin(yn[vai], [oc, cc])
        Sb = np.concatenate([Xa[k][tri][mtr] for k in ["clean", "blur05", "wedge49"]])
        y2 = np.concatenate([(yn[tri][mtr] == oc).astype(int)] * 3)
        ms, scs = fit_mlp(Sb, y2, 2, 32, 150)
        s = probs(ms, scs, Xa["clean"][vai][mva])[:, 1]
        vv = np.where(mva)[0]
        use = np.isin(yp, [oc, cc])
        for k, pos in enumerate(vv):
            if use[pos]:
                yp[pos] = oc if s[k] > 0.5 else cc
    return yp, P


def temper_weights(meta):
    def sig(x):
        return 1 / (1 + np.exp(-x))
    w = (sig((4.2 - meta["defocus_um"].values) / 0.5)
         * sig((170.0 - meta["ice_thickness_nm"].values) / 15.0)
         * sig((meta["tilt_max_deg"].values - 55.0) / 2.0))
    # rescale to mean 1 so in-distribution ranking is ~untouched
    return w / w.mean()


def eval_with(yp, P, yn, pid, meta_rows, temper):
    conf = P.max(1)
    if temper:
        conf = conf * temper_weights(meta_rows)
    return evaluate(yn, yp, conf, pid)


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    za = np.load(ROOT / "cache_aug.npz")
    Xa = {k: za[k] for k in ["clean", "blur05", "wedge49"]}
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    for temper in [False, True]:
        tots, sels = [], []
        YP = np.empty(len(y), dtype=object)
        PP = np.zeros((len(y), 10))
        for tri, vai in GroupKFold(3).split(Xa["clean"], y, groups):
            yp, P = run_fold(Xa, y, yn, pid, tri, vai, temper)
            YP[vai] = yp
            PP[vai] = P
        for tm in [False, True]:
            m = eval_with(YP, PP, yn, pid, train, tm)
            tag = "temper" if tm else "plain "
            print(f"final temper={tm}: OOF total={m['total']:.4f} "
                  f"macro={m['macro_f1']:.3f} conf={m['conformer_acc']:.3f} "
                  f"min={m['min_class_f1']:.3f} sel={m['selective_accuracy']:.4f}")
        # pseudo-CD with same recipe
        yp, P = run_fold(Xa, y, yn, pid, base_idx, pseudo_idx, temper)
        for tm in [False, True]:
            m = eval_with(yp, P, yn[pseudo_idx], pid[pseudo_idx],
                          train.iloc[pseudo_idx], tm)
            tag = "temper" if tm else "plain "
            print(f"  pseudo-CD {tag}: total={m['total']:.4f} sel={m['selective_accuracy']:.4f}")


if __name__ == "__main__":
    main()
