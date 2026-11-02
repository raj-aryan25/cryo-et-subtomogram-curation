"""Phase 4: pseudo-validated conservative self-training + prior-offset check.

Experiment (all labels known, simulates deployment on shifted tomos):
 base = E3 MLP+override trained on base_idx; pseudo_idx acts as fake test.
 V1: base. V2: base + pseudo-labeled rows (p1>0.9, >0.8 rare, margin>0.3,
     cap 15%, per-class caps) retrained. V3: oracle (true pseudo labels).
 V4: base + clipped uniform-prior logit offsets (no EM).
Gate: adopt V2 procedure for real test only if V2-V1 >= +0.005 total AND no
class support collapse; report V4 separately.
Run: python research/phase4_selftrain.py (CPU, ~10 min)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
torch.manual_seed(42)
np.random.seed(42)
RARE = {"atpase_open", "atpase_closed", "unassigned_density"}
EPOCHS = 100


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


def fit_all(Xa, y, idx, epochs=EPOCHS):
    """E3 10-way + specialists. Returns (m10, sc10, specs dict)."""
    Xb = np.concatenate([Xa[k][idx] for k in ["clean", "blur05", "wedge49"]])
    yb = np.concatenate([y[idx]] * 3)
    m10, sc10 = fit_mlp(Xb, yb, 10, 64, epochs)
    yn = np.array([SPECIES[i] for i in y])
    specs = {}
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        m = np.isin(yn[idx], [oc, cc])
        Sb = np.concatenate([Xa[k][idx][m] for k in ["clean", "blur05", "wedge49"]])
        y2 = np.concatenate([(yn[idx][m] == oc).astype(int)] * 3)
        specs[(oc, cc)] = fit_mlp(Sb, y2, 2, 32, 150)
    return (m10, sc10), specs


def predict_all(models, specs, X, yn_true=None):
    (m10, sc10) = models
    P = probs(m10, sc10, X)
    yp = np.array(SPECIES)[P.argmax(1)].copy()
    S = {}
    for (oc, cc), (ms, scs) in specs.items():
        s = probs(ms, scs, X)[:, 1]
        S[(oc, cc)] = s
        use = np.isin(yp, [oc, cc])
        yp[use] = np.where(s[use] > 0.5, oc, cc)
    return yp, P, S


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    za = np.load(ROOT / "cache_aug.npz")
    z2 = np.load(ROOT / "cache_m2.npz")
    names = list(z2["names"])
    tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
    cc29 = [i for i, n in enumerate(names) if n not in tab]
    # E3 rows must be image-29: cache_aug was built on image-29 already?
    print("cache_aug keys:", list(za.keys()), za["clean"].shape)
    Xa = {k: za[k] for k in ["clean", "blur05", "wedge49"]}
    assert Xa["clean"].shape[1] == 29, Xa["clean"].shape

    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    print(f"base {len(base_idx)} pseudo {len(pseudo_idx)}")
    models, specs = fit_all(Xa, y, base_idx)
    Xp = Xa["clean"][pseudo_idx]
    yp, Pp, Sp = predict_all(models, specs, Xp)
    m1 = evaluate(yn[pseudo_idx], yp, Pp.max(1), pid[pseudo_idx])
    print(f"V1 base: total={m1['total']:.4f} macro={m1['macro_f1']:.3f} "
          f"sel={m1['selective_accuracy']:.3f}")

    # selection
    srt = np.sort(Pp, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    thr = np.array([0.8 if c in RARE else 0.9 for c in yp])
    keep = (Pp.max(1) > thr) & (margin > 0.3)
    # cap 15% + per-class cap (<=40% of kept from any single class)
    idx = np.where(keep)[0]
    order = np.argsort(-Pp.max(1)[idx])
    idx = idx[order[:max(1, int(0.15 * len(pseudo_idx)))]]
    from collections import Counter
    cnt = Counter(yp[idx])
    print("kept:", len(idx), dict(cnt))
    # per-class cap: drop lowest-confidence beyond 40%
    if len(idx):
        cap = max(1, int(0.4 * len(idx)))
        sel = []
        for c in SPECIES:
            ii = idx[yp[idx] == c]
            ii = ii[np.argsort(-Pp[ii].max(1))[:cap]]
            sel += list(ii)
        idx = np.array(sorted(sel))
    print("kept after cap:", len(idx), dict(Counter(yp[idx])))

    # V2: retrain with pseudo labels
    Xb2 = np.concatenate([Xa[k][base_idx] for k in ["clean", "blur05", "wedge49"]]
                         + [Xa["clean"][pseudo_idx[idx]]])
    yb2 = np.concatenate([y[base_idx]] * 3 + [np.array(
        [SPECIES.index(c) for c in yp[idx]])])
    m10b, sc10b = fit_mlp(Xb2, yb2, 10, 64, EPOCHS)
    # specialists refit with pseudo pair rows added
    yn2 = np.concatenate([yn[base_idx]] * 3 + [yp[idx]])
    Xf2 = np.concatenate([Xa[k][base_idx] for k in ["clean", "blur05", "wedge49"]]
                         + [Xa["clean"][pseudo_idx[idx]]])
    specs2 = {}
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        m = np.isin(yn2, [oc, cc])
        specs2[(oc, cc)] = fit_mlp(Xf2[m], (yn2[m] == oc).astype(int), 2, 32, 150)
    yp2, Pp2, _ = predict_all((m10b, sc10b), specs2, Xp)
    m2 = evaluate(yn[pseudo_idx], yp2, Pp2.max(1), pid[pseudo_idx])
    print(f"V2 selftrain: total={m2['total']:.4f} macro={m2['macro_f1']:.3f} "
          f"sel={m2['selective_accuracy']:.3f} (delta {m2['total']-m1['total']:+.4f})")

    # V4: clipped uniform-prior offsets on V1 logits
    prior = np.bincount(y[base_idx], minlength=10) / len(base_idx)
    off = np.clip(np.log((0.1 + 1e-9) / (prior + 1e-9)), -0.3, 0.3)
    P4 = Pp * np.exp(off)[None, :]
    P4 /= P4.sum(1, keepdims=True)
    yp4 = np.array(SPECIES)[P4.argmax(1)]
    m4 = evaluate(yn[pseudo_idx], yp4, P4.max(1), pid[pseudo_idx])
    print(f"V4 prior-offset: total={m4['total']:.4f} macro={m4['macro_f1']:.3f} "
          f"(delta {m4['total']-m1['total']:+.4f})")
    print("offsets:", dict(zip(SPECIES, off.round(2))))


if __name__ == "__main__":
    main()
