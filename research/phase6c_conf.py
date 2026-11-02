"""Workstream C: 74-feat OOF (saved) + class-weight/hinge confidence tuning.
Saves research/oof74.npz (probs, pred, correct). Reports selective for:
maxprob | +class_w (greedy) | +hinge trust | +both, on OOF and pseudo-CD.
Run: python research/phase6c_conf.py (CPU, ~20 min)
"""
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
np.random.seed(42)
SEEDS = [7, 42, 123]


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp_ens(Xtr, ytr, Xte, ncls, d_hid, epochs, seeds):
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr).astype(np.float32)
    B = sc.transform(Xte).astype(np.float32)
    freq = np.bincount(ytr, minlength=ncls).astype(np.float64)
    cw = torch.tensor(len(ytr) / (ncls * freq), dtype=torch.float32)
    XA = torch.tensor(A)
    YA = torch.tensor(ytr, dtype=torch.long)
    XB = torch.tensor(B)
    P = 0
    for sd in seeds:
        torch.manual_seed(sd)
        net = MLP(A.shape[1], d_hid, ncls)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(XA), YA, weight=cw)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            P = P + torch.softmax(net(XB), 1).numpy()
    return P / len(seeds)


def spec_ens(Xtr, yb, Xte, seeds):
    P = 0
    for sd in seeds:
        torch.manual_seed(sd)
        sc = StandardScaler().fit(Xtr)
        A = sc.transform(Xtr).astype(np.float32)
        B = sc.transform(Xte).astype(np.float32)
        net = MLP(A.shape[1], 32, 2)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        XA = torch.tensor(A)
        YA = torch.tensor(yb, dtype=torch.long)
        net.train()
        for _ in range(150):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(XA), YA)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            P = P + torch.softmax(net(torch.tensor(B)), 1).numpy()[:, 1]
    return P / len(seeds)


def main():
    train, test = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    z2 = np.load(ROOT / "cache_m2.npz")
    names = list(z2["names"])
    tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
    cc = [i for i, n in enumerate(names) if n not in tab]
    zb = np.load(ROOT / "cache_bank.npz")
    X = np.concatenate([z2["X_train"][:, cc], zb["X_train"]], axis=1)
    n = len(y)
    Poof = np.zeros((n, 10))
    YP = np.empty(n, dtype=object)
    for tri, vai in GroupKFold(3).split(X, y, groups):
        P = fit_mlp_ens(X[tri], y[tri], X[vai], 10, 64, 200, SEEDS)
        Poof[vai] = P
        yp = np.array(SPECIES)[P.argmax(1)].copy()
        for oc, cc2 in [("chaperonin_open", "chaperonin_closed"),
                        ("atpase_open", "atpase_closed")]:
            mtr = np.isin(yn[tri], [oc, cc2])
            s = spec_ens(X[tri][mtr], (yn[tri][mtr] == oc).astype(int),
                         X[vai], SEEDS)
            use = np.isin(yp, [oc, cc2])
            # NOTE: specialist scored on ALL val rows here (not just pair rows);
            # override still applies only where 10-way predicts in-pair.
            yp[use] = np.where(s[use] > 0.5, oc, cc2)
        YP[vai] = yp
        print("fold done", flush=True)
    correct = (YP == yn).astype(int)
    np.savez_compressed(ROOT / "oof74.npz", probs=Poof, pred=np.array(YP),
                        correct=correct)
    print(f"OOF acc={correct.mean():.3f}", flush=True)

    base = evaluate(yn, YP, Poof.max(1), pid)
    print(f"maxprob: sel={base['selective_accuracy']:.4f} "
          f"total={base['total']:.4f}", flush=True)

    # greedy per-class multipliers on OOF selective
    pred_idx = np.array([SPECIES.index(c) for c in YP])
    W = np.ones(10)
    grid = [0.7, 0.85, 1.0, 1.15, 1.3]
    improved = True
    while improved:
        improved = False
        for c in range(10):
            best = (evaluate(yn, YP, Poof.max(1) * W[pred_idx], pid)
                    ["selective_accuracy"], W[c])
            for g in grid:
                W[c] = g
                s = evaluate(yn, YP, Poof.max(1) * W[pred_idx], pid)[
                    "selective_accuracy"]
                if s > best[0] + 1e-4:
                    best = (s, g)
            if best[1] != W[c]:
                W[c] = best[1]
                improved = True
    mc = evaluate(yn, YP, Poof.max(1) * W[pred_idx], pid)
    print("class_w:", dict(zip(SPECIES, W.round(2))), flush=True)
    print(f"class_w: sel={mc['selective_accuracy']:.4f} (delta "
          f"{mc['selective_accuracy']-base['selective_accuracy']:+.4f})", flush=True)

    # hinge trust grid (OOF should be ~neutral; pseudo-CD is the weak signal)
    meta_tr = train[["ice_thickness_nm", "defocus_um", "tilt_max_deg"]].values
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    for k1, k2, k3 in [(0.5, 0.05, 0.3), (1.0, 0.1, 0.5), (2.0, 0.2, 1.0)]:
        trust = np.exp(-k1 * np.maximum(0, meta_tr[:, 1] - 3.5)
                       - k2 * np.maximum(0, meta_tr[:, 0] - 150)
                       - k3 * np.maximum(0, 55 - meta_tr[:, 2]))
        cf = Poof.max(1) * W[pred_idx] * trust
        mo = evaluate(yn, YP, cf, pid)
        mb = evaluate(yn[base_idx], YP[base_idx], cf[base_idx], pid[base_idx])
        mp = evaluate(yn[pseudo_idx], YP[pseudo_idx], cf[pseudo_idx],
                      pid[pseudo_idx])
        print(f"hinge({k1},{k2},{k3}): OOF-sel={mo['selective_accuracy']:.4f} "
              f"pseudo-sel={mp['selective_accuracy']:.4f}", flush=True)
    np.savez_compressed(ROOT / "conf74.npz", class_w=W)


if __name__ == "__main__":
    main()
