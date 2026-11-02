"""Workstream A gate: HGB/MLP x {base29, base29+bank45}, OOF + pseudo-CD.
Run: python research/phase6_eval.py (CPU)
"""
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
torch.manual_seed(42)
np.random.seed(42)


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


def fit_hgb(Xtr, ytr, Xte):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=20,
                                         l2_regularization=1.0,
                                         class_weight="balanced",
                                         random_state=42)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)


def fit_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             random_state=42)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))


def run_all(X, y, yn, pid, tri, vai):
    out = {}
    out["hgb"] = fit_hgb(X[tri], y[tri], X[vai])
    out["lr"] = fit_lr(X[tri], y[tri], X[vai])
    out["mlp1"] = fit_mlp_ens(X[tri], y[tri], X[vai], 10, 64, 200, [42])
    return out


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    z2 = np.load(ROOT / "cache_m2.npz")
    names = list(z2["names"])
    tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
    cc = [i for i, n in enumerate(names) if n not in tab]
    Xb = z2["X_train"][:, cc]
    zb = np.load(ROOT / "cache_bank.npz")
    Xf = np.concatenate([Xb, zb["X_train"]], axis=1)
    print("shapes:", Xb.shape, Xf.shape, flush=True)
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    for tag, X in [("base29", Xb), ("base29+bank", Xf)]:
        # OOF
        acc = {"hgb": [], "lr": [], "mlp1": []}
        for tri, vai in GroupKFold(3).split(X, y, groups):
            P = run_all(X, y, yn, pid, tri, vai)
            for k in acc:
                yp = np.array(SPECIES)[P[k].argmax(1)]
                m = evaluate(yn[vai], yp, P[k].max(1), pid[vai])
                acc[k].append((m["total"], m["macro_f1"], m["min_class_f1"],
                               m["selective_accuracy"]))
        print(f"== {tag} OOF ==", flush=True)
        for k in ["hgb", "lr", "mlp1"]:
            a = np.mean(acc[k], axis=0)
            print(f"  {k}: total={a[0]:.4f} macro={a[1]:.3f} min={a[2]:.3f} "
                  f"sel={a[3]:.3f}", flush=True)
        # pseudo-CD with hgb + mlp1
        for k, fn in [("hgb", lambda: fit_hgb(X[base_idx], y[base_idx], X[pseudo_idx])),
                      ("mlp1", lambda: fit_mlp_ens(X[base_idx], y[base_idx],
                                                  X[pseudo_idx], 10, 64, 200, [42]))]:
            P = fn()
            yp = np.array(SPECIES)[P.argmax(1)]
            m = evaluate(yn[pseudo_idx], yp, P.max(1), pid[pseudo_idx])
            print(f"  pseudo-{k}: total={m['total']:.4f} macro={m['macro_f1']:.3f} "
                  f"min={m['min_class_f1']:.3f} sel={m['selective_accuracy']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
