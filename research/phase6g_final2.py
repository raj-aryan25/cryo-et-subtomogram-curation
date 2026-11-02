"""Final recipe shootout (honest, no templates): F1 vs F2 (+CORAL).
F = MLP-ens3 + ens specialists + plain override + class_w conf.
OOF 3-fold + pseudo-CD. Winner ships in solution.py.
Run: python research/phase6g_final2.py (CPU, ~30 min)
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
SEEDS = [7, 42, 123]
WCONF = np.load(ROOT / "conf74.npz")["class_w"]


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


def coral_to_source(Xs, Xt, eps=1e-3):
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs.T) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + eps * np.eye(Xt.shape[1])
    es, Vs = np.linalg.eigh(Cs)
    es = np.clip(es, 1e-9, None)
    et, Vt = np.linalg.eigh(Ct)
    et = np.clip(et, 1e-9, None)
    A = (Vt @ np.diag(1.0 / np.sqrt(et)) @ Vt.T) @ \
        (Vs @ np.diag(np.sqrt(es)) @ Vs.T)
    return (Xt - mu_t) @ A + mu_s


def run_split(X, y, yn, pid, tri, vai, use_coral):
    Xva = coral_to_source(X[tri], X[vai]) if use_coral else X[vai]
    P = fit_mlp_ens(X[tri], y[tri], Xva, 10, 64, 200, SEEDS)
    yp = np.array(SPECIES)[P.argmax(1)].copy()
    for oc, cc2 in [("chaperonin_open", "chaperonin_closed"),
                    ("atpase_open", "atpase_closed")]:
        mtr = np.isin(yn[tri], [oc, cc2])
        s = spec_ens(X[tri][mtr], (yn[tri][mtr] == oc).astype(int),
                     Xva, SEEDS)
        use = np.isin(yp, [oc, cc2])
        yp[use] = np.where(s[use] > 0.5, oc, cc2)
    pi = np.array([SPECIES.index(c) for c in yp])
    return evaluate(yn[vai], yp, P.max(1) * WCONF[pi], pid[vai])


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
    zb = np.load(ROOT / "cache_bank.npz")
    X = np.concatenate([z2["X_train"][:, cc], zb["X_train"]], axis=1)
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    for use_coral in [False, True]:
        tots = []
        for tri, vai in GroupKFold(3).split(X, y, groups):
            m = run_split(X, y, yn, pid, tri, vai, use_coral)
            tots.append((m["total"], m["macro_f1"], m["conformer_acc"],
                         m["min_class_f1"], m["selective_accuracy"]))
        a = np.mean(tots, axis=0)
        mp = run_split(X, y, yn, pid, base_idx, pseudo_idx, use_coral)
        print(f"coral={use_coral}: OOF total={a[0]:.4f} macro={a[1]:.3f} "
              f"conf={a[2]:.3f} min={a[3]:.3f} sel={a[4]:.4f} | pseudo "
              f"total={mp['total']:.4f} macro={mp['macro_f1']:.3f} "
              f"sel={mp['selective_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
