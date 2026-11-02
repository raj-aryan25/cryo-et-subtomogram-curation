"""Full recipe OOF on base29+bank: MLP-ens3+override (primary), HGB+override,
LR+override. Plus pseudo-CD for the winner.
Run: python research/phase6_full.py (CPU, ~20 min)
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


def override(yp, P10unused, X_tri_rows, yn_tri, X_vai_rows, fitter, seeds, hid, ep):
    yp = yp.copy()
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        mtr = np.isin(yn_tri, [oc, cc])
        s = fitter(X_tri_rows[mtr], (yn_tri[mtr] == oc).astype(int),
                   X_vai_rows, seeds, hid, ep)
        use = np.isin(yp, [oc, cc])
        yp[use] = np.where(s[use] > 0.5, oc, cc)
    return yp


def spec_mlp(Xtr, yb, Xte, seeds, hid, ep):
    P = 0
    for sd in seeds:
        torch.manual_seed(sd)
        sc = StandardScaler().fit(Xtr)
        A = sc.transform(Xtr).astype(np.float32)
        B = sc.transform(Xte).astype(np.float32)
        net = MLP(A.shape[1], hid, 2)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        XA = torch.tensor(A)
        YA = torch.tensor(yb, dtype=torch.long)
        net.train()
        for _ in range(ep):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(XA), YA)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            P = P + torch.softmax(net(torch.tensor(B)), 1).numpy()[:, 1]
    return P / len(seeds)


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
    print("X:", X.shape, flush=True)

    def run_split(tri, vai):
        out = {}
        # MLP-ens3 + override(ens specs)
        P = fit_mlp_ens(X[tri], y[tri], X[vai], 10, 64, 200, SEEDS)
        yp = np.array(SPECIES)[P.argmax(1)].copy()
        for oc, cc2 in [("chaperonin_open", "chaperonin_closed"),
                        ("atpase_open", "atpase_closed")]:
            mtr = np.isin(yn[tri], [oc, cc2])
            s = spec_mlp(X[tri][mtr], (yn[tri][mtr] == oc).astype(int),
                         X[vai], SEEDS, 32, 150)
            use = np.isin(yp, [oc, cc2])
            yp[use] = np.where(s[use] > 0.5, oc, cc2)
        out["mlpens"] = (yp, P.max(1))
        # HGB + override (single-seed mlp specs for speed)
        Ph = fit_hgb(X[tri], y[tri], X[vai])
        yh = np.array(SPECIES)[Ph.argmax(1)].copy()
        for oc, cc2 in [("chaperonin_open", "chaperonin_closed"),
                        ("atpase_open", "atpase_closed")]:
            mtr = np.isin(yn[tri], [oc, cc2])
            mva = np.isin(yn[vai], [oc, cc2])
            torch.manual_seed(42)
            sc = StandardScaler().fit(X[tri][mtr])
            A = sc.transform(X[tri][mtr]).astype(np.float32)
            net = MLP(A.shape[1], 32, 2)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
            XA = torch.tensor(A)
            YA = torch.tensor((yn[tri][mtr] == oc).astype(int), dtype=torch.long)
            net.train()
            for _ in range(150):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(net(XA), YA)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                s = torch.softmax(net(torch.tensor(
                    sc.transform(X[vai]).astype(np.float32))), 1)[:, 1].numpy()
            use = np.isin(yh, [oc, cc2])
            yh[use] = np.where(s[use] > 0.5, oc, cc2)
        out["hgb"] = (yh, Ph.max(1))
        return out

    for tag, idxs in [("OOF", None)]:
        acc = {"mlpens": [], "hgb": []}
        for tri, vai in GroupKFold(3).split(X, y, groups):
            R = run_split(tri, vai)
            for k in acc:
                yp, cf = R[k]
                m = evaluate(yn[vai], yp, cf, pid[vai])
                acc[k].append((m["total"], m["macro_f1"], m["conformer_acc"],
                               m["min_class_f1"], m["selective_accuracy"]))
        for k in acc:
            a = np.mean(acc[k], axis=0)
            print(f"OOF {k}: total={a[0]:.4f} macro={a[1]:.3f} conf={a[2]:.3f} "
                  f"min={a[3]:.3f} sel={a[4]:.4f}", flush=True)
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    R = run_split(base_idx, pseudo_idx)
    for k in ["mlpens", "hgb"]:
        yp, cf = R[k]
        m = evaluate(yn[pseudo_idx], yp, cf, pid[pseudo_idx])
        print(f"pseudo {k}: total={m['total']:.4f} macro={m['macro_f1']:.3f} "
              f"min={m['min_class_f1']:.3f} sel={m['selective_accuracy']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
