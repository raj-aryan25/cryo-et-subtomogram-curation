"""EXP: torch MLP parity check vs logreg (10-way + specialists), GroupKFold OOF."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, SPECIES
from splits import load_meta

ROOT = Path(__file__).resolve().parent
torch.manual_seed(42)
np.random.seed(42)
DEV = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, Xte, d_hid=64, epochs=120, lr=3e-3, wd=1e-4, ncls=10):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
    w = len(ytr) / (ncls * np.bincount(ytr, minlength=ncls))
    cw = torch.tensor(w, dtype=torch.float32, device=DEV)
    net = MLP(A.shape[1], d_hid, ncls).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    XA = torch.tensor(A, device=DEV)
    YA = torch.tensor(ytr, dtype=torch.long, device=DEV)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(XA), YA, weight=cw)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        P = torch.softmax(net(torch.tensor(B, device=DEV)), 1).cpu().numpy()
    return P


def main():
    z = np.load(ROOT / "cache_m2.npz")
    names = list(z["names"])
    tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
    cols = [i for i, n in enumerate(names) if n not in tab]
    X = z["X_train"][:, cols]
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    oof = np.zeros((len(X), 10))
    for tri, vai in GroupKFold(3).split(X, y, groups):
        oof[vai] = fit_mlp(X[tri], y[tri], X[vai])
    yp10 = np.array(SPECIES)[oof.argmax(1)]
    m = evaluate(yn, yp10, oof.max(1), pid)
    print("MLP-10way OOF total=%.4f macro=%.4f conf=%.4f minF1=%.4f sel=%.4f" % (
        m["total"], m["macro_f1"], m["conformer_acc"], m["min_class_f1"],
        m["selective_accuracy"]))
    # specialists as torch binary MLPs
    spec = {}
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        mm = np.isin(yn, [oc, cc])
        yb = (yn[mm] == oc).astype(int)
        o = np.zeros(mm.sum())
        for tri, vai in GroupKFold(3).split(X[mm], yb, groups[mm]):
            o[vai] = fit_mlp(X[mm][tri], yb[tri], X[mm][vai], d_hid=32,
                             epochs=150, ncls=2)[:, 1]
        acc = (((o > 0.5).astype(int)) == yb).mean()
        print(f"{oc.split('_')[0]} torch-spec pair-acc={acc:.4f}")
        spec[(oc, cc)] = o
    yp = yp10.copy()
    for (oc, cc), s in spec.items():
        mm = np.isin(yn, [oc, cc])
        full = np.empty(len(yn), dtype=object)
        full[mm] = np.where(s > 0.5, oc, cc)
        use = np.isin(yp, [oc, cc])
        yp[use] = full[use]
    m2 = evaluate(yn, yp, oof.max(1), pid)
    print("MLP+override OOF total=%.4f macro=%.4f conf=%.4f minF1=%.4f sel=%.4f" % (
        m2["total"], m2["macro_f1"], m2["conformer_acc"], m2["min_class_f1"],
        m2["selective_accuracy"]))


if __name__ == "__main__":
    main()
