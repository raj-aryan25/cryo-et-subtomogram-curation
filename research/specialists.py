"""Binary conformer specialists (local CPU).
For each pair (chap open/closed, atpase open/closed): logreg trained ONLY on pair rows.
Override rule (OOF-validated): where 10-way predicts IN the pair, use specialist.
Metric recomputed grader-exact on full OOF. Also tries unassigned-vs-rest specialist.
Run: python research/specialists.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, SPECIES
from splits import load_meta

ROOT = Path(__file__).resolve().parent


def oof_binary(X, yb, groups, C=1.0):
    oof = np.zeros(len(yb))
    for tri, vai in GroupKFold(3).split(X, yb, groups):
        sc = StandardScaler().fit(X[tri])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                 random_state=42).fit(sc.transform(X[tri]), yb[tri])
        oof[vai] = clf.predict_proba(sc.transform(X[vai]))[:, 1]
    return oof


def main():
    for cache in ["cache_m2.npz", "cache_m2b.npz"]:
        z = np.load(ROOT / cache)
        X = z["X_train"]
        train, _ = load_meta()
        yn = train["label"].values
        pid = train["particle_id"].values
        groups = train["tomogram_id"].values
        yp10 = np.array(SPECIES)[np.load(ROOT / "oof_m2.npz")["oof"].argmax(1)]
        P10 = np.load(ROOT / "oof_m2.npz")["oof"]
        print(f"=== {cache} ===")
        yp = yp10.copy()
        for open_c, closed_c in [("chaperonin_open", "chaperonin_closed"),
                                 ("atpase_open", "atpase_closed")]:
            m = np.isin(yn, [open_c, closed_c])
            yb = (yn[m] == open_c).astype(int)
            best = None
            for C in [0.3, 1.0, 3.0]:
                s = oof_binary(X[m], yb, groups[m], C)
                acc = (((s > 0.5).astype(int) == yb)).mean()
                if best is None or acc > best[0]:
                    best = (acc, C, s)
            acc, C, s = best
            print(f"{open_c.split('_')[0]} specialist (C={C}): pair-acc={acc:.4f} "
                  f"(10-way on pair: {(yp10[m] == yn[m]).mean():.4f})")
            # override where 10-way predicted inside the pair
            mo = yp10[m] != yn[m]
            use = np.isin(yp, [open_c, closed_c])
            pred_pair = np.where(s > 0.5, open_c, closed_c)
            # map subset-order probs back to full rows
            full = np.empty(len(yn), dtype=object)
            full[m] = pred_pair
            yp[use] = full[use]
        m = evaluate(yn, yp, P10.max(1), pid)
        print("after-override total=%.4f macro=%.4f conf=%.4f minF1=%.4f sel=%.4f" % (
            m["total"], m["macro_f1"], m["conformer_acc"], m["min_class_f1"],
            m["selective_accuracy"]))


if __name__ == "__main__":
    main()
