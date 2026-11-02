"""M2 training: sklearn on handcrafted features. Local CPU, ~5 min.
Compares HistGradientBoosting vs LogisticRegression via:
  (a) pseudo-CD holdout (train on base, eval on pseudo-shift tomos)
  (b) 3-fold GroupKFold-by-tomogram OOF (grader-exact total)
Saves research/oof_m2.npz (winning model OOF probs) + research/m2_report.json.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, SPECIES
from splits import load_meta, group_folds, pseudo_cd_holdout

ROOT = Path(__file__).resolve().parent
CLS = {c: i for i, c in enumerate(SPECIES)}


def build(which):
    if which == "hgb":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                              max_leaf_nodes=31, min_samples_leaf=20,
                                              l2_regularization=1.0, random_state=42)
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                              random_state=42)


def report(y_true, y_pred, conf, pid, tag):
    m = evaluate(y_true, y_pred, conf, pid)
    print(f"[{tag}] total={m['total']:.4f} macro={m['macro_f1']:.4f} "
          f"conf_acc={m['conformer_acc']:.4f} minF1={m['min_class_f1']:.4f} "
          f"sel={m['selective_accuracy']:.4f} (cov={['%.3f' % a for a in m['cov_accs']]}) "
          f"acc={m['overall_acc']:.4f}")
    print(f"   per-class F1: " + " ".join(f"{k[:9]}={v:.2f}" for k, v in m["per_class_f1"].items()))
    return m


def main():
    z = np.load(ROOT / "cache_m2.npz")
    X = z["X_train"]
    train, _ = load_meta()
    y = train["label"].map(CLS).values
    yname = train["label"].values
    pid = train["particle_id"].values

    base_idx, pseudo_idx, hold_tomos = pseudo_cd_holdout(train)
    print("pseudo-CD holdout tomos:", hold_tomos)

    results = {}
    oofs = {}
    for which in ["hgb", "logreg"]:
        # (a) pseudo-CD
        sc = StandardScaler().fit(X[base_idx])
        clf = build(which).fit(sc.transform(X[base_idx]), y[base_idx])
        pv = clf.predict_proba(sc.transform(X[pseudo_idx]))
        pp = pv.argmax(1)
        m = report(yname[pseudo_idx], np.array(SPECIES)[pp], pv.max(1),
                   pid[pseudo_idx], f"{which}/pseudo-CD")
        results[f"{which}_pseudo"] = m
        # (b) 3-fold OOF
        oof = np.zeros((len(X), len(SPECIES)))
        for fi, (tri, vai) in enumerate(group_folds(train, 3)):
            sc = StandardScaler().fit(X[tri])
            c = build(which).fit(sc.transform(X[tri]), y[tri])
            oof[vai] = c.predict_proba(sc.transform(X[vai]))
        op = oof.argmax(1)
        m = report(yname, np.array(SPECIES)[op], oof.max(1), pid, f"{which}/OOF3")
        results[f"{which}_oof"] = {k: v for k, v in m.items() if k != "per_class_f1"}
        oofs[which] = oof

    winner = max(["hgb", "logreg"], key=lambda w: results[f"{w}_oof"]["total"])
    print("winner:", winner)
    np.savez_compressed(ROOT / "oof_m2.npz", oof=oofs[winner], model=winner)
    with open(ROOT / "m2_report.json", "w") as f:
        json.dump({k: (v if isinstance(v, dict) else v) for k, v in results.items()},
                  f, indent=1, default=float)
    print("saved oof_m2.npz + m2_report.json")


if __name__ == "__main__":
    main()
