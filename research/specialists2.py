"""Specialists v2: tuned pair classifiers + pair-vs-rest detectors + gated cascade.
Reports full metric deltas vs M2-logreg base. Saves research/oof_special.npz
(pred labels + confidence) for the winning cascade.
Run: python research/specialists2.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, SPECIES
from splits import load_meta

ROOT = Path(__file__).resolve().parent
PAIRS = [("chaperonin_open", "chaperonin_closed"), ("atpase_open", "atpase_closed")]


def oof_proba(X, yb, groups, mk):
    o = np.zeros(len(yb))
    for tri, vai in GroupKFold(3).split(X, yb, groups):
        sc = StandardScaler().fit(X[tri])
        mk.fit(sc.transform(X[tri]), yb[tri])
        o[vai] = mk.predict_proba(sc.transform(X[vai]))[:, 1]
    return o


def main():
    z = np.load(ROOT / "cache_m2.npz")
    X = z["X_train"]
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    P10 = np.load(ROOT / "oof_m2.npz")["oof"]
    yp10 = np.array(SPECIES)[P10.argmax(1)]
    conf10 = P10.max(1)
    m0 = evaluate(yn, yp10, conf10, pid)
    print("BASE total=%.4f macro=%.4f conf=%.4f minF1=%.4f sel=%.4f" % (
        m0["total"], m0["macro_f1"], m0["conformer_acc"], m0["min_class_f1"],
        m0["selective_accuracy"]))

    spec = {}   # (open,closed) -> proba(open) OOF over PAIR rows only
    det = {}    # pairname -> proba(in-pair) OOF over ALL rows
    for open_c, closed_c in PAIRS:
        tag = open_c.split("_")[0]
        m = np.isin(yn, [open_c, closed_c])
        yb = (yn[m] == open_c).astype(int)
        best = None
        cands = {"lr01": LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=42),
                 "lr03": LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced", random_state=42),
                 "lda": LinearDiscriminantAnalysis()}
        for nm, mk in cands.items():
            s = oof_proba(X[m], yb, groups[m], mk)
            acc = ((s > 0.5).astype(int) == yb).mean()
            if best is None or acc > best[0]:
                best = (acc, nm, s)
        print(f"{tag} pair: best={best[1]} acc={best[0]:.4f}")
        spec[(open_c, closed_c)] = best[2]
        # detector: pair vs rest
        yd = m.astype(int)
        bestd = None
        for nm, mk in cands.items():
            s = oof_proba(X, yd, groups, mk)
            # detector quality: recall at 5% FPR-ish -> use balanced acc
            pred = (s > 0.5).astype(int)
            bacc = 0.5 * ((pred[m] == 1).mean() + (pred[~m] == 0).mean())
            if bestd is None or bacc > bestd[0]:
                bestd = (bacc, nm, s)
        print(f"{tag} detector: best={bestd[1]} bacc={bestd[0]:.4f} "
              f"recall={((bestd[2][m] > 0.5).mean()):.3f} fpr={((bestd[2][~m] > 0.5).mean()):.3f}")
        det[tag] = bestd[2]

    # cascade variants
    variants = {}
    # v_plain: override where 10-way pred in pair
    yp = yp10.copy()
    for (open_c, closed_c), s in spec.items():
        m = np.isin(yn, [open_c, closed_c])
        full = np.empty(len(yn), dtype=object)
        full[m] = np.where(s > 0.5, open_c, closed_c)
        use = np.isin(yp, [open_c, closed_c])
        yp[use] = full[use]
    variants["plain"] = yp
    # v_margin: override only if specialist confident (|s-0.5|>t)
    for t in [0.1, 0.2, 0.3]:
        yp = yp10.copy()
        for (open_c, closed_c), s in spec.items():
            m = np.isin(yn, [open_c, closed_c])
            full = np.empty(len(yn), dtype=object)
            full[m] = np.where(s > 0.5, open_c, closed_c)
            sfull = np.full(len(yn), 0.5)
            sfull[m] = s
            use = np.isin(yp, [open_c, closed_c]) & (np.abs(sfull - 0.5) > t)
            yp[use] = full[use]
        variants[f"margin{t}"] = yp
    # v_cascade: detector-gated (det>td -> specialist), plus plain override
    for td in [0.3, 0.5]:
        yp = yp10.copy()
        for (open_c, closed_c), s in spec.items():
            tag = open_c.split("_")[0]
            m = np.isin(yn, [open_c, closed_c])
            full = np.empty(len(yn), dtype=object)
            full[m] = np.where(s > 0.5, open_c, closed_c)
            use = np.isin(yp, [open_c, closed_c]) | (det[tag] > td)
            yp[use] = np.where(m, full, yp)[use]
        variants[f"casc{td}"] = yp

    best = ("base", yp10, m0["total"])
    for nm, ypv in variants.items():
        m = evaluate(yn, ypv, conf10, pid)
        print(f"{nm}: total=%.4f macro=%.4f conf=%.4f minF1=%.4f sel=%.4f" % (
            m["total"], m["macro_f1"], m["conformer_acc"], m["min_class_f1"],
            m["selective_accuracy"]))
        if m["total"] > best[2]:
            best = (nm, ypv, m["total"])
    print("WINNER:", best[0], round(best[2], 4))
    np.savez_compressed(ROOT / "oof_special.npz",
                        pred=np.array(best[1]), conf=conf10, variant=best[0])


if __name__ == "__main__":
    main()
