"""FINAL pipeline (validated): image-only rotation-invariant features -> 10-way logreg
+ binary conformer-specialist plain override -> max-prob confidence.
OOF: total 0.414 / macro 0.427 / conf 0.288 / minF1 0.258 / sel 0.541.
Sanity gates fall back to 10-way labels if override degenerates on test.
Run: python research/predict_test.py -> submissions/submission.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, SPECIES
from splits import load_meta

ROOT = Path(__file__).resolve().parent
TABULAR = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
PAIRS = [("chaperonin_open", "chaperonin_closed"), ("atpase_open", "atpase_closed")]
SPEC_C = 0.1


def fit_predict(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                             random_state=42).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))


def main():
    z = np.load(ROOT / "cache_m2.npz")
    names = list(z["names"])
    cols = [i for i, n in enumerate(names) if n not in TABULAR]
    Xtr, Xte = z["X_train"][:, cols], z["X_test"][:, cols]
    train, test = load_meta()
    yn = train["label"].values
    y = np.array([SPECIES.index(c) for c in yn])

    P10 = fit_predict(Xtr, y, Xte, 1.0)
    yp = np.array(SPECIES)[P10.argmax(1)].copy()
    # plain specialist override (only where 10-way already predicts in-pair)
    for open_c, closed_c in PAIRS:
        m = np.isin(yn, [open_c, closed_c])
        s = fit_predict(Xtr[m], (yn[m] == open_c).astype(int), Xte, SPEC_C)[:, 1]
        use = np.isin(yp, [open_c, closed_c])
        yp[use] = np.where(s[use] > 0.5, open_c, closed_c)
    # sanity gates: every class present, none dominant -> else fall back
    dist = Counter(yp)
    missing = [c for c in SPECIES if dist.get(c, 0) == 0]
    dom = [c for c, n in dist.items() if n > 0.5 * len(yp)]
    if missing or dom:
        print(f"GATE-TRIP missing={missing} dominant={dom}: falling back to 10-way")
        yp = np.array(SPECIES)[P10.argmax(1)]
    else:
        print("gates pass:", dict(sorted(dist.items())))
    h = np.array([int(p.split("_")[1]) for p in test["particle_id"].values])
    conf = np.clip(P10.max(1), 1e-3, 1 - 1e-3) + 1e-9 * (h % 1000)
    sub = pd.DataFrame({"particle_id": test["particle_id"].values,
                        "label": yp, "confidence": conf})
    assert len(sub) == 1132 and sub["particle_id"].is_unique
    assert (sub["particle_id"].values == test["particle_id"].values).all()
    assert set(sub["label"]).issubset(set(SPECIES)) and len(set(sub["label"])) == 10
    assert sub["confidence"].between(0, 1).all() and sub["confidence"].notna().all()
    out = ROOT.parent / "submissions" / "submission.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, "| conf deciles:",
          np.round(np.percentile(sub["confidence"], [0, 10, 25, 50, 75, 90, 100]), 3))


if __name__ == "__main__":
    main()
