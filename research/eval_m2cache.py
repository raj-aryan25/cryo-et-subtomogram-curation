"""OOF eval for a given M2 cache. Run: python research/eval_m2cache.py research/cache_m2b.npz"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from metrics import evaluate, SPECIES
from splits import load_meta


def main(path):
    z = np.load(path)
    X = z["X_train"]
    train, _ = load_meta()
    y = train["label"].map({c: i for i, c in enumerate(SPECIES)}).values
    yn = train["label"].values
    pid = train["particle_id"].values
    oof = np.zeros((len(X), 10))
    for tri, vai in GroupKFold(3).split(X, y, train["tomogram_id"].values):
        sc = StandardScaler().fit(X[tri])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 random_state=42).fit(sc.transform(X[tri]), y[tri])
        oof[vai] = clf.predict_proba(sc.transform(X[vai]))
    m = evaluate(yn, np.array(SPECIES)[oof.argmax(1)], oof.max(1), pid)
    print("total=%.4f macro=%.4f conf=%.4f minF1=%.4f sel=%.4f acc=%.4f" % (
        m["total"], m["macro_f1"], m["conformer_acc"], m["min_class_f1"],
        m["selective_accuracy"], m["overall_acc"]))
    for k, v in m["per_class_f1"].items():
        print("  %-18s %.3f" % (k, v))


if __name__ == "__main__":
    main(sys.argv[1])
