"""Workstream B: unsupervised alignment procedures, validated on pseudo-CD.
V0 none | V1 global CORAL pseudo->base | V2 per-tomogram z-score |
V3 mean-shift only. Logreg on 74 feats. Winner gets OOF check.
Run: python research/phase6b_align.py (CPU, minutes)
"""
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
EPS = 1e-3


def fit_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             random_state=42)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))


def coral_to_source(Xs, Xt):
    """Map target rows Xt into source distribution (mean+cov)."""
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs.T) + EPS * np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + EPS * np.eye(Xt.shape[1])
    es, Vs = np.linalg.eigh(Cs)
    es = np.clip(es, 1e-9, None)
    As = Vs @ np.diag(1.0 / np.sqrt(es)) @ Vs.T  # whiten source... we need:
    # standard CORAL: Xt_ad = (Xt-mu_t) @ A, A = Ct^{-1/2} Cov_s^{1/2}
    et, Vt = np.linalg.eigh(Ct)
    et = np.clip(et, 1e-9, None)
    A = (Vt @ np.diag(1.0 / np.sqrt(et)) @ Vt.T) @ \
        (Vs @ np.diag(np.sqrt(es)) @ Vs.T)
    return (Xt - mu_t) @ A + mu_s


def pertomo_z(X, tomos):
    X = X.copy()
    for t in np.unique(tomos):
        m = tomos == t
        mu = X[m].mean(0)
        sd = X[m].std(0) + 1e-9
        X[m] = (X[m] - mu) / sd
    return X


def main():
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    tomos = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    z2 = np.load(ROOT / "cache_m2.npz")
    names = list(z2["names"])
    tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
    cc = [i for i, n in enumerate(names) if n not in tab]
    zb = np.load(ROOT / "cache_bank.npz")
    X = np.concatenate([z2["X_train"][:, cc], zb["X_train"]], axis=1)
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    Xb, Xp = X[base_idx], X[pseudo_idx]
    yb = y[base_idx]

    variants = {"V0_none": (Xb, Xp)}
    variants["V1_coral"] = (Xb, coral_to_source(Xb, Xp))
    Xpt = pertomo_z(X, tomos)
    variants["V2_pertomo"] = (Xpt[base_idx], Xpt[pseudo_idx])
    variants["V3_meanshift"] = (Xb, Xp - Xp.mean(0) + Xb.mean(0))
    for name, (A, B) in variants.items():
        P = fit_lr(A, yb, B)
        yp = np.array(SPECIES)[P.argmax(1)]
        m = evaluate(yn[pseudo_idx], yp, P.max(1), pid[pseudo_idx])
        print(f"pseudo {name}: total={m['total']:.4f} macro={m['macro_f1']:.3f} "
              f"min={m['min_class_f1']:.3f} sel={m['selective_accuracy']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
