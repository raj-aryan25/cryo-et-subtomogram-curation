"""Phase 2b: cross-fitted correctness ranker + calibration candidates.
Ranker feats: p1, margin, ent, energy, tta_mp1/agree/sp1/ment, wedge fillfrac/
inout, ice/defocus/tilt. Correctness from OOF. Candidates: maxprob | ranker |
ranker+per-tomo-temp | ranker+class-offsets. Gate: OOF selective AND
pseudo-CD selective must beat maxprob. Saves research/conf_ranker.npz.
Run: python research/phase2_rank.py (CPU, minutes)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent


def build_matrix(prefix, z, train, test):
    n = len(z[f"{prefix}_p1"])
    F = np.stack([z[f"{prefix}_{k}"] for k in
                  ["p1", "margin", "ent", "energy", "tta_mp1", "tta_agree",
                   "tta_sp1", "tta_ment"]], axis=1)
    zv = np.load(ROOT / "cache_v3.npz")
    print("cache_v3 names tail:", list(zv["names"][-4:]))
    return F


def main():
    train, test = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    z = np.load(ROOT / "tta_stats.npz")
    oof_pred = z["oof_pred"]
    correct = (oof_pred == y).astype(int)
    print(f"OOF acc (E3 head): {correct.mean():.3f}")

    zv = np.load(ROOT / "cache_v3.npz")
    names3 = list(zv["names"])
    fi = names3.index("wedge_fillfrac")
    ii = names3.index("wedge_inout")
    meta = train[["ice_thickness_nm", "defocus_um", "tilt_max_deg"]].values
    F_oof = np.concatenate([
        np.stack([z["oof_p1"], z["oof_margin"], z["oof_ent"], z["oof_energy"],
                  z["oof_tta_mp1"], z["oof_tta_agree"], z["oof_tta_sp1"],
                  z["oof_tta_ment"]], axis=1),
        zv["X_train"][:, [fi, ii]], meta], axis=1)
    F_te = np.concatenate([
        np.stack([z["test_p1"], z["test_margin"], z["test_ent"], z["test_energy"],
                  z["test_tta_mp1"], z["test_tta_agree"], z["test_tta_sp1"],
                  z["test_tta_ment"]], axis=1),
        zv["X_test"][:, [fi, ii]],
        test[["ice_thickness_nm", "defocus_um", "tilt_max_deg"]].values], axis=1)
    fnames = ["p1", "margin", "ent", "energy", "tta_mp1", "tta_agree",
              "tta_sp1", "tta_ment", "fillfrac", "inout", "ice", "df", "tilt"]
    print("ranker features:", fnames)

    # cross-fitted ranker scores
    ranker_oof = np.zeros(len(y))
    for tri, vai in GroupKFold(3).split(F_oof, y, groups):
        sc = StandardScaler().fit(F_oof[tri])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(F_oof[tri]),
                                                           correct[tri])
        ranker_oof[vai] = clf.predict_proba(sc.transform(F_oof[vai]))[:, 1]
    sc = StandardScaler().fit(F_oof)
    clf_full = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(F_oof), correct)
    ranker_te = clf_full.predict_proba(sc.transform(F_te))[:, 1]
    print("ranker coefs:", dict(zip(fnames, clf_full.coef_[0].round(3))))

    yp_oof = np.array(SPECIES)[oof_pred]
    base = evaluate(yn, yp_oof, z["oof_p1"], pid)
    mr = evaluate(yn, yp_oof, ranker_oof, pid)
    print(f"maxprob : selective={base['selective_accuracy']:.4f} "
          f"cov={[round(a,3) for a in base['cov_accs']]}")
    print(f"ranker  : selective={mr['selective_accuracy']:.4f} "
          f"cov={[round(a,3) for a in mr['cov_accs']]}")

    # per-tomogram temperature on ranker (fit scale per tomo on OOF? unsupervised:
    # center ranker scores per tomogram to remove tomo-level optimism bias)
    tomos = train["tomogram_id"].values
    ranker_c = ranker_oof.copy()
    for tm in np.unique(tomos):
        m = tomos == tm
        ranker_c[m] = ranker_oof[m] - ranker_oof[m].mean()
    # rank by centered score breaks ties across tomos poorly; instead blend:
    for w in [0.5, 0.7]:
        blend = w * ranker_oof + (1 - w) * (ranker_c - ranker_c.min() + 1e-9)
        mb = evaluate(yn, yp_oof, blend, pid)
        print(f"blend w={w}: selective={mb['selective_accuracy']:.4f}")

    # pseudo-CD gate for maxprob vs ranker
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    for nm, cf in [("maxprob", z["oof_p1"]), ("ranker", ranker_oof)]:
        mb = evaluate(yn[base_idx], yp_oof[base_idx], cf[base_idx], pid[base_idx])
        mp = evaluate(yn[pseudo_idx], yp_oof[pseudo_idx], cf[pseudo_idx], pid[pseudo_idx])
        print(f"{nm}: base-sel={mb['selective_accuracy']:.4f} "
              f"pseudo-sel={mp['selective_accuracy']:.4f}")

    np.savez_compressed(ROOT / "conf_ranker.npz", ranker_oof=ranker_oof,
                        ranker_test=ranker_te,
                        coef=clf_full.coef_[0], mean=sc.mean_, scale=sc.scale_,
                        fnames=np.array(fnames))
    print("wrote research/conf_ranker.npz")


if __name__ == "__main__":
    main()
