"""Phase 0a diagnostics (read-only on data/): prediction shift, per-tomogram
confidence stats, per-feature train->test shift, radius-proxy collapse check.

Writes: research/diag0_dist.json + prints tables.
Run: python research/diag0_dist.py  (CPU, a few minutes)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sstats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta
from metrics import SPECIES

ROOT = Path(__file__).resolve().parent
TABULAR = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]


def main():
    z = np.load(ROOT / "cache_m2.npz")
    names = list(z["names"])
    img_cols = [i for i, n in enumerate(names) if n not in TABULAR]
    Xtr, Xte = z["X_train"][:, img_cols], z["X_test"][:, img_cols]
    inames = [names[i] for i in img_cols]
    train, test = load_meta()
    sub = pd.read_csv(ROOT.parent / "submissions" / "submission.csv")

    print("== oof_special keys ==")
    try:
        oz = np.load(ROOT / "oof_special.npz")
        print("  ", list(oz.keys()), {k: oz[k].shape for k in oz.keys()})
    except Exception as e:
        print("  missing:", e)

    # ---- A. prediction distribution: submission(test) vs train prior & OOF ----
    print("\n== A. pred distribution ==")
    tr_prior = train["label"].value_counts(normalize=True).sort_index()
    te_pred = sub["label"].value_counts(normalize=True).sort_index()
    print(f"{'class':<20}{'train_prior':>12}{'test_pred':>10}")
    for c in SPECIES:
        print(f"{c:<20}{tr_prior.get(c, 0):>12.3f}{te_pred.get(c, 0):>10.3f}")
    # chi2 distance between train-prior-expected counts and test pred counts
    exp = np.array([tr_prior.get(c, 0) for c in SPECIES]) * len(sub)
    obs = np.array([(sub["label"] == c).sum() for c in SPECIES], dtype=float)
    chi2 = float((((obs - exp) ** 2) / np.maximum(exp, 1)).sum())
    print(f"chi2 vs train prior: {chi2:.1f}  missing classes: "
          f"{[c for c in SPECIES if (sub['label'] == c).sum() == 0]}")

    # ---- B. per-tomogram test confidence stats vs metadata ----
    print("\n== B. per-tomogram test stats (via cached full-train M2 probs) ==")
    pz = np.load(ROOT / "test_m2_probs.npy")  # (1132, 10) probs
    P = pz / pz.sum(1, keepdims=True)
    p1 = P.max(1)
    ent = -(P * np.log(P + 1e-12)).sum(1)
    srt = np.sort(P, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    t = test.copy()
    t["p1"], t["ent"], t["margin"] = p1, ent, margin
    g = t.groupby("tomogram_id").agg(
        n=("particle_id", "size"), mp1=("p1", "mean"), ment=("ent", "mean"),
        mmarg=("margin", "mean"), ice=("ice_thickness_nm", "first"),
        df=("defocus_um", "first"), tilt=("tilt_max_deg", "first"))
    print(g.sort_values("mp1").head(8).round(3).to_string())
    print("...")
    print(g.sort_values("mp1").tail(4).round(3).to_string())
    for col in ["ice", "df", "tilt"]:
        r = np.corrcoef(g[col], g["mp1"])[0, 1]
        print(f"corr({col}, mean-p1) = {r:+.3f}")
    g.to_csv(ROOT / "diag0_pertomo.csv")

    # ---- C. per-feature train->test shift (image-only feats) ----
    print("\n== C. feature shift (std units + KS statistic) ==")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    shift = (Xte.mean(0) - mu) / sd
    rows = []
    for j, nm in enumerate(inames):
        ks = sstats.ks_2samp(Xtr[:, j], Xte[:, j])
        rows.append((nm, float(shift[j]), float(ks.statistic), float(ks.pvalue)))
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"{'feat':<16}{'shift_sd':>9}{'KS':>8}")
    for nm, sh, ks, pv in rows:
        print(f"{nm:<16}{sh:>+9.2f}{ks:>8.3f}")
    # same, stratified: pseudo-CD-like train tomos vs rest (is shift within-train too?)
    tmeta = train.groupby("tomogram_id")[["ice_thickness_nm"]].first()
    hi_tomos = set(tmeta.sort_values("ice_thickness_nm", ascending=False).head(20).index)
    hi_mask = train["tomogram_id"].isin(hi_tomos).values
    shift_hi = (Xtr[hi_mask].mean(0) - Xtr[~hi_mask].mean(0)) / sd
    print("\nwithin-train (top20 ice tomos vs rest) top shifts:")
    for j in np.argsort(-np.abs(shift_hi))[:8]:
        print(f"  {inames[j]:<16}{shift_hi[j]:>+7.2f}sd  (train->test: {shift[j]:+.2f}sd)")

    # ---- E. radius-proxy collapse check ----
    print("\n== E. conformer radius-proxy distributions ==")
    vols_tr = np.load(str(ROOT.parent / "data" / "train_volumes.npy"), mmap_mode="r")
    vols_te = np.load(str(ROOT.parent / "data" / "test_volumes.npy"), mmap_mode="r")
    ax = np.arange(32) - 15.5
    XX, YY, ZZ = np.meshgrid(ax, ax, ax, indexing="ij")
    RR = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)

    def radius_proxy(v, lo, hi, ref_lo=2.0, ref_hi=3.5):
        vn = v.astype(np.float32)
        lo_p, hi_p = np.percentile(vn, [1.0, 99.0])
        vn = np.clip(vn, lo_p, hi_p)
        vn = (vn - vn.mean()) / (vn.std() + 1e-6)
        band = vn[(RR >= lo) & (RR < hi)].mean()
        ref = vn[(RR >= ref_lo) & (RR < ref_hi)].mean()
        return float(band - ref)

    out = {}
    # chaperonin: open 5.4-6.2 vs closed 4.2-5.0
    for cls, lo, hi in [("chaperonin_open", 5.4, 6.2), ("chaperonin_closed", 4.2, 5.0)]:
        idx = np.where(train["label"].values == cls)[0][:120]
        vals = [radius_proxy(vols_tr[i], 4.2, 6.2) for i in idx]
        out[cls + "_train"] = [float(np.mean(vals)), float(np.std(vals)), len(vals)]
    pred_chap = sub[sub["label"].isin(["chaperonin_open", "chaperonin_closed"])]
    for cls in ["chaperonin_open", "chaperonin_closed"]:
        idx = test[test["particle_id"].isin(pred_chap[pred_chap["label"] == cls]["particle_id"])].index.values[:120]
        vals = [radius_proxy(vols_te[i], 4.2, 6.2) for i in idx] if len(idx) else [np.nan]
        out[cls + "_testpred"] = [float(np.nanmean(vals)), float(np.nanstd(vals)), len(idx)]
    # atpase: open 4.0-4.6 vs closed 3.0-3.6
    for cls in ["atpase_open", "atpase_closed"]:
        idx = np.where(train["label"].values == cls)[0][:120]
        vals = [radius_proxy(vols_tr[i], 3.0, 4.6, 1.5, 2.5) for i in idx]
        out[cls + "_train"] = [float(np.mean(vals)), float(np.std(vals)), len(vals)]
    pred_atp = sub[sub["label"].isin(["atpase_open", "atpase_closed"])]
    for cls in ["atpase_open", "atpase_closed"]:
        idx = test[test["particle_id"].isin(pred_atp[pred_atp["label"] == cls]["particle_id"])].index.values[:120]
        vals = [radius_proxy(vols_te[i], 3.0, 4.6, 1.5, 2.5) for i in idx] if len(idx) else [np.nan]
        out[cls + "_testpred"] = [float(np.nanmean(vals)), float(np.nanstd(vals)), len(idx)]
    for k, (mn, sd_, n) in out.items():
        print(f"  {k:<28}mean={mn:+.3f} sd={sd_:.3f} n={n}")

    with open(ROOT / "diag0_dist.json", "w") as f:
        json.dump({"chi2_vs_train_prior": chi2,
                   "feat_shift": {r[0]: r[1] for r in rows},
                   "radius_proxy": out}, f, indent=1)
    print("\nwrote research/diag0_dist.json + research/diag0_pertomo.csv")


if __name__ == "__main__":
    main()
