"""Confidence + prior work that cannot change ID labels (labels stay M2-logreg).

1) Ranking search on OOF: candidates p1 / margin / -entropy / energy + per-class
   temperature + metadata trust; metric = grader selective_accuracy.
2) Saerens-Latinne-Decaestecker EM prior-shift: validate the PROCEDURE on OOF
   (treat OOF as 'test'), then estimate real test priors from test_m2_probs.
Saves research/calib.json.
Run: python research/calibrate.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, selective_curve, SPECIES
from splits import load_meta

ROOT = Path(__file__).resolve().parent


def scores_from_probs(P, T=1.0):
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    S = np.sort(P, axis=1)
    p1, p2 = S[:, -1], S[:, -2]
    ent = -(P * np.log(P + 1e-12)).sum(1)
    energy = -T * np.log(np.exp(np.clip(P, 1e-12, 1) / T).sum(1))  # on probs: proxy
    return {"p1": p1, "margin": p1 - p2, "negent": -ent, "negenergy": -energy}


def sel_metric(yn, yp, conf, pid):
    return evaluate(yn, yp, conf, pid)["selective_accuracy"]


def fit_temperature(P, y, per_class=False, iters=200):
    """Minimize NLL over T (scalar) or T_c (per predicted class). Grid+refine."""
    P = np.clip(np.asarray(P, float), 1e-9, 1.0)
    logP = np.log(P)
    pred = P.argmax(1)
    best, bestT = 1e9, None
    grid = np.concatenate([[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]])
    if not per_class:
        for T in grid:
            Q = np.exp(logP / T)
            Q /= Q.sum(1, keepdims=True)
            nll = -np.log(Q[np.arange(len(y)), y]).mean()
            if nll < best:
                best, bestT = nll, T
        return float(bestT), best
    Tc = np.ones(P.shape[1])
    for c in range(P.shape[1]):
        m = pred == c
        if m.sum() < 20:
            continue
        bb, bt = 1e9, 1.0
        for T in grid:
            Q = np.exp(logP[m] / T)
            Q /= Q.sum(1, keepdims=True)
            nll = -np.log(Q[:, y[m]]).mean() if False else -np.log(
                Q[np.arange(m.sum()), y[m]]).mean()
            if nll < bb:
                bb, bt = nll, T
        Tc[c] = bt
    return Tc, None


def apply_T(P, T):
    if np.ndim(T) == 0:
        Q = np.exp(np.log(np.clip(P, 1e-12, 1)) / T)
    else:
        Q = np.exp(np.log(np.clip(P, 1e-12, 1)) / np.asarray(T)[None, :])
    return Q / Q.sum(1, keepdims=True)


def em_prior(P, pi_train, iters=100):
    """Saerens EM: estimate test priors from posteriors. Returns (pi_test, Q_adj)."""
    P = np.asarray(P, float)
    pi = P.mean(0).copy()
    for _ in range(iters):
        W = (P * (pi / np.maximum(pi_train, 1e-9))[None, :])
        W /= W.sum(1, keepdims=True)
        pi_new = W.mean(0)
        if np.abs(pi_new - pi).max() < 1e-8:
            pi = pi_new
            break
        pi = pi_new
    Q = P * (pi / np.maximum(pi_train, 1e-9))[None, :]
    return pi / pi.sum(), Q / Q.sum(1, keepdims=True)


def main():
    o = np.load(ROOT / "oof_m2.npz")
    P = np.asarray(o["oof"], float)  # OOF probs, rows = train.csv order
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    yp = np.array(SPECIES)[P.argmax(1)]
    y = np.array([SPECIES.index(c) for c in yn])
    base_sel = sel_metric(yn, yp, P.max(1), pid)
    print(f"labels=logreg frozen. max-prob selective={base_sel:.4f}")

    # --- 1) temperature on OOF ---
    T0, _ = fit_temperature(P, y)
    Tc, _ = fit_temperature(P, y, per_class=True)
    print("global T:", T0, "per-class T:", np.round(Tc, 2))
    res = {"p1": base_sel}
    for nm, Q in [("T_global", apply_T(P, T0)), ("T_perclass", apply_T(P, Tc))]:
        s = scores_from_probs(Q)
        for k, v in s.items():
            msel = sel_metric(yn, yp, v, pid)
            res[f"{nm}/{k}"] = round(float(msel), 4)
            print(f"  {nm}/{k}: selective={msel:.4f}")
    # margin of temperature-scaled as the likely winner; try simple blends
    Qb = apply_T(P, Tc)
    sb = scores_from_probs(Qb)
    for w in [0.3, 0.5, 0.7]:
        blend = w * (sb["margin"] / (sb["margin"].max() + 1e-12)) + \
            (1 - w) * (sb["p1"] / (sb["p1"].max() + 1e-12))
        print(f"  blend margin*{w}+p1: selective={sel_metric(yn, yp, blend, pid):.4f}")
    print("curve (max-prob):", [(p, round(a, 3)) for p, a, _ in selective_curve(yn, yp, P.max(1), pid)])
    print("curve (T/margin):", [(p, round(a, 3)) for p, a, _ in selective_curve(yn, yp, sb['margin'], pid)])

    # --- 2) EM prior: validate procedure on OOF, then real test ---
    pi_train = np.bincount(y, minlength=10) / len(y)
    pi_em_oof, Qadj = em_prior(P, pi_train)
    yp_adj = np.array(SPECIES)[Qadj.argmax(1)]
    mo = evaluate(yn, yp_adj, Qadj.max(1), pid)
    print("OOF-EM pi_test_hat:", np.round(pi_em_oof, 3))
    print("OOF-EM train-prior:", np.round(pi_train, 3))
    print("OOF labels-after-EM-adjust: total=%.4f macro=%.4f (vs base macro 0.3894)" % (
        mo["total"], mo["macro_f1"]))
    Pt = np.load(ROOT / "test_m2_probs.npy").astype(float)
    pi_test, Qtest = em_prior(Pt, pi_train)
    print("REAL test pi_hat:", dict(zip(SPECIES, np.round(pi_test, 3).tolist())))
    print("REAL test train-prior:", dict(zip(SPECIES, np.round(pi_train, 3).tolist())))
    np.save(ROOT / "test_m2_probs_em.npy", Qtest.astype(np.float32))
    calib = {"T_global": T0, "T_perclass": Tc.tolist(),
             "pi_train": pi_train.tolist(), "pi_test_hat": pi_test.tolist(),
             "oof_selective_maxprob": base_sel}
    with open(ROOT / "calib.json", "w") as f:
        json.dump(calib, f, indent=1)
    print("saved calib.json + test_m2_probs_em.npy")


if __name__ == "__main__":
    main()
