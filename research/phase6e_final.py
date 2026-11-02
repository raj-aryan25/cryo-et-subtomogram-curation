"""Final gate: (1) split-half honesty check for templates; (2) full pipeline
OOF + pseudo-CD with/without CORAL (MLP-ens3 + specialists + template-override
+ class_w conf). Winner = shipped recipe.
Run: python research/phase6e_final.py (CPU, ~30 min)
"""
import itertools
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
N = 32
np.random.seed(42)
KX = np.fft.fftfreq(N) * N
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)
ROTS = []
for perm in itertools.permutations([0, 1, 2]):
    for signs in itertools.product([-1, 1], repeat=3):
        M = np.zeros((3, 3), dtype=int)
        for i, p in enumerate(perm):
            M[i, p] = signs[i]
        if int(round(np.linalg.det(M))) == 1:
            ROTS.append(M)
assert len(ROTS) == 24


def apply_rot(v, M):
    idx = np.stack(np.meshgrid(np.arange(N), np.arange(N), np.arange(N),
                               indexing="ij"), axis=-1).reshape(-1, 3) - 15.5
    src = (idx @ M.T + 15.5).round().astype(int)
    src = np.clip(src, 0, N - 1)
    return v[src[:, 0], src[:, 1], src[:, 2]].reshape(N, N, N)


def wedge_mask(tilt_deg, az_deg):
    phi = np.deg2rad(az_deg)
    kn = -KXm * np.sin(phi) + KYm * np.cos(phi)
    edge = np.tan(np.deg2rad(tilt_deg)) * np.abs(kn)
    return ((np.abs(KZm) <= edge) | (R_FREQ < 1.5)).astype(np.float32)


def pervol_norm(v):
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    return (v - v.mean()) / (s + 1e-6)


def masked_ncc(Fo, Ft, W):
    num = np.abs((np.conj(Fo) * Ft * W).sum())
    den = np.sqrt((np.abs(Fo) ** 2 * W).sum() * (np.abs(Ft) ** 2 * W).sum()) + 1e-9
    return float(num / den)


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp_ens(Xtr, ytr, Xte, ncls, d_hid, epochs, seeds):
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr).astype(np.float32)
    B = sc.transform(Xte).astype(np.float32)
    freq = np.bincount(ytr, minlength=ncls).astype(np.float64)
    cw = torch.tensor(len(ytr) / (ncls * freq), dtype=torch.float32)
    XA = torch.tensor(A)
    YA = torch.tensor(ytr, dtype=torch.long)
    XB = torch.tensor(B)
    P = 0
    for sd in seeds:
        torch.manual_seed(sd)
        net = MLP(A.shape[1], d_hid, ncls)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(XA), YA, weight=cw)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            P = P + torch.softmax(net(XB), 1).numpy()
    return P / len(seeds)


def coral_to_source(Xs, Xt, eps=1e-3):
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs.T) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + eps * np.eye(Xt.shape[1])
    es, Vs = np.linalg.eigh(Cs)
    es = np.clip(es, 1e-9, None)
    et, Vt = np.linalg.eigh(Ct)
    et = np.clip(et, 1e-9, None)
    A = (Vt @ np.diag(1.0 / np.sqrt(et)) @ Vt.T) @ \
        (Vs @ np.diag(np.sqrt(es)) @ Vs.T)
    return (Xt - mu_t) @ A + mu_s


def build_templates(vols_arr, yn_arr, oof_probs, oof_pred, take_idx, n_each=150):
    """Average high-agreement rows within take_idx. Returns dict + rot FFTs."""
    correct = (oof_pred == yn_arr)
    T = {}
    for cls in ["chaperonin_open", "chaperonin_closed", "atpase_open",
                "atpase_closed"]:
        idx = np.where((yn_arr == cls) & correct)[0]
        idx = np.intersect1d(idx, take_idx)
        p1 = oof_probs[idx].max(1)
        idx = idx[np.argsort(-p1)[:n_each]]
        T[cls] = np.mean([pervol_norm(np.asarray(vols_arr[i], dtype=np.float32))
                          for i in idx], axis=0).astype(np.float32)
    return T


def template_scores(TF, vols_arr, ridx, masks):
    out = {}
    for (oc, cc) in [("chaperonin_open", "chaperonin_closed"),
                     ("atpase_open", "atpase_closed")]:
        s = np.zeros(len(ridx))
        for j, i in enumerate(ridx):
            v = pervol_norm(np.asarray(vols_arr[i], dtype=np.float32))
            Fo = np.fft.fftn(v)
            W = masks[j]
            so = max(masked_ncc(Fo, Ft, W) for Ft in TF[oc])
            sc = max(masked_ncc(Fo, Ft, W) for Ft in TF[cc])
            s[j] = so - sc
        out[(oc, cc)] = s
    return out


def template_pair_scores(TF_oc, TF_cc, vols_arr, grows, masks):
    """NCC-margin (open minus closed), aligned with grows."""
    s = np.zeros(len(grows))
    for j, g in enumerate(grows):
        v = pervol_norm(np.asarray(vols_arr[g], dtype=np.float32))
        Fo = np.fft.fftn(v)
        W = masks[j]
        so = max(masked_ncc(Fo, Ft, W) for Ft in TF_oc)
        sc = max(masked_ncc(Fo, Ft, W) for Ft in TF_cc)
        s[j] = so - sc
    return s


def main():
    train, test = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    z2 = np.load(ROOT / "cache_m2.npz")
    names = list(z2["names"])
    tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
    cc = [i for i, n in enumerate(names) if n not in tab]
    zb = np.load(ROOT / "cache_bank.npz")
    X = np.concatenate([z2["X_train"][:, cc], zb["X_train"]], axis=1)
    tr = np.load(ROOT.parent / "data" / "train_volumes.npy", mmap_mode="r")
    z = np.load(ROOT / "oof74.npz", allow_pickle=True)
    Poof, YP = z["probs"], np.array(z["pred"])
    Wconf = np.load(ROOT / "conf74.npz")["class_w"]
    pi_full = np.array([SPECIES.index(c) for c in YP])

    # ---- (1) split-half honesty: templates from even rows, eval odd routed ----
    inpair = np.isin(YP, ["chaperonin_open", "chaperonin_closed",
                          "atpase_open", "atpase_closed"])
    oidx = np.where(inpair)[0]
    ev, od = oidx[::2], oidx[1::2]
    Tev = build_templates(tr, yn, Poof, YP, ev)
    TFev = {c: [np.fft.fftn(apply_rot(T, M)) for M in ROTS] for c, T in Tev.items()}
    masks_od = [wedge_mask(train["tilt_max_deg"].iloc[i],
                           train["tilt_axis_azimuth_deg"].iloc[i]) for i in od]
    S = template_scores(TFev, tr, od, masks_od)
    for (oc, cc), s in S.items():
        mrows = np.where(np.isin(yn[od], [oc, cc]))[0]
        acc = ((((s[mrows] > 0).astype(int)) == (yn[od][mrows] == oc).astype(int)).mean()
               if len(mrows) else float("nan"))
        print(f"SPLIT-HALF {oc[:4]}: pair-acc={acc:.3f} n={len(mrows)}", flush=True)
    # and swapped
    Tod = build_templates(tr, yn, Poof, YP, od)
    TFod = {c: [np.fft.fftn(apply_rot(T, M)) for M in ROTS] for c, T in Tod.items()}
    masks_ev = [wedge_mask(train["tilt_max_deg"].iloc[i],
                           train["tilt_axis_azimuth_deg"].iloc[i]) for i in ev]
    S2 = template_scores(TFod, tr, ev, masks_ev)
    for (oc, cc), s in S2.items():
        mrows = np.where(np.isin(yn[ev], [oc, cc]))[0]
        acc = ((((s[mrows] > 0).astype(int)) == (yn[ev][mrows] == oc).astype(int)).mean()
               if len(mrows) else float("nan"))
        print(f"SPLIT-HALF-swapped {oc[:4]}: pair-acc={acc:.3f} n={len(mrows)}",
              flush=True)

    # ---- (2) full pipeline OOF + pseudo with/without CORAL ----
    SEEDS = [7, 42, 123]

    def run_all(tri, vai, use_coral):
        Xva = coral_to_source(X[tri], X[vai]) if use_coral else X[vai]
        P = fit_mlp_ens(X[tri], y[tri], Xva, 10, 64, 200, SEEDS)
        yp = np.array(SPECIES)[P.argmax(1)].copy()
        # template override on in-pair preds (templates from take=tri rows)
        T = build_templates(tr, yn, Poof, YP, tri)
        TF = {c: [np.fft.fftn(apply_rot(Tv, M)) for M in ROTS]
              for c, Tv in T.items()}
        for (oc, cc) in [("chaperonin_open", "chaperonin_closed"),
                         ("atpase_open", "atpase_closed")]:
            inp_idx = np.where(np.isin(yp, [oc, cc]))[0]
            if len(inp_idx) == 0:
                continue
            grows = vai[inp_idx]
            masks = [wedge_mask(train["tilt_max_deg"].iloc[g],
                                train["tilt_axis_azimuth_deg"].iloc[g])
                     for g in grows]
            s = template_pair_scores(TF[oc], TF[cc], tr, grows, masks)
            yp[inp_idx] = np.where(s > 0, oc, cc)
        return yp, P
    for use_coral in [False, True]:
        YP2 = np.empty(len(y), dtype=object)
        PP2 = np.zeros((len(y), 10))
        for tri, vai in GroupKFold(3).split(X, y, groups):
            yp, P = run_all(tri, vai, use_coral)
            YP2[vai], PP2[vai] = yp, P
        pi2 = np.array([SPECIES.index(c) for c in YP2])
        m = evaluate(yn, YP2, PP2.max(1) * Wconf[pi2], pid)
        print(f"OOF coral={use_coral}: total={m['total']:.4f} "
              f"macro={m['macro_f1']:.3f} conf={m['conformer_acc']:.3f} "
              f"min={m['min_class_f1']:.3f} sel={m['selective_accuracy']:.4f}",
              flush=True)
    base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
    for use_coral in [False, True]:
        yp, P = run_all(base_idx, pseudo_idx, use_coral)
        pi2 = np.array([SPECIES.index(c) for c in yp])
        m = evaluate(yn[pseudo_idx], yp, P.max(1) * Wconf[pi2], pid[pseudo_idx])
        print(f"pseudo coral={use_coral}: total={m['total']:.4f} "
              f"macro={m['macro_f1']:.3f} sel={m['selective_accuracy']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
