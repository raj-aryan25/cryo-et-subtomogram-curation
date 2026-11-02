"""Phase 2a: TTA probability stats (cached). Views: clean, wedge49, blur05
(from caches), shift(+2x), shift(-2z) (computed here). OOF via 3-fold
E3-augmented MLP head + override; test via full-train E3 model. Logits saved
for energy. Cache: research/tta_stats.npz.
Run: python research/phase2_tta.py (CPU, ~20 min incl. OOF training)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta
from features_m2 import pervol_norm, radial_profile, spatial_feats, FEAT_NAMES
from metrics import SPECIES

ROOT = Path(__file__).resolve().parent
TABULAR = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
COLS29 = [i for i, n in enumerate(FEAT_NAMES) if n not in TABULAR]
torch.manual_seed(42)
np.random.seed(42)
EPOCHS = 100


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Dropout(drop), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def fit_models(Xa, y, tri, epochs=EPOCHS):
    """Train 10-way + 2 specialist MLPs on E3-augmented rows. Returns models
    as (state, scaler) tuples + scalers for specialists."""
    def train(Xt, yt, ncls, d_hid, ep):
        sc = StandardScaler().fit(Xt)
        A = sc.transform(Xt).astype(np.float32)
        freq = np.bincount(yt, minlength=ncls).astype(np.float64)
        cw = torch.tensor(len(yt) / (ncls * freq), dtype=torch.float32)
        net = MLP(A.shape[1], d_hid, ncls)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        XA, YA = torch.tensor(A), torch.tensor(yt, dtype=torch.long)
        net.train()
        for _ in range(ep):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(XA), YA, weight=cw)
            loss.backward()
            opt.step()
        return net.eval(), sc
    Xa3 = {k: Xa[k][tri] for k in ["clean", "blur05", "wedge49"]}
    Xbig = np.concatenate([Xa3[k] for k in ["clean", "blur05", "wedge49"]])
    ybig = np.concatenate([y[tri]] * 3)
    m10, sc10 = train(Xbig, ybig, 10, 64, epochs)
    specs = {}
    yn = np.array([SPECIES[i] for i in y])
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        m = np.isin(yn[tri], [oc, cc])
        Sb = np.concatenate([Xa[k][tri][m] for k in ["clean", "blur05", "wedge49"]])
        yb = np.concatenate([(yn[tri][m] == oc).astype(int)] * 3)
        specs[(oc, cc)] = train(Sb, yb, 2, 32, 150)
    return (m10, sc10), specs


@torch.no_grad()
def logits_of(model_tuple, X):
    net, sc = model_tuple
    return net(torch.tensor(sc.transform(X).astype(np.float32))).numpy()


def shift_view(v, dx, dy, dz):
    p = np.pad(np.asarray(v, dtype=np.float32), 2)
    return p[2 + dx:34 + dx, 2 + dy:34 + dy, 2 + dz:34 + dz].copy()


def feats29(vols):
    n = len(vols)
    X = np.zeros((n, 29), dtype=np.float32)
    for i, vv in enumerate(vols):
        vn = pervol_norm(np.asarray(vv, dtype=np.float32))
        prof = radial_profile(vn)
        s = spatial_feats(vn)
        row = list(prof) + [s[k] for k in FEAT_NAMES[14:29]]
        X[i] = np.nan_to_num(np.array(row, dtype=np.float32))
    return X


def tta_stats(models, specs, views, yn_rows=None):
    """views: dict name->X (same rows). Returns dict of stat arrays using
    10-way logits; override applied for pred/agree bookkeeping."""
    L = {k: logits_of(models, v) for k, v in views.items()}
    P = {k: None for k in views}
    for k in views:
        e = np.exp(L[k] - L[k].max(1, keepdims=True))
        P[k] = e / e.sum(1, keepdims=True)
    base = P["clean"].argmax(1)
    agree = np.zeros(len(base))
    for k in views:
        if k != "clean":
            agree += (P[k].argmax(1) == base)
    agree /= (len(views) - 1)
    p1 = P["clean"].max(1)
    srt = np.sort(P["clean"], axis=1)
    margin = srt[:, -1] - srt[:, -2]
    ent = -(P["clean"] * np.log(P["clean"] + 1e-12)).sum(1)
    energy = -np.log(np.exp(L["clean"]).sum(1))
    mp1 = np.stack([P[k].max(1) for k in views]).mean(0)
    ment = np.stack([-(P[k] * np.log(P[k] + 1e-12)).sum(1) for k in views]).mean(0)
    sp1 = np.stack([P[k].max(1) for k in views]).std(0)
    return {"logits": L["clean"].astype(np.float32),
            "p1": p1, "margin": margin, "ent": ent, "energy": energy,
            "tta_mp1": mp1, "tta_agree": agree, "tta_sp1": sp1,
            "tta_ment": ment, "pred": base}


def main():
    train, test = load_meta()
    yn = train["label"].values
    y = np.array([SPECIES.index(c) for c in yn])
    groups = train["tomogram_id"].values
    tr = np.load(ROOT.parent / "data" / "train_volumes.npy", mmap_mode="r")
    te = np.load(ROOT.parent / "data" / "test_volumes.npy", mmap_mode="r")
    za = np.load(ROOT / "cache_aug.npz")
    Xa = {k: za[k][:, COLS29] for k in ["clean", "blur05", "wedge49"]}

    print("featurizing shift views (train)...")
    Xshx = feats29([shift_view(tr[i], 2, 0, 0) for i in range(len(tr))])
    Xshz = feats29([shift_view(tr[i], 0, 0, -2) for i in range(len(tr))])
    print("featurizing shift views (test)...")
    Tshx = feats29([shift_view(te[i], 2, 0, 0) for i in range(len(te))])
    Tshz = feats29([shift_view(te[i], 0, 0, -2) for i in range(len(te))])
    zt = np.load(ROOT / "cache_m2.npz")
    Xt_all = zt["X_test"][:, [i for i, n in enumerate(zt["names"]) if n not in TABULAR]]
    # test aug views from full cache_aug (has test? no - only train). Build:
    print("featurizing test blur05/wedge49...")
    from scipy.ndimage import gaussian_filter
    az_te = test["tilt_axis_azimuth_deg"].values
    KX = np.fft.fftfreq(32) * 32
    KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
    RF = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)

    def wmask(tilt, az):
        phi = np.deg2rad(az)
        kn = -KXm * np.sin(phi) + KYm * np.cos(phi)
        return ((np.abs(KZm) <= np.tan(np.deg2rad(tilt)) * np.abs(kn)) | (RF < 1.5)).astype(np.float32)

    Tblur = feats29([gaussian_filter(np.asarray(te[i], dtype=np.float32), 0.5)
                     for i in range(len(te))])
    Tw = []
    for i in range(len(te)):
        v = np.asarray(te[i], dtype=np.float32)
        F = np.fft.fftn(v)
        Tw.append(np.real(np.fft.ifftn(F * wmask(49.0, az_te[i]))).astype(np.float32))
        if (i + 1) % 500 == 0:
            print(f"  wedge {i + 1}/{len(te)}", flush=True)
    Twedge = feats29(Tw)

    out = {}
    # OOF
    n = len(y)
    acc = {k: np.zeros(n, dtype=np.float32) for k in
           ["p1", "margin", "ent", "energy", "tta_mp1", "tta_agree", "tta_sp1", "tta_ment"]}
    acc["logits"] = np.zeros((n, 10), dtype=np.float32)
    acc["pred"] = np.zeros(n, dtype=np.int64)
    for tri, vai in GroupKFold(3).split(Xa["clean"], y, groups):
        models, specs = fit_models(Xa, y, tri)
        views = {"clean": Xa["clean"][vai], "wedge49": Xa["wedge49"][vai],
                 "blur05": Xa["blur05"][vai], "shx": Xshx[vai], "shz": Xshz[vai]}
        st = tta_stats(models, specs, views)
        for k in acc:
            acc[k][vai] = st[k]
        # apply plain override to OOF pred
        yp = np.array(SPECIES)[st["pred"]]
        for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                       ("atpase_open", "atpase_closed")]:
            net_s, sc_s = specs[(oc, cc)]
            mva = np.isin(yn[vai], [oc, cc])
            with torch.no_grad():
                s = torch.softmax(net_s(torch.tensor(
                    sc_s.transform(views["clean"][mva]).astype(np.float32))), 1)[:, 1].numpy()
            vv = np.where(mva)[0]
            use = np.isin(yp, [oc, cc])
            for k2, pos in enumerate(vv):
                if use[pos]:
                    yp[pos] = oc if s[k2] > 0.5 else cc
        acc["pred"][vai] = np.array([SPECIES.index(c) for c in yp])
        print("  fold done", flush=True)
    out["oof"] = acc
    # test (full-train model)
    tri_all = np.arange(n)
    models, specs = fit_models(Xa, y, tri_all)
    views_te = {"clean": Xt_all, "wedge49": Twedge, "blur05": Tblur,
                "shx": Tshx, "shz": Tshz}
    st = tta_stats(models, specs, views_te)
    yp = np.array(SPECIES)[st["pred"]]
    for oc, cc in [("chaperonin_open", "chaperonin_closed"),
                   ("atpase_open", "atpase_closed")]:
        net_s, sc_s = specs[(oc, cc)]
        with torch.no_grad():
            s = torch.softmax(net_s(torch.tensor(
                sc_s.transform(views_te["clean"]).astype(np.float32))), 1)[:, 1].numpy()
        use = np.isin(yp, [oc, cc])
        yp[use] = np.where(s[use] > 0.5, oc, cc)
    st["pred"] = np.array([SPECIES.index(c) for c in yp])
    out["test"] = st
    np.savez_compressed(ROOT / "tta_stats.npz",
                        **{f"oof_{k}": v for k, v in acc.items()},
                        **{f"test_{k}": v for k, v in st.items()})
    print("wrote research/tta_stats.npz")
    # quick disagreement summary
    print(f"OOF tta_agree mean={acc['tta_agree'].mean():.3f} "
          f"test tta_agree mean={st['tta_agree'].mean():.3f}")


if __name__ == "__main__":
    main()
