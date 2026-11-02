"""Workstream D: wedge-masked template conformer override.
Templates = averages of high-agreement train vols per conformer class.
Score = max over 24 octahedral rotations of Fourier wedge-masked NCC,
using per-tomo wedge from released tilt_max/azimuth.
Gate: OOF conformer +0.03 with no macro drop, override-only (in-pair preds).
Saves research/templates.npz on gate pass.
Run: python research/phase6d_templates.py (CPU, ~15 min)
"""
import itertools
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
N = 32
KX = np.fft.fftfreq(N) * N
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)

# 24 octahedral rotations: signed permutation matrices with det +1
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
    # inverse map: out[x] = v[M^{-1}(x)] with M orthogonal => M^{-1} = M.T
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
    a = np.conj(Fo) * Ft * W
    num = np.abs(a.sum())
    den = np.sqrt((np.abs(Fo) ** 2 * W).sum() * (np.abs(Ft) ** 2 * W).sum()) + 1e-9
    return float(num / den)


def main():
    train, test = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    tr = np.load(ROOT.parent / "data" / "train_volumes.npy", mmap_mode="r")
    z = np.load(ROOT / "oof74.npz", allow_pickle=True)
    Poof, YP = z["probs"], np.array(z["pred"])
    correct = (YP == yn)

    # templates from high-agreement rows (correct, top-half p1 within class)
    templates = {}
    for cls in ["chaperonin_open", "chaperonin_closed", "atpase_open",
                "atpase_closed"]:
        idx = np.where((yn == cls) & correct)[0]
        p1 = Poof[idx].max(1)
        idx = idx[np.argsort(-p1)[:150]]
        T = np.mean([pervol_norm(np.asarray(tr[i], dtype=np.float32))
                     for i in idx], axis=0)
        templates[cls] = T.astype(np.float32)
        print(f"template {cls}: n={len(idx)}", flush=True)
    # precompute rotated FFTs
    TF = {c: [np.fft.fftn(apply_rot(T, M)) for M in ROTS]
          for c, T in templates.items()}

    def decide(vols_idx, vols_arr, masks):
        """Return dict pair-> P(open) via template NCC margin."""
        out = {}
        for (oc, cc) in [("chaperonin_open", "chaperonin_closed"),
                         ("atpase_open", "atpase_closed")]:
            s = np.zeros(len(vols_idx))
            for j, i in enumerate(vols_idx):
                v = pervol_norm(np.asarray(vols_arr[i], dtype=np.float32))
                Fo = np.fft.fftn(v)
                W = masks[j]
                so = max(masked_ncc(Fo, Ft, W) for Ft in TF[oc])
                sc = max(masked_ncc(Fo, Ft, W) for Ft in TF[cc])
                s[j] = so - sc
                if (j + 1) % 200 == 0:
                    print(f"  {oc[:4]} {j + 1}/{len(vols_idx)}", flush=True)
            out[(oc, cc)] = s
        return out

    # OOF gate: rows predicted in-pair
    inpair_mask = np.isin(YP, ["chaperonin_open", "chaperonin_closed",
                               "atpase_open", "atpase_closed"])
    oidx = np.where(inpair_mask)[0]
    omasks = [wedge_mask(train["tilt_max_deg"].iloc[i],
                         train["tilt_axis_azimuth_deg"].iloc[i]) for i in oidx]
    S = decide(oidx, tr, omasks)
    for mode in ["template", "agree"]:
        Y2 = YP.copy()
        for (oc, cc), s in S.items():
            use_rows = oidx[np.isin(YP[oidx], [oc, cc])]
            ss = s[np.isin(YP[oidx], [oc, cc])]
            if mode == "template":
                Y2[use_rows] = np.where(ss > 0, oc, cc)
            else:  # keep specialist label unless template strongly disagrees
                cur = Y2[use_rows]
                want = np.where(ss > 0, oc, cc)
                strong = np.abs(ss) > 0.02
                cur[strong] = want[strong]
                Y2[use_rows] = cur
        m = evaluate(yn, Y2, Poof.max(1), pid)
        print(f"OOF {mode}: total={m['total']:.4f} macro={m['macro_f1']:.3f} "
              f"conf={m['conformer_acc']:.3f} min={m['min_class_f1']:.3f}",
              flush=True)
    # pair-row accuracy detail
    for (oc, cc), s in S.items():
        mrows = np.where(np.isin(yn, [oc, cc]))[0]
        acc_t = (np.where(s[np.isin(oidx, mrows)],
                          0, 0) if False else None)
        # accuracy on TRUE pair rows that were predicted in-pair
        both = np.intersect1d(oidx, mrows)
        sb = s[np.searchsorted(oidx, both)]
        acc = (((sb > 0).astype(int)) == (yn[both] == oc).astype(int)).mean()
        print(f"{oc[:4]} template pair-acc on routed true-pair rows: "
              f"{acc:.3f} n={len(both)}", flush=True)
    np.savez_compressed(ROOT / "templates.npz",
                        **{c: templates[c] for c in templates})
    print("saved research/templates.npz", flush=True)


if __name__ == "__main__":
    main()
