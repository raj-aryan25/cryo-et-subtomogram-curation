"""Workstream C tuning from saved OOF (fast, no retraining).
Loads research/oof74.npz; tunes class multipliers (greedy) + hinge trust;
reports selective on OOF and pseudo-CD. Saves research/conf74.npz.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent

train, test = load_meta()
yn = train["label"].values
pid = train["particle_id"].values
z = np.load(ROOT / "oof74.npz", allow_pickle=True)
Poof = z["probs"]
YP = np.array(z["pred"])
base = evaluate(yn, YP, Poof.max(1), pid)
print(f"maxprob: sel={base['selective_accuracy']:.4f} total={base['total']:.4f}",
      flush=True)

pred_idx = np.array([SPECIES.index(c) for c in YP])
W = np.ones(10)
grid = [0.7, 0.85, 1.0, 1.15, 1.3]
for _pass in range(3):
    improved = False
    for c in range(10):
        cur = evaluate(yn, YP, Poof.max(1) * W[pred_idx], pid)["selective_accuracy"]
        best_s, best_g = cur, W[c]
        for g in grid:
            W[c] = g
            s = evaluate(yn, YP, Poof.max(1) * W[pred_idx], pid)["selective_accuracy"]
            if s > best_s + 1e-4:
                best_s, best_g = s, g
        W[c] = best_g
        improved |= (best_g != cur and False)
    print(f"pass done W={W.round(2)}", flush=True)
mc = evaluate(yn, YP, Poof.max(1) * W[pred_idx], pid)
print("class_w:", dict(zip(SPECIES, W.round(2))), flush=True)
print(f"class_w: sel={mc['selective_accuracy']:.4f} (delta "
      f"{mc['selective_accuracy']-base['selective_accuracy']:+.4f})", flush=True)

meta_tr = train[["ice_thickness_nm", "defocus_um", "tilt_max_deg"]].values
base_idx, pseudo_idx, hold = pseudo_cd_holdout(train)
for k1, k2, k3 in [(0.5, 0.05, 0.3), (1.0, 0.1, 0.5), (2.0, 0.2, 1.0)]:
    trust = np.exp(-k1 * np.maximum(0, meta_tr[:, 1] - 3.5)
                   - k2 * np.maximum(0, meta_tr[:, 0] - 150)
                   - k3 * np.maximum(0, 55 - meta_tr[:, 2]))
    cf = Poof.max(1) * W[pred_idx] * trust
    mo = evaluate(yn, YP, cf, pid)
    mp = evaluate(yn[pseudo_idx], YP[pseudo_idx], cf[pseudo_idx],
                  pid[pseudo_idx])
    print(f"hinge({k1},{k2},{k3}): OOF-sel={mo['selective_accuracy']:.4f} "
          f"pseudo-sel={mp['selective_accuracy']:.4f}", flush=True)
# hinge-only (no class_w) for comparison
for k1, k2, k3 in [(1.0, 0.1, 0.5)]:
    trust = np.exp(-k1 * np.maximum(0, meta_tr[:, 1] - 3.5)
                   - k2 * np.maximum(0, meta_tr[:, 0] - 150)
                   - k3 * np.maximum(0, 55 - meta_tr[:, 2]))
    cf = Poof.max(1) * trust
    mo = evaluate(yn, YP, cf, pid)
    mp = evaluate(yn[pseudo_idx], YP[pseudo_idx], cf[pseudo_idx],
                  pid[pseudo_idx])
    print(f"hinge-only({k1},{k2},{k3}): OOF-sel={mo['selective_accuracy']:.4f} "
          f"pseudo-sel={mp['selective_accuracy']:.4f}", flush=True)
np.savez_compressed(ROOT / "conf74.npz", class_w=W)
print("wrote research/conf74.npz", flush=True)
