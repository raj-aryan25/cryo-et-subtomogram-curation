"""CORAL OOF safety check: align each val fold to its train fold, logreg/74."""
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta
from metrics import evaluate, SPECIES
from phase6b_align import coral_to_source, fit_lr

ROOT = Path(__file__).resolve().parent

train, _ = load_meta()
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
for tag, align in [("none", False), ("coral", True)]:
    tots = []
    for tri, vai in GroupKFold(3).split(X, y, groups):
        B = X[vai] if not align else coral_to_source(X[tri], X[vai])
        P = fit_lr(X[tri], y[tri], B)
        yp = np.array(SPECIES)[P.argmax(1)]
        m = evaluate(yn[vai], yp, P.max(1), pid[vai])
        tots.append(m["total"])
    print(f"OOF {tag}: total={np.mean(tots):.4f}", flush=True)
