"""Pseudo-CD gate for class_w confidence (fast)."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta, pseudo_cd_holdout
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
train, _ = load_meta()
yn = train["label"].values
pid = train["particle_id"].values
z = np.load(ROOT / "oof74.npz", allow_pickle=True)
P, YP = z["probs"], np.array(z["pred"])
W = np.load(ROOT / "conf74.npz")["class_w"]
pi = np.array([SPECIES.index(c) for c in YP])
_, pseudo_idx, _ = pseudo_cd_holdout(train)
for nm, cf in [("maxprob", P.max(1)), ("class_w", P.max(1) * W[pi])]:
    a = evaluate(yn, YP, cf, pid)
    b = evaluate(yn[pseudo_idx], YP[pseudo_idx], cf[pseudo_idx],
                 pid[pseudo_idx])
    print(nm, "OOF-sel=%.4f" % a["selective_accuracy"],
          "pseudo-sel=%.4f" % b["selective_accuracy"], flush=True)
