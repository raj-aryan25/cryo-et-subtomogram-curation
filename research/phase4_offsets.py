"""Offset-risk check on OOF (train-prior regime): clipped uniform-prior offsets."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import load_meta
from metrics import evaluate, SPECIES

ROOT = Path(__file__).resolve().parent
train, _ = load_meta()
yn = train["label"].values
pid = train["particle_id"].values
y = np.array([SPECIES.index(c) for c in yn])
o = np.load(ROOT / "oof_m2.npz")["oof"]
yp = np.array(SPECIES)[o.argmax(1)]
m0 = evaluate(yn, yp, o.max(1), pid)
prior = np.bincount(y, minlength=10) / len(y)
for clip in [0.15, 0.3, 0.5]:
    off = np.clip(np.log((0.1 + 1e-9) / (prior + 1e-9)), -clip, clip)
    P = o * np.exp(off)[None, :]
    P /= P.sum(1, keepdims=True)
    m = evaluate(yn, np.array(SPECIES)[P.argmax(1)], P.max(1), pid)
    d = m["total"] - m0["total"]
    print("clip=%.2f total=%.4f delta=%+.4f macro=%.4f minF1=%.4f" % (
        clip, m["total"], d, m["macro_f1"], m["min_class_f1"]))
print("base      total=%.4f macro=%.4f minF1=%.4f" % (
    m0["total"], m0["macro_f1"], m0["min_class_f1"]))
