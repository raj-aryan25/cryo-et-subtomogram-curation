"""R1 @120ep full OOF (matches exp_mlp config) to resolve epoch question."""
import numpy as np
import sys
from pathlib import Path
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_final as F
from splits import load_meta
from metrics import SPECIES

ROOT = Path(__file__).resolve().parent
F.EPOCHS = 120
torch.manual_seed(7)
np.random.seed(7)
train, _ = load_meta()
yn = train["label"].values
pid = train["particle_id"].values
groups = train["tomogram_id"].values
y = np.array([SPECIES.index(c) for c in yn])
za = np.load(ROOT / "cache_aug.npz")
Xa = {k: za[k] for k in ["clean", "blur05", "wedge49"]}
tots = []
for tri, vai in GroupKFold(3).split(Xa["clean"], y, groups):
    torch.manual_seed(7 + int(vai[0]))
    m = F.run_recipe(Xa, y, yn, pid, tri, vai, False, False)
    tots.append(m["total"])
    print("fold total=%.4f macro=%.3f sel=%.3f" % (
        m["total"], m["macro_f1"], m["selective_accuracy"]), flush=True)
print("R1@120 OOF total=%.4f" % np.mean(tots))
