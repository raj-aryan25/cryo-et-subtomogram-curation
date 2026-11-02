"""Noise quantification: R1 recipe (clean, no offsets) full OOF at 2 seeds."""
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


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 123
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, _ = load_meta()
    yn = train["label"].values
    pid = train["particle_id"].values
    groups = train["tomogram_id"].values
    y = np.array([SPECIES.index(c) for c in yn])
    za = np.load(ROOT / "cache_aug.npz")
    Xa = {k: za[k] for k in ["clean", "blur05", "wedge49"]}
    # rebuild module-level determinism for this seed (module set its own)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tots = []
    for tri, vai in GroupKFold(3).split(Xa["clean"], y, groups):
        torch.manual_seed(seed + int(vai[0]))
        m = F.run_recipe(Xa, y, yn, pid, tri, vai, False, False)
        tots.append(m["total"])
        print(f"seed={seed} fold total={m['total']:.4f}", flush=True)
    print(f"R1 seed={seed} ep={F.EPOCHS} OOF total={np.mean(tots):.4f}")


if __name__ == "__main__":
    main()
