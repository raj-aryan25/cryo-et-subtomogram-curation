# Dataset

The dataset is **not committed** to this repository. The arrays are large
(`train_volumes.npy` is 179 MB) and the competition data is not redistributed here.

Place the released files in this folder before running `solution.py`:

| File | Rows / shape | Notes |
|---|---|---|
| `train.csv` | 2,861 rows | features + `label` |
| `test.csv` | 1,132 rows | features only |
| `train_volumes.npy` | `(2861, 32, 32, 32)` float16 | one 32³ volume per training row |
| `test_volumes.npy` | `(1132, 32, 32, 32)` float16 | one 32³ volume per test row |
| `species.json` | — | class list, conformer pairs, metadata |
| `sample_submission.csv` | 1,132 rows | format reference |

Expected columns in `train.csv`:

```text
volume_index, particle_id, tomogram_id, session, defocus_um,
tilt_max_deg, tilt_axis_azimuth_deg, ice_thickness_nm, pixel_size_a, label
```

`test.csv` is identical except it has no `label` column.

Quick integrity check:

```bash
python - <<'PY'
import json
import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
tr = np.load("data/train_volumes.npy", mmap_mode="r")
te = np.load("data/test_volumes.npy", mmap_mode="r")
species = json.load(open("data/species.json"))

assert tr.shape == (len(train), 32, 32, 32)
assert te.shape == (len(test), 32, 32, 32)
assert train["volume_index"].tolist() == list(range(len(train)))
assert test["volume_index"].tolist() == list(range(len(test)))
assert set(train["label"]) <= set(species["classes"])
print("dataset OK")
PY
```

Then run the final solution from the repository root:

```bash
python solution.py data submissions/submission.csv
```
