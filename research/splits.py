"""GroupKFold-by-tomogram splits + pseudo-CD (simulated test-shift) holdout."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

DATA = Path(__file__).resolve().parent.parent / "data"


def load_meta():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


def group_folds(train, n=3, seed=42):
    gkf = GroupKFold(n_splits=n)
    y = train["label"].values
    g = train["tomogram_id"].values
    return list(gkf.split(np.zeros(len(train)), y, g))


def pseudo_cd_holdout(train, frac=0.2, seed=42):
    """Hold out ~frac of tomograms most resembling test (thick ice, high defocus, narrow tilt).

    Score per tomogram = rank(ice) + rank(defocus) - rank(tilt_max); take top-frac.
    Returns (base_idx, pseudo_idx) over rows.
    """
    t = train.groupby("tomogram_id")[["ice_thickness_nm", "defocus_um",
                                       "tilt_max_deg"]].first()
    score = (t["ice_thickness_nm"].rank() + t["defocus_um"].rank()
             - t["tilt_max_deg"].rank())
    n_hold = max(4, int(round(frac * len(t))))
    hold_tomos = set(score.sort_values(ascending=False).head(n_hold).index)
    is_hold = train["tomogram_id"].isin(hold_tomos).values
    idx = np.arange(len(train))
    return idx[~is_hold], idx[is_hold], sorted(hold_tomos)
