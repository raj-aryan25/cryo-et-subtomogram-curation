"""M1: 3D SE-ResNet18-narrow with hierarchical conformer heads.

Heads: 10-way (primary) + coarse-8 (conformer pairs merged) + 2 binary fine.
Input: 28^3 crop from 32^3. ~1.1M params.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

SPECIES = ["ribosome_60s", "proteasome_20s", "chaperonin_open",
           "chaperonin_closed", "atpase_open", "atpase_closed",
           "ferritin_shell", "filament_seg", "membrane_frag",
           "unassigned_density"]
CLS = {c: i for i, c in enumerate(SPECIES)}
# coarse-8: merge chap pair -> 2, atpase pair -> 3
COARSE = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6, 7])
N_COARSE = 8


class SE3D(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                nn.Linear(c, max(4, c // r)), nn.ReLU(inplace=True),
                                nn.Linear(max(4, c // r), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(-1, x.shape[1], 1, 1, 1)


def _norm(norm, c):
    if norm == "bn":
        return nn.BatchNorm3d(c)
    g = min(8, c)
    return nn.GroupNorm(g, c)


class Block(nn.Module):
    def __init__(self, ci, co, stride=1, groups=8, norm="gn"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ci, co, 3, stride, 1, bias=False), _norm(norm, co),
            nn.ReLU(inplace=True),
            nn.Conv3d(co, co, 3, 1, 1, bias=False), _norm(norm, co))
        self.se = SE3D(co)
        self.skip = nn.Sequential(nn.Conv3d(ci, co, 1, stride, bias=False),
                                  _norm(norm, co)) if (ci != co or stride != 1) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.skip(x))


class SEResNet3D(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128), emb=128, drop=0.3,
                 radial_dim=0, norm="gn"):
        super().__init__()
        self.radial_dim = radial_dim
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, 1, 1, bias=False),
                                  _norm(norm, widths[0]), nn.ReLU(inplace=True))
        self.l1 = self._layer(widths[0], widths[0], 2, 1, norm)
        self.l2 = self._layer(widths[0], widths[1], 2, 2, norm)
        self.l3 = self._layer(widths[1], widths[2], 2, 2, norm)
        self.l4 = self._layer(widths[2], widths[3], 2, 2, norm)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fuse = (nn.Linear(widths[3] + radial_dim, emb) if radial_dim
                     else nn.Linear(widths[3], emb))
        self.head_emb = nn.Sequential(nn.Flatten(), self.fuse,
                                      nn.LayerNorm(emb), nn.ReLU(inplace=True),
                                      nn.Dropout(drop))
        self.fc10 = nn.Linear(emb, 10)
        self.fc8 = nn.Linear(emb, N_COARSE)
        self.fc_chap = nn.Linear(emb, 1)   # open(1) vs closed(0), chaperonin
        self.fc_atp = nn.Linear(emb, 1)    # open(1) vs closed(0), atpase

    def _layer(self, ci, co, n, stride, norm):
        return nn.Sequential(Block(ci, co, stride, norm=norm),
                             *[Block(co, co, norm=norm) for _ in range(n - 1)])

    def forward(self, x, radial=None):
        h = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        f = self.pool(h)
        if self.radial_dim:
            assert radial is not None
            f = torch.cat([f, radial.view(-1, self.radial_dim, 1, 1, 1)
                           .expand(-1, -1, f.shape[2], f.shape[3], f.shape[4])], dim=1)
        e = self.head_emb(f)
        return {"logits10": self.fc10(e), "logits8": self.fc8(e),
                "chap": self.fc_chap(e).squeeze(1), "atp": self.fc_atp(e).squeeze(1)}


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- augmentation (numpy, per-sample, on 32^3 float32) ----------------
def octahedral(v, rng):
    # 8 flips x 4 xy-rot90 = 32 (covers octahedral group incl. z-flips)
    for ax in range(3):
        if rng.random() < 0.5:
            v = np.flip(v, axis=ax)
    v = np.rot90(v, rng.integers(0, 4), axes=(0, 1))
    return np.ascontiguousarray(v)


def wedge_dropout(v, rng, tmax_deg=None, phi_deg=None, soft=0.15):
    """Zero a random extra wedge in Fourier space (mimics narrower test tilt)."""
    n = v.shape[0]
    if tmax_deg is None:
        tmax_deg = float(rng.uniform(44, 54))
    if phi_deg is None:
        phi_deg = float(rng.uniform(0, 180))
    phi = np.deg2rad(phi_deg)
    k = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    b = -KX * np.sin(phi) + KY * np.cos(phi)
    c = np.abs(KZ)
    edge = np.abs(b) * np.tan(np.deg2rad(tmax_deg)) + 0.75
    M = np.clip((edge - c) / (soft * n + 1e-9) + 0.5, 0, 1)  # 1=keep
    V = np.fft.fftn(v) * M
    return np.real(np.fft.ifftn(V)).astype(np.float32)


def augment(v32, rng, train=True, wedge_p=0.5, erase_p=0.5):
    v = v32.astype(np.float32)
    if train and rng.random() < wedge_p:
        v = wedge_dropout(v, rng)
    if train:
        v = octahedral(v, rng)
        if rng.random() < 0.8:  # gain/contrast + offset jitter (post-norm-ish)
            v = v * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.15, 0.15))
        if rng.random() < 0.5:  # extra noise
            v = v + rng.normal(0, float(rng.uniform(0.05, 0.25)),
                               size=v.shape).astype(np.float32)
    # per-vol norm
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    v = (v - v.mean()) / (s + 1e-6)
    # crop 32->28 (train: random => +-2 shift; val: centre)
    if train:
        i, j, k = (int(rng.integers(0, 5)) for _ in range(3))
    else:
        i = j = k = 2
    v = v[i:i + 28, j:j + 28, k:k + 28]
    if train and rng.random() < erase_p:  # crowding/occlusion erase (mean-fill=0)
        e = 6
        a, b, c = (int(rng.integers(0, 28 - e)) for _ in range(3))
        v[a:a + e, b:b + e, c:c + e] = 0
    return np.ascontiguousarray(v, dtype=np.float32)


class VolDataset(Dataset):
    def __init__(self, vols_mmap, labels, idx, train, seed=0, wedge_p=0.5,
                 erase_p=0.5, ret_pos=False, feats=None):
        self.vols = vols_mmap
        self.labels = np.asarray(labels, dtype=np.int64)
        self.idx = np.asarray(idx, dtype=np.int64)
        assert len(self.labels) == len(self.idx), "labels/idx length mismatch"
        self.train = train
        self.seed = seed
        self.wedge_p = wedge_p
        self.erase_p = erase_p
        self.ret_pos = ret_pos
        self.feats = feats  # (N_full, F) float32 radial side-input or None
        self.vols = vols_mmap
        self.labels = np.asarray(labels, dtype=np.int64)
        self.idx = np.asarray(idx, dtype=np.int64)
        assert len(self.labels) == len(self.idx), "labels/idx length mismatch"
        self.train = train
        self.seed = seed
        self.wedge_p = wedge_p
        self.erase_p = erase_p
        self.ret_pos = ret_pos

    def __len__(self):
        return len(self.labels)

    def augment_wrap(self, i, rng):
        v = augment(np.asarray(self.vols[self.idx[i]], dtype=np.float32), rng,
                    train=self.train, wedge_p=self.wedge_p, erase_p=self.erase_p)
        return v

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i * 7919)
        v = self.augment_wrap(i, rng)
        x = torch.from_numpy(v).unsqueeze(0)
        y = torch.tensor(self.labels[i])
        if self.feats is not None:
            f = torch.from_numpy(np.asarray(self.feats[self.idx[i]],
                                            dtype=np.float32))
            if self.ret_pos:
                return x, y, f, torch.tensor(i)
            return x, y, f
        if self.ret_pos:
            return x, y, torch.tensor(i)
        return x, y
