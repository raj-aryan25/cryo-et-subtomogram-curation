"""Phase 3: tiny GN 3D-CNN (~200k params, no BN/SE) + fingerprint-destroying aug.

Design: full-32^3 input, per-vol norm, GN8+SiLU, GAP head. Aug per train view:
octahedral flips+rot90 (exact), translate +-3 (pad+crop, exact), gain U[0.5,2]
+offset with re-norm, blur sigma~U[0,0.8] (p=0.4), wedge-dropout tilt~U[40,60]
(p=0.5), spectral band rescale (p=0.3), neighbour-paste 8^3 (p=0.3).
FORBIDDEN here: erase/CutMix/MixUp/elastic/continuous-rotate, tabular inputs.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

N = 32


def _groups(c, cap=8):
    g = min(cap, c)
    while g > 1 and c % g != 0:
        g -= 1
    return g
_ax = np.arange(N) - (N - 1) / 2
XX, YY, ZZ = np.meshgrid(_ax, _ax, _ax, indexing="ij")
R_SPAT = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
KX = np.fft.fftfreq(N) * N
KXm, KYm, KZm = np.meshgrid(KX, KX, KX, indexing="ij")
R_FREQ = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2)
BINS = np.linspace(0.5, 16.0, 15)
BIN_IDX = np.clip(np.digitize(R_FREQ.ravel(), BINS) - 1, 0, 13)


class BasicBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        g1, g2 = _groups(co), _groups(co)
        self.conv = nn.Sequential(
            nn.Conv3d(ci, co, 3, stride, 1, bias=False),
            nn.GroupNorm(g1, co), nn.SiLU(inplace=True),
            nn.Conv3d(co, co, 3, 1, 1, bias=False),
            nn.GroupNorm(g2, co))
        self.skip = (nn.Sequential(nn.Conv3d(ci, co, 1, stride, bias=False),
                                   nn.GroupNorm(_groups(co), co))
                     if (ci != co or stride != 1) else nn.Identity())
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class TinyNet(nn.Module):
    """~200k params. Returns (logits10, gap_embedding)."""

    def __init__(self, drop=0.4, widths=(8, 16, 32, 48)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv3d(1, w0, 3, 1, 1, bias=False),
            nn.GroupNorm(_groups(w0), w0), nn.SiLU(inplace=True),
            nn.Conv3d(w0, w0, 3, 1, 1, bias=False),
            nn.GroupNorm(_groups(w0), w0), nn.SiLU(inplace=True))
        self.s2 = BasicBlock(w0, w1, 2)
        self.s3 = nn.Sequential(BasicBlock(w1, w2, 2), BasicBlock(w2, w2))
        self.s4 = BasicBlock(w2, w3, 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(w3, 10)

    def forward(self, x):
        h = self.s4(self.s3(self.s2(self.stem(x))))
        g = self.pool(h).flatten(1)
        return self.fc(self.drop(g)), g


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def pervol_norm(v):
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    s = v.std()
    return (v - v.mean()) / (s + 1e-6)


def octahedral_exact(v, rng):
    for ax in range(3):
        if rng.random() < 0.5:
            v = np.flip(v, axis=ax)
    return np.ascontiguousarray(np.rot90(v, int(rng.integers(0, 4)), axes=(0, 1)))


def wedge_dropout(v, rng):
    tmax = float(rng.uniform(40, 60))
    phi = float(rng.uniform(0, 180))
    kk = np.deg2rad(phi)
    kn = -KXm * np.sin(kk) + KYm * np.cos(kk)
    edge = np.tan(np.deg2rad(tmax)) * np.abs(kn) + 0.75
    M = np.clip((edge - np.abs(KZm)) / (0.15 * N + 1e-9) + 0.5, 0, 1)
    return np.real(np.fft.ifftn(np.fft.fftn(v) * M)).astype(np.float32)


def spectral_rescale(v, rng):
    F = np.fft.fftn(v)
    P = np.abs(F) ** 2
    scale = np.ones_like(P)
    for b in range(14):
        m = (BIN_IDX.reshape(P.shape) == b)
        if m.any():
            scale[m] = float(rng.uniform(0.7, 1.4))
    return np.real(np.fft.ifftn(F * np.sqrt(scale))).astype(np.float32)


def translate_pad(v, rng, pad=3):
    p = np.pad(v, pad)
    i, j, k = (int(rng.integers(0, 2 * pad + 1)) for _ in range(3))
    return np.ascontiguousarray(p[i:i + N, j:j + N, k:k + N])


def augment_train(v32, donor32_or_none, rng):
    v = v32.astype(np.float32)
    # gain/offset fingerprint destruction BEFORE norm
    v = v * float(rng.uniform(0.5, 2.0)) + float(rng.uniform(-0.5, 0.5))
    if rng.random() < 0.5:
        v = wedge_dropout(v, rng)
    if rng.random() < 0.4:
        v = gaussian_filter(v, float(rng.uniform(0, 0.8))).astype(np.float32)
    if rng.random() < 0.3:
        v = spectral_rescale(v, rng)
    v = octahedral_exact(v, rng)
    v = translate_pad(v, rng)
    if donor32_or_none is not None and rng.random() < 0.3:
        d = np.asarray(donor32_or_none, dtype=np.float32)
        d = d - d.mean()
        s = 8
        a, b, c = (int(rng.integers(0, N - s)) for _ in range(3))
        v[a:a + s, b:b + s, c:c + s] += 0.7 * d[a:a + s, b:b + s, c:c + s]
    return np.ascontiguousarray(pervol_norm(v), dtype=np.float32)


class TinyDataset(Dataset):
    def __init__(self, vols_mmap, labels, idx, train, seed=0,
                 donor_idx=None):
        self.vols = vols_mmap
        self.labels = np.asarray(labels, dtype=np.int64)
        self.idx = np.asarray(idx, dtype=np.int64)
        assert len(self.labels) == len(self.idx)
        self.train = train
        self.seed = seed
        self.donor_idx = (np.asarray(donor_idx, dtype=np.int64)
                          if donor_idx is not None else None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i * 7919)
        v = np.asarray(self.vols[self.idx[i]], dtype=np.float32)
        if self.train:
            d = None
            if self.donor_idx is not None:
                d = self.vols[int(rng.choice(self.donor_idx))]
            v = augment_train(v, d, rng)
        else:
            v = np.ascontiguousarray(pervol_norm(v), dtype=np.float32)
        return torch.from_numpy(v).unsqueeze(0), torch.tensor(self.labels[i])
