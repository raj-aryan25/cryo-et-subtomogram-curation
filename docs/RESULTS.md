# Results and experiment summary

All scores are out-of-fold (OOF) predictions from `GroupKFold(3)` grouped by
`tomogram_id`, so no tomogram is split between train and validation. "Pseudo-CD" is a
held-out subset of the most test-like training tomograms (thick ice, high defocus,
narrow tilt) used as a second, harder validation regime.

The competition metric is:

```text
0.30 · macro_F1 + 0.20 · conformer_accuracy
+ 0.15 · min_class_F1 + 0.35 · selective_accuracy
```

## Progression

| Stage | Features | Model | Total | Macro F1 | Conformer | Min F1 | Selective |
|---|---|---:|---:|---:|---:|---:|---:|
| M2 baseline | 29 | logistic regression | 0.394 | 0.389 | 0.287 | 0.206 | 0.539 |
| + conformer specialists | 29 | logreg + binary heads | 0.414 | 0.427 | 0.288 | 0.258 | 0.541 |
| MLP ensemble | 29 | 3-seed MLP | 0.4193 | — | — | — | — |
| Feature bank | 74 | logistic regression | 0.4701 | — | — | — | — |
| **Final** | **74** | **3-seed MLP + specialists** | **0.4773** | **0.474** | **0.327** | **0.244** | **0.666** |

The final recipe scored **0.4837** on the pseudo-CD holdout. Greedy per-class confidence
multipliers, bounded to `[0.7, 1.3]`, improved OOF selective accuracy from `0.6477` to
`0.6660` without changing any predicted labels.

## What worked

- **Per-volume normalisation** — mandatory because each tomogram has its own gain and
  offset.
- **Invariant features by construction** — at roughly −20 dB SNR, computing
  rotation/gain-invariant descriptors and fitting a small classifier generalised better
  than learning a 3D CNN from ~2.9k volumes.
- **Conformer specialists with plain in-pair override** — improves the fine-grained
  open/closed decision while keeping all 10 classes represented.
- **A richer invariant bank** — spherical-harmonic shell power, shell moments, and
  inertia-tensor ratios lifted logreg OOF from 0.395 to 0.470 on their own.
- **Seed ensembling** — a reliable variance reduction; every metric term improved.
- **Bounded per-class confidence calibration** — a small, validated gain on the
  selective-accuracy term.

## What did not work

- **3D CNNs (7 variants, 0.5–2.1M parameters).** All memorised per-tomogram
  fingerprints: training accuracy reached 1.0 while validation macro F1 stayed near
  chance. A tomogram-ID probe on the learned embedding was 5× chance.
- **Self-training / pseudo-labelling.** Only filament rows passed the conservative
  gate; the pseudo-validated delta was negative.
- **TTA-mean labels.** OOF total dropped from 0.41 to 0.33.
- **Wedge-masked template NCC.** Looked strong (pair accuracy 0.77) but a split-half
  honesty check showed chance-level performance — the gain was template self-inclusion
  leakage.
- **CORAL alignment.** Helped a logreg proxy but hurt the full MLP pipeline.
- **EM prior shift, detector cascade, TTA ranker, hinge tomogram trust, stacking.**
  Rejected by OOF and pseudo-CD gates.

The main lesson: with this data scale, invariance by construction beat learned features,
and every change had to clear an honest validation gate rather than look plausible.
