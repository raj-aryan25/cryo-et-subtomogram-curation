# Subtomogram Curation — Step-by-Step Solution Guide
*From scratch to the shipped solution (OOF 0.477), including everything that failed and why.*

---

## 0. The problem in one paragraph

You get 2,861 small 3D volumes (32×32×32 voxels, cryo-electron tomography data) from 100 tomograms (sessions AB) **with labels**, and 1,132 volumes from 40 *different* tomograms (sessions CD) **without labels**. Assign each test volume one of 10 macromolecule classes plus a confidence in [0,1]. The test sessions were acquired differently (thicker ice, higher defocus, narrower tilt range), so this is classification **under domain shift**, not i.i.d. classification.

## 1. What the score terms mean

Total = `0.30·macro_F1 + 0.20·conformer_acc + 0.15·min_class_F1 + 0.35·selective_accuracy` (higher is better, max 1).

- **macro_F1 (0.30):** F1 score per class, averaged uniformly over the 10 classes. Rare classes count as much as common ones, so you cannot ignore hard classes.
- **conformer_acc (0.20):** accuracy restricted to the 4 conformer rows (chaperonin open/closed, ATPase open/closed). These are the *same complex in two states*, differing by a ~4 Å domain shift — the hardest fine-grained distinction.
- **min_class_F1 (0.15):** the *worst* per-class F1. A weakest-link term: if you completely miss one class (predict it never, or always wrong), this is ~0 and drags the total. This is why every pipeline here has "all 10 classes present" guardrails.
- **selective_accuracy (0.35, the biggest term):** sort test rows by *your* confidence, compute accuracy on the top 25%, 50%, 75%, 100%, and average the four. Constant confidence ⇒ ranking is random ⇒ selective = overall accuracy (no gain). Points come only from confidence that genuinely separates right from wrong answers. **Only the ranking matters, not absolute values; ties break by particle_id.**

Reference scale: constant confidence ≈ 0.602 contribution basis, max-probability ≈ 0.654, oracle (knowing which rows are right) ≈ 0.683 on the curation part.

## 2. The data and its difficulties

- **Per-tomogram gain/offset (×3.4 span):** raw voxel values are incomparable across tomograms. ⇒ *Per-volume* normalization (clip 1–99 percentiles, z-score) is mandatory and is step 0 of every pipeline here.
- **Colored noise (SNR ≈ −20 dB):** single volumes are mostly noise; signal emerges statistically.
- **Missing wedge (lab frame):** tilt series cover only ±tilt_max (train 55–62°, test 44–54°) about a per-tomo azimuth. Fourier components outside are unmeasured. Applied *after* random particle rotation, so the corruption direction differs per particle.
- **Crowding:** 0–4 neighbours intrude through box faces (heavier in test). ⇒ Central-focused features help; box edges are suspect.
- **Centre jitter (σ 2.3 vox):** target is near, not at, box centre. ⇒ Translation-tolerant features/pooling.
- **Subunit deletion (≤26%, mass renormalised):** shape varies within a class; total mass carries no information.
- **3% random train label noise (~86 rows):** don't trust labels blindly; prefer regularized models.
- **Shift train→test:** ice 127→193 nm, defocus 2.8→4.4 µm, tilt 58.7→49.0°, heavier crowding, rare-enriched priors.
- **Validation setup used throughout:** GroupKFold (3 folds) grouped by `tomogram_id` (never split a tomogram across train/val — that leaks), plus a pseudo-CD holdout (most test-like AB tomos: ice>150/def>3.5/tilt<55). All scores below are OOF (out-of-fold) unless marked "test".

## 3. Step 1 — M2 baseline: invariant features + logistic regression (OOF 0.394)

**Idea:** at −20 dB SNR, don't learn features from 1,900 volumes — *compute* features that are invariant to the nuisances by construction, then fit a linear model.

- Radial Fourier power profile (14 shells): fully rotation-invariant; gain-normalized. A sphere, rod, shell, and cage have different radial spectra regardless of orientation.
- Spatial cues computed *after* normalization (so gain-invariant): skew, kurtosis, tail fractions, percentiles, central/mid/outer shell masses, z-elongation ratio, 90°-rotation and mirror self-correlations (these separate rods/planes/shells).
- Acquisition metadata was tried as extra features and **dropped**: defocus slopes extrapolate badly off-range (test atpase_open predictions 398→115 when dropped).
- Balanced logistic regression (29 image-only features). Result: **OOF total 0.394** (macro .389 / conf .287 / min .206 / sel .539) — ~3× the best reference (0.1404).

*Lesson: invariance by construction beats learned features at this data scale.*

## 4. Step 2 — conformer specialists + plain override (OOF 0.414)

**Idea:** a 10-way head spreads capacity across all classes; the 4 Å open/closed boundary deserves a dedicated binary classifier trained *only* on that pair's rows.
- Two logreg specialists (C=0.1). Override rule: replace the 10-way label **only where it already predicts inside the pair** — this preserves all-10-class support by construction (protects min_class_F1).
- Result: **OOF 0.414** (macro .427 / min .258). Small, validated, safe.

## 5. Step 3 — torch MLPs + 3-seed ensemble (OOF 0.419)

Same 29 features; replace logreg with small MLPs (10-way 29→64, specialists 29→32), class-weighted cross-entropy, CUDA-if-available. A 3-seed prediction average gives **OOF 0.4193** — pure variance reduction, every term up. (This version scored test **0.258**.)

## 6. Everything that failed (equally instructive)

- **3D CNNs, 7 variants (0.5–2.1M params, BN and GN, distillation, radial fusion, residual learning, tiny 216k net + heavy augmentation):** all memorized tomogram fingerprints — train acc → 1.0, val macro stuck at chance (~0.10); tomo-ID probe 5× chance. With ~67 tomos/fold, fingerprint SNR beats biology SNR. Killed per kill-rule; DANN judged low-EV and skipped.
- **EM prior-shift:** OOF macro 0.39→0.19. Feeding miscalibrated posteriors back into priors diverges.
- **Detector cascade:** OOF 0.507 but predicted only 9/10 classes on test → minF1 = 0 guaranteed. Rejected by the presence gate.
- **Temperature scaling:** T=1.0 already optimal. **Stacking/LDA/HGB blends, extra M2b features:** no gain or worse.
- **Phase-flip + common-wedge masking:** all lose OOF (0.36–0.38 vs 0.39) — wrong CTF-sign convention suspected + pure information loss in-distribution.
- **Self-training:** −0.007 pseudo-validated; only filament rows pass the gate (collapse risk).
- **TTA-mean for labels:** 0.41→0.33 crater. **Extreme-tomo tempering:** −0.025 selective. **Clipped prior offsets:** unstable sign across regimes.
- **Correctness ranker:** +0.001. Max-probability is near-optimal for ranking this model class.
- **Wedge-masked template NCC:** looked great (pair-acc 0.77) but split-half honesty check = chance — the 0.77 was template self-inclusion leakage. *Always split-half leakage-suspect gains.*
- **CORAL alignment:** helped a logreg proxy (+0.011 pseudo) but hurt the real MLP pipeline (−0.010 OOF). Proxy evidence doesn't transfer across heads.

*Lesson: every idea here earned its place (or rejection) from a gate — OOF + pseudo-CD + honesty checks — never from intuition.*

## 7. Step 4 — richer invariant bank (OOF 0.471)

The one genuine new signal of the whole project: 45 more gain-invariant features — spherical-harmonic shell power (l=0–4 × 6 shells = 30), within-shell std/skew (12), inertia-tensor eigenvalue ratios (3, separating rods/planes/shells). Plain logreg jumped 0.395→0.470 on these alone. They shift mildly train→test (max 0.5σ vs 2.1σ for radial bins), so transfer is plausible. Full recipe (MLP-ens3 + specialists + override on 74 feats): **OOF 0.4713** (macro .474 / conf .327 / min .244 / sel .649), pseudo-CD 0.4803.

## 8. Step 5 — calibrated confidence that survived (selective 0.648→0.666)

Greedy per-class confidence multipliers bounded to [0.7, 1.3]: OOF selective +0.018, pseudo-CD +0.028. Small, bounded, adopted as fixed constants. (Hinge-style tomogram trust was rejected three times — any tomogram-varying scaling reorders rows and hurt everywhere.)

## 9. Final shipped recipe (solution.py, ~74 s CPU)

`per-volume norm → 74 invariant feats → 3-seed MLP ensemble (74→64→10) + ensembled binary specialists (74→32→1) → plain in-pair override → maxprob × per-class constant + row-order tie-break → submission.csv with guards (all 10 classes present, none >50%, else 10-way fallback)`. Test distribution is balanced across all classes.

## 10. Glossary

- **OOF (out-of-fold):** predictions on validation folds from models that never saw those rows — the honest offline score.
- **Pseudo-CD:** held-out extreme-AB tomos simulating test shift (weaker than the real thing — it never reproduced the test collapse, a key lesson).
- **Conformer:** same complex, two functional states (open/closed), 4 Å apart.
- **Missing wedge:** unmeasured Fourier region from limited tilt range; lab-frame, so orientation-dependent per particle.
- **CTF/defocus:** microscope contrast transfer; shifts with defocus between sessions.
- **Transductive:** using (unlabeled) test volumes at train time — allowed here.
- **minF1 collapse:** predicting a class never (or always wrong) zeroes the weakest-link term.
