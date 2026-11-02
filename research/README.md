# Research archive

These are the experiment scripts used while developing the final solution. They are
preserved for transparency — they are not a packaged library and some are one-off runs.

The polished explanation of the method and its failures is in
[`../docs/SOLUTION_GUIDE.md`](../docs/SOLUTION_GUIDE.md). The chronological lab notebook
is [`CONTEXT.md`](CONTEXT.md).

## Phase map

| Area | Scripts |
|---|---|
| Evaluation utilities | `metrics.py`, `splits.py` |
| M2 feature baseline | `features_m2.py`, `features_m2b.py`, `train_m2.py`, `eval_m2cache.py` |
| Conformer specialists / MLP parity | `specialists.py`, `specialists2.py`, `exp_mlp.py` |
| Diagnostics | `diag0_dist.py`, `diag0_stress.py` |
| Phase 1 — spectral features and augmentation | `phase1_feats.py`, `phase1_aug.py`, `phase1_eval.py`, `phase1_match.py` |
| Phase 2 — TTA and confidence ranking | `phase2_tta.py`, `phase2_rank.py` |
| Phase 4–5 — offsets, self-training, ensembles | `phase4_offsets.py`, `phase4_selftrain.py`, `phase5_ens.py`, `phase5_epochs.py`, `phase5_final.py`, `phase5_noise.py`, `phase5_r1e120.py`, `phase5_validate.py` |
| Phase 6 — 74-feature bank, calibration, templates | `phase6_bank.py`, `phase6_eval.py`, `phase6_full.py`, `phase6b_align.py`, `phase6b_oof.py`, `phase6c_conf.py`, `phase6c_gate.py`, `phase6c_tune.py`, `phase6d_templates.py`, `phase6e_final.py`, `phase6g_final2.py` |
| 3D-CNN attempts on Modal A10G | `model_m1.py`, `model_m3.py`, `modal_m1.py`, `modal_m3.py` |
| Legacy final pipeline / calibration | `predict_test.py`, `calibrate.py` |

Small result files (`calib.json`, `m2_report.json`, `diag0_*.json`, `diag0_pertomo.csv`)
are kept alongside the scripts.

## Running the archived scripts

- The dataset is expected in `../data` relative to this folder, using the same layout as
  [`../data/README.md`](../data/README.md).
- Intermediate caches (`.npz`, `.npy`) are **not committed**. Regenerate the main one
  with:

  ```bash
  python research/features_m2.py   # writes research/cache_m2.npz
  ```

  Several later scripts depend on that cache and on additional caches written by the
  phase scripts.
- Outputs are written into `research/`; the legacy `predict_test.py` writes
  `../submissions/submission.csv`.
- `modal_m1.py` and `modal_m3.py` are the Modal A10G launchers for the rejected 3D-CNN
  experiments. They require `pip install modal` and upload `research/` to `/opt/working`
  in the container.

The final, self-contained deliverable is [`../solution.py`](../solution.py); it does not
need any cache generated here.
