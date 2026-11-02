# Confidence-Aware Subtomogram Curation under a Missing Wedge — Challenge Description

## Overview
In this Computer Vision challenge, **Cryo-electron tomography (cryo-ET)** is the only technique that can image molecular machines in situ — inside intact, unstained, frozen cells. Once a tomogram is reconstructed, the scientific payload comes from **subtomogram classification**: cutting a small box around every candidate particle and deciding which macromolecular complex sits at its centre.

## Key Difficulties
This challenge embeds four major obstacles:
- **Dose-limited noise**: Real cryo-ET noise spectrum, strongly coloured at low spatial frequencies.
- **Missing wedge**: Tilt limited to ±50°, leaving wedge-shaped gaps in Fourier space. Reconstructions are elongated and anisotropically blurred.
- **Molecular crowding**: Boxes often contain fragments of neighbouring complexes from other classes.
- **Structural heterogeneity**: Variants with missing subunits make silhouette-based classification unreliable.

## Dataset
- **Training set**: 2,861 subtomograms from 100 tomograms
- **Test set**: 1,132 subtomograms from 40 tomograms
- **Classes**: 10 species, including ribosome_60s, proteasome_20s, chaperonin (open/closed), atpase (open/closed), ferritin_shell, filament_seg, membrane_frag, and unassigned_density.
- **Domain shift**: Test data has thicker ice, higher defocus, and narrower tilt range.
- **Conformers**: Two pairs of classes represent functional states (open/closed).

## Evaluation
Submissions are scored on a weighted combination of four terms (range: 0–1):
- **Macro F1 (30%)**: Mean per-class F1 across all species.
- **Conformer accuracy (20%)**: Exact identification of conformational states.
- **Min-class F1 (15%)**: Lowest per-class F1, ensuring weak classes are not ignored.
- **Selective accuracy (35%)**: Accuracy measured at confidence-ranked subsets (25%, 50%, 75%, 100%).

Confidence calibration is critical — knowing what you do not know is scored separately.

## Rules
- **Not allowed**:
  - Regenerating the dataset or recovering test labels from the generator.
  - Using withheld per-tomogram parameters (e.g., snr, crowding).
  - Hand-labelling test data or lookup tables keyed on particle_id.
  - External datasets, network access, or libraries outside the Kaggle Python Docker image.
  - LLM-generated content in submissions.

- **Allowed**:
  - Reimplementing the forward model.
  - Test-time augmentation, pseudo-labelling, and domain adaptation.
  - Using acquisition metadata (e.g., defocus, tilt axis).
  - Estimating withheld quantities (e.g., noise level, wedge geometry).

## Submission Format
- File: `./working/submission.csv`
- Columns: `particle_id`, `label`, `confidence`
- Requirements:
  - Exactly 1,132 rows + header.
  - Labels must match species.json.
  - Confidence values in [0,1]; ranking matters more than absolute scale.

---
