# Fall Pre-Impact Warning Paper Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22538536.svg)](https://doi.org/10.5281/zenodo.22538536)

This repository contains reviewer-facing, privacy-screened code and data
products for the family-disjoint v2 paper protocol.

## Protocol snapshot

- 47,381 causal windows from 530 source events and 284 physical groups;
- family-disjoint nested five-fold evaluation;
- six fixed seeds: 0, 7, 13, 21, 42, and 123;
- development-only policy selection under FAR <= 1/12;
- 30 terminal runs for the paper teacher and each matched baseline;
- matched feature ablations and reconstructed false-alert action categories.

The v2 manifest excludes all 15 known full-video/event-clip aliases. Known
physical-family overlap across outer folds is zero. The older v1 manifest is
included only for historical audit and must not be used for paper claims.

## Included data products

- `paper_family_disjoint_v2_manifest_public.csv`: pseudonymized v2 outer-fold
  manifest with dataset, fold, event type, and window count;
- `paper_matched_5fold_manifest_public.csv`: legacy v1 audit manifest;
- `release_summary.json`: sanitized protocol and headline-result summary;
- `results/`: aggregate main, baseline, ablation, Bootstrap, and failure-mode tables;
- `figure_data/`: tidy aggregate inputs for the registered paper figures;
- `figures/`: final paper figures without source video frames;
- `scripts/`, `tools/`, and `utils/`: the minimal v2 training, evaluation,
  aggregation, plotting, split-audit, and feature-contract implementation;
- `tests/`: focused protocol, baseline, and event-policy tests;
- `REPRODUCIBILITY.md`: environment and command sequence;
- `CITATION.cff`: software citation metadata;
- `THIRD_PARTY_DATA.md`: upstream provenance and redistribution boundaries.

## Data boundary

This release contains no raw video, image, audio, pose array, checkpoint,
participant name, e-mail address, device identifier, or local absolute path.
Private FieldCollection media and third-party dataset copies are not included.
The public manifests use deterministic truncated SHA-256 identifiers; they are
pseudonymous split-audit keys, not public participant identifiers.

## Release status

Version `v0.1.1` passed its privacy, manifest, checksum, compilation, and
focused test checks and has been archived in Zenodo.

The archived `v0.1.1` release is available at
<https://doi.org/10.5281/zenodo.22538537>. The all-versions concept DOI is
<https://doi.org/10.5281/zenodo.22538536>.
