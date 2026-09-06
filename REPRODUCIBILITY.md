# Reproducibility guide

## Environment

Python 3.8 on Windows was used for the recorded experiments. Install the pinned
packages from the repository root:

```powershell
python -m pip install -r requirements-lock.txt
```

The K230 deployment environment is separate from this paper-teacher protocol.
Do not treat a paper checkpoint as a deployable K230 model.

## Frozen data contract

- feature profile: `paper_physics_v1`;
- input: 80 float features per frame;
- window: 64 causal frames;
- split: family-disjoint v2 manifest;
- window weighting: `loss_only`;
- policy selection: development split only, FAR <= 1/12.

The private/local NPZ is intentionally not redistributed. Reproduction starts
from locally obtained upstream datasets and the repository preprocessing code.

## Audit and command preview

```powershell
python scripts/build_dataset_release_materials.py
python scripts/audit_paper_v2_preflight.py
python scripts/run_paper_v2_full.py --dry-run
python scripts/run_paper_v2_matched_ablation.py --dry-run
python scripts/run_paper_v2_tcnte_baseline.py --baseline tcnte --dry-run
python scripts/run_paper_v2_tcnte_baseline.py --baseline pure_tcn --dry-run
```

The runners fix seeds to `0 7 13 21 42 123` and outer folds to `0 1 2 3 4`.

## Full experiment and aggregation

```powershell
python scripts/run_paper_v2_full.py
python scripts/run_paper_v2_matched_ablation.py
python scripts/run_paper_v2_tcnte_baseline.py --baseline tcnte
python scripts/run_paper_v2_tcnte_baseline.py --baseline pure_tcn
python scripts/aggregate_paper_v2_results.py
python scripts/aggregate_paper_v2_matched_ablation.py
python scripts/aggregate_paper_v2_baselines.py
python scripts/reconstruct_v2_false_alert_actions.py
python scripts/aggregate_v2_false_alert_actions.py
python scripts/plot_paper_figures_v2.py
python scripts/audit_paper_submission.py
```

Each runner saves new checkpoint filenames and records the commands,
development policy, outer event evaluation, hashes, logs, and terminal status.
Development-infeasible runs remain reported and are not replaced by a weaker
false-alarm constraint.

## Interpretation boundary

The paper teacher completed 30 runs: 20 were development-feasible and received
outer evaluation, 10 were development-infeasible, and 8 of 30 satisfied the
outer FAR budget. Baseline paired recall intervals cross zero, so this release
does not claim stable superiority over either matched baseline or public SOTA.
