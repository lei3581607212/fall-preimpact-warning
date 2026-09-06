"""Aggregate matched v2 feature ablations and prepare F4/T4 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 7, 13, 21, 42, 123)
FOLDS = (0, 1, 2, 3, 4)
FULL_ROOT = ROOT / "outputs/paper_track/paper_family_disjoint_v2_5x6"
ABLATION_ROOT = ROOT / "outputs/paper_track/paper_family_disjoint_v2_matched_ablation_5x6"
FIGURE_DIR = ROOT / "outputs/paper_track/v2_figure_data"
SUMMARY_DIR = ROOT / "outputs/paper_track/v2_summary"
REPS = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def heldout(root: Path, variant: str | None, seed: int, fold: int) -> dict | None:
    if variant is None:
        path = root / f"seed{seed}" / f"outer{fold}" / "heldout_event_evaluation.json"
    else:
        path = root / variant / f"seed{seed}" / f"outer{fold}" / "heldout_event_evaluation.json"
    return read_json(path)


def run_status(root: Path, variant: str | None, seed: int, fold: int) -> str:
    if variant is None:
        run_dir = root / f"seed{seed}" / f"outer{fold}"
    else:
        run_dir = root / variant / f"seed{seed}" / f"outer{fold}"
    if (run_dir / "infeasible_policy_summary.json").exists():
        return "infeasible"
    result = heldout(root, variant, seed, fold)
    if result is None:
        return "missing"
    far = float(result.get("false_alert_rate", float("nan")))
    return "pass" if np.isfinite(far) and far <= 1 / 12 else "far_exceeded"


def bootstrap(values: np.ndarray, seed: int = 20260905) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    samples = np.asarray([
        float(np.mean(values[rng.randint(0, len(values), len(values))]))
        for _ in range(REPS)
    ])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true", help="Write rows even when not all 30 runs exist.")
    args = parser.parse_args()
    variants = {
        "full": {"profile": "paper_physics_v1", "input_dim": 80},
        "no_physics": {"profile": "paper_raw_v1", "input_dim": 68},
        "no_quality": {"profile": "paper_physics_no_quality_v1", "input_dim": 76},
    }
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for variant, info in variants.items():
        status_counts = {"pass": 0, "far_exceeded": 0, "infeasible": 0, "missing": 0}
        if variant == "full":
            rows = []
            for seed in SEEDS:
                for fold in FOLDS:
                    status = run_status(FULL_ROOT, None, seed, fold)
                    status_counts[status] += 1
                    result = heldout(FULL_ROOT, None, seed, fold)
                    rows.append((seed, fold, status, result))
        else:
            rows = []
            for seed in SEEDS:
                for fold in FOLDS:
                    status = run_status(ABLATION_ROOT, variant, seed, fold)
                    status_counts[status] += 1
                    result = heldout(ABLATION_ROOT, variant, seed, fold)
                    rows.append((seed, fold, status, result))
        if variant == "full":
            continue
        deltas = []
        for seed, fold, ab_status, ab_result in rows:
            full_result = heldout(FULL_ROOT, None, seed, fold)
            full_status = run_status(FULL_ROOT, None, seed, fold)
            delta = None
            if ab_result is not None and full_result is not None:
                delta = float(ab_result.get("recall_at_0.50s", 0.0)) - float(full_result.get("recall_at_0.50s", 0.0))
                deltas.append(delta)
            pair_rows.append({
                "variant": variant, "seed": seed, "fold": fold,
                "full_status": full_status, "ablation_status": ab_status,
                "full_recall_050": "" if full_result is None else full_result.get("recall_at_0.50s"),
                "ablation_recall_050": "" if ab_result is None else ab_result.get("recall_at_0.50s"),
                "delta_recall_050": "" if delta is None else delta,
                "full_far": "" if full_result is None else full_result.get("false_alert_rate"),
                "ablation_far": "" if ab_result is None else ab_result.get("false_alert_rate"),
            })
        values = np.asarray(deltas, dtype=float)
        lo, hi = bootstrap(values, 9100 + len(variant))
        summary_rows.append({
            "variant": variant,
            "feature_profile": info["profile"],
            "input_dim": info["input_dim"],
            "paired_runs": len(values),
            "expected_runs": 30,
            "ablation_complete_runs": 30 - status_counts["infeasible"] - status_counts["missing"],
            "ablation_pass_runs": status_counts["pass"],
            "ablation_far_exceeded_runs": status_counts["far_exceeded"],
            "ablation_infeasible_runs": status_counts["infeasible"],
            "delta_recall_050": "" if not len(values) else float(np.mean(values)),
            "ci_low": "" if not len(values) else lo,
            "ci_high": "" if not len(values) else hi,
            "eligible": bool(len(values) > 0 and (args.allow_partial or status_counts["missing"] == 0)),
            "status": "complete" if status_counts["missing"] == 0 and len(values) > 0 else "partial",
            "bootstrap_reps": REPS,
            "bootstrap_unit": "paired seed-fold run differences",
        })
    fields = [
        "variant", "feature_profile", "input_dim", "paired_runs", "expected_runs",
        "ablation_complete_runs", "ablation_pass_runs", "ablation_far_exceeded_runs",
        "ablation_infeasible_runs", "delta_recall_050", "ci_low", "ci_high",
        "eligible", "status", "bootstrap_reps", "bootstrap_unit",
    ]
    write_csv(FIGURE_DIR / "ablation_results.csv", summary_rows, fields)
    write_csv(SUMMARY_DIR / "ablation_pair_results.csv", pair_rows, [
        "variant", "seed", "fold", "full_status", "ablation_status",
        "full_recall_050", "ablation_recall_050", "delta_recall_050", "full_far", "ablation_far",
    ])
    metadata = {
        "protocol": "paper_family_disjoint_v2",
        "seeds": list(SEEDS), "outer_folds": list(FOLDS), "far_budget": 1 / 12,
        "variants": variants, "bootstrap_reps": REPS,
        "bootstrap_unit": "paired seed-fold run differences",
        "full_root": str(FULL_ROOT), "ablation_root": str(ABLATION_ROOT),
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "ablation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": summary_rows, "pair_count": len(pair_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
