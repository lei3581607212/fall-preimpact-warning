"""Aggregate the same-protocol v2 TCNTE baseline and comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 7, 13, 21, 42, 123)
FOLDS = (0, 1, 2, 3, 4)
BASELINE_ROOT = ROOT / "outputs/paper_track/paper_family_disjoint_v2_tcnte_baseline_5x6"
TEACHER_ROOT = ROOT / "outputs/paper_track/paper_family_disjoint_v2_5x6"
OUT_DIR = ROOT / "outputs/paper_track/v2_summary"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def status(root: Path, seed: int, fold: int) -> tuple[str, dict | None]:
    run = root / f"seed{seed}" / f"outer{fold}"
    heldout = read_json(run / "heldout_event_evaluation.json")
    if heldout is not None:
        far = float(heldout.get("false_alert_rate", float("nan")))
        return ("pass" if np.isfinite(far) and far <= 1 / 12 else "far_exceeded"), heldout
    if (run / "infeasible_policy_summary.json").exists():
        return "infeasible", None
    return "missing", None


def aggregate(rows: list[dict]) -> dict:
    evaluated = [row for row in rows if row["status"] in {"pass", "far_exceeded"}]
    feasible = [row for row in evaluated if row["status"] == "pass"]
    out = {
        "total_runs": len(rows), "evaluated_runs": len(evaluated),
        "outer_far_feasible_runs": len(feasible),
        "outer_far_exceeded_runs": sum(row["status"] == "far_exceeded" for row in evaluated),
        "infeasible_runs": sum(row["status"] == "infeasible" for row in rows),
    }
    for label, subset in (("all_evaluated", evaluated), ("far_feasible", feasible)):
        out[f"{label}_far"] = float(np.average([row["far"] for row in subset], weights=[row["normal_events"] for row in subset])) if subset else None
        for metric in ("recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s", "mean_lead_seconds", "median_lead_seconds"):
            values = [row[metric] for row in subset if row.get(metric) is not None]
            weights = [row["fall_events"] for row in subset if row.get(metric) is not None]
            out[f"{label}_{metric}"] = float(np.average(values, weights=weights)) if values else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            run_status, heldout = status(BASELINE_ROOT, seed, fold)
            row = {"method": "TCNTE baseline", "seed": seed, "fold": fold, "status": run_status}
            if heldout:
                row.update({key: heldout.get(key) for key in (
                    "false_alert_rate", "false_alert_events", "fall_events", "normal_events",
                    "recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s", "mean_lead_seconds", "median_lead_seconds")})
                row["far"] = row.pop("false_alert_rate")
            else:
                row.update({"far": None, "false_alert_events": None, "fall_events": None, "normal_events": None,
                            "recall_at_0.25s": None, "recall_at_0.50s": None, "recall_at_1.00s": None,
                            "mean_lead_seconds": None, "median_lead_seconds": None})
            rows.append(row)
    if not args.allow_partial and any(row["status"] == "missing" for row in rows):
        raise RuntimeError("baseline is incomplete; use --allow-partial only for diagnostics")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["method", "seed", "fold", "status", "far", "false_alert_events", "fall_events", "normal_events",
              "recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s", "mean_lead_seconds", "median_lead_seconds"]
    with (OUT_DIR / "tcnte_baseline_seed_fold_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    comparison = []
    teacher_rows = []
    teacher_by_run = {}
    for seed in SEEDS:
        for fold in FOLDS:
            run = TEACHER_ROOT / f"seed{seed}" / f"outer{fold}"
            heldout = read_json(run / "heldout_event_evaluation.json")
            if heldout:
                teacher_row = {"status": "pass" if float(heldout["false_alert_rate"]) <= 1 / 12 else "far_exceeded",
                               "far": heldout["false_alert_rate"], "fall_events": heldout["fall_events"], "normal_events": heldout["normal_events"],
                               **{metric: heldout.get(metric) for metric in ("recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s", "mean_lead_seconds", "median_lead_seconds")}}
                teacher_rows.append(teacher_row)
                teacher_by_run[(seed, fold)] = teacher_row
            elif (run / "infeasible_policy_summary.json").exists():
                teacher_rows.append({"status": "infeasible"})
    for method, method_rows in (("Paper teacher", teacher_rows), ("TCNTE baseline", rows)):
        summary = aggregate(method_rows)
        summary["method"] = method
        comparison.append(summary)
    comp_fields = ["method", "total_runs", "evaluated_runs", "outer_far_feasible_runs", "outer_far_exceeded_runs", "infeasible_runs",
                   "all_evaluated_far", "all_evaluated_recall_at_0.25s", "all_evaluated_recall_at_0.50s", "all_evaluated_recall_at_1.00s",
                   "all_evaluated_mean_lead_seconds", "all_evaluated_median_lead_seconds", "far_feasible_far",
                   "far_feasible_recall_at_0.25s", "far_feasible_recall_at_0.50s", "far_feasible_recall_at_1.00s",
                   "far_feasible_mean_lead_seconds", "far_feasible_median_lead_seconds"]
    with (OUT_DIR / "sota_baseline_comparison_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comp_fields); writer.writeheader(); writer.writerows(comparison)
    (OUT_DIR / "sota_baseline_comparison_v2.json").write_text(json.dumps({"protocol": "paper_family_disjoint_v2", "rows": comparison}, indent=2), encoding="utf-8")
    baseline_by_run = {(int(row["seed"]), int(row["fold"])): row for row in rows if row["status"] in {"pass", "far_exceeded"}}
    paired_keys = sorted(set(teacher_by_run) & set(baseline_by_run))
    rng = np.random.RandomState(20260906)
    paired_rows = []
    for metric, weight_key in (
        ("far", "normal_events"),
        ("recall_at_0.25s", "fall_events"),
        ("recall_at_0.50s", "fall_events"),
        ("recall_at_1.00s", "fall_events"),
        ("mean_lead_seconds", "fall_events"),
        ("median_lead_seconds", "fall_events"),
    ):
        deltas = np.asarray([float(teacher_by_run[key][metric]) - float(baseline_by_run[key][metric]) for key in paired_keys])
        weights = np.asarray([float(teacher_by_run[key][weight_key]) for key in paired_keys])
        estimate = float(np.average(deltas, weights=weights))
        samples = []
        for _ in range(1000):
            index = rng.randint(0, len(paired_keys), len(paired_keys))
            samples.append(float(np.average(deltas[index], weights=weights[index])))
        paired_rows.append({
            "metric": metric, "paired_runs": len(paired_keys),
            "teacher_minus_tcnte": estimate,
            "ci_low": float(np.percentile(samples, 2.5)),
            "ci_high": float(np.percentile(samples, 97.5)),
            "bootstrap_reps": 1000, "bootstrap_unit": "paired seed-fold run",
        })
    with (OUT_DIR / "sota_baseline_paired_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
    paired_payload = {
        "protocol": "paper_family_disjoint_v2",
        "paired_runs": len(paired_keys),
        "paired_seed_folds": [{"seed": seed, "fold": fold} for seed, fold in paired_keys],
        "both_outer_far_feasible_runs": sum(
            teacher_by_run[key]["status"] == "pass" and baseline_by_run[key]["status"] == "pass"
            for key in paired_keys
        ),
        "bootstrap_reps": 1000,
        "bootstrap_unit": "paired seed-fold run",
        "rows": paired_rows,
    }
    (OUT_DIR / "sota_baseline_paired_v2.json").write_text(json.dumps(paired_payload, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(json.dumps(paired_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
