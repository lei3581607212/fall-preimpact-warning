"""Aggregate teacher, TCNTE and pure-TCN under the frozen v2 protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from aggregate_paper_v2_tcnte_baseline import FOLDS, SEEDS, aggregate, status


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_track/v2_summary"
ROOTS = {
    "Paper teacher": ROOT / "outputs/paper_track/paper_family_disjoint_v2_5x6",
    "TCNTE baseline": ROOT / "outputs/paper_track/paper_family_disjoint_v2_tcnte_baseline_5x6",
    "Pure TCN baseline": ROOT / "outputs/paper_track/paper_family_disjoint_v2_tcn_baseline_5x6",
}
METRICS = ("far", "recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s",
           "mean_lead_seconds", "median_lead_seconds")


def collect(method: str, root: Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            run_status, heldout = status(root, seed, fold)
            row = {"method": method, "seed": seed, "fold": fold, "status": run_status}
            if heldout:
                row.update({
                    "far": heldout.get("false_alert_rate"),
                    "false_alert_events": heldout.get("false_alert_events"),
                    "fall_events": heldout.get("fall_events"),
                    "normal_events": heldout.get("normal_events"),
                    **{metric: heldout.get(metric) for metric in METRICS if metric != "far"},
                })
            else:
                row.update({key: None for key in ("far", "false_alert_events", "fall_events",
                                                   "normal_events", *METRICS[1:])})
            rows.append(row)
    return rows


def paired_comparison(teacher: list[dict], baseline: list[dict], baseline_name: str) -> tuple[list[dict], dict]:
    evaluated = {"pass", "far_exceeded"}
    teacher_by_key = {(row["seed"], row["fold"]): row for row in teacher if row["status"] in evaluated}
    baseline_by_key = {(row["seed"], row["fold"]): row for row in baseline if row["status"] in evaluated}
    keys = sorted(set(teacher_by_key) & set(baseline_by_key))
    if not keys:
        return [], {"baseline": baseline_name, "paired_runs": 0, "paired_seed_folds": [], "rows": []}
    rng = np.random.RandomState(20260906)
    rows = []
    for metric in METRICS:
        weight_key = "normal_events" if metric == "far" else "fall_events"
        deltas = np.asarray([float(teacher_by_key[key][metric]) - float(baseline_by_key[key][metric]) for key in keys])
        weights = np.asarray([float(teacher_by_key[key][weight_key]) for key in keys])
        samples = []
        for _ in range(1000):
            index = rng.randint(0, len(keys), len(keys))
            samples.append(float(np.average(deltas[index], weights=weights[index])))
        rows.append({
            "baseline": baseline_name, "metric": metric, "paired_runs": len(keys),
            "teacher_minus_baseline": float(np.average(deltas, weights=weights)),
            "ci_low": float(np.percentile(samples, 2.5)),
            "ci_high": float(np.percentile(samples, 97.5)),
            "bootstrap_reps": 1000, "bootstrap_unit": "paired seed-fold run",
        })
    payload = {
        "baseline": baseline_name, "paired_runs": len(keys),
        "paired_seed_folds": [{"seed": seed, "fold": fold} for seed, fold in keys],
        "both_outer_far_feasible_runs": sum(
            teacher_by_key[key]["status"] == "pass" and baseline_by_key[key]["status"] == "pass" for key in keys
        ),
        "rows": rows,
    }
    return rows, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    by_method = {method: collect(method, root) for method, root in ROOTS.items()}
    missing_by_method = {
        method: sum(row["status"] == "missing" for row in rows) for method, rows in by_method.items()
    }
    if not args.allow_partial:
        if any(missing_by_method.values()):
            raise RuntimeError(f"one or more methods are incomplete: {missing_by_method}")
    OUT.mkdir(parents=True, exist_ok=True)
    run_fields = ["method", "seed", "fold", "status", "far", "false_alert_events", "fall_events",
                  "normal_events", *METRICS[1:]]
    with (OUT / "all_baselines_seed_fold_results_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields); writer.writeheader()
        for rows in by_method.values(): writer.writerows(rows)
    summaries = []
    for method, rows in by_method.items():
        summary = aggregate(rows); summary["method"] = method; summaries.append(summary)
    summary_fields = ["method", "total_runs", "evaluated_runs", "outer_far_feasible_runs",
                      "outer_far_exceeded_runs", "infeasible_runs", "all_evaluated_far",
                      "all_evaluated_recall_at_0.25s", "all_evaluated_recall_at_0.50s",
                      "all_evaluated_recall_at_1.00s", "all_evaluated_mean_lead_seconds",
                      "all_evaluated_median_lead_seconds", "far_feasible_far",
                      "far_feasible_recall_at_0.25s", "far_feasible_recall_at_0.50s",
                      "far_feasible_recall_at_1.00s", "far_feasible_mean_lead_seconds",
                      "far_feasible_median_lead_seconds"]
    with (OUT / "all_baselines_comparison_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summaries)
    paired_rows, paired_payloads = [], []
    for baseline in ("TCNTE baseline", "Pure TCN baseline"):
        rows, payload = paired_comparison(by_method["Paper teacher"], by_method[baseline], baseline)
        paired_rows.extend(rows); paired_payloads.append(payload)
    if paired_rows:
        with (OUT / "all_baselines_paired_v2.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
    payload = {
        "protocol": "paper_family_disjoint_v2",
        "status": "PARTIAL" if any(missing_by_method.values()) else "COMPLETE",
        "missing_runs_by_method": missing_by_method,
        "summaries": summaries,
        "paired": paired_payloads,
    }
    (OUT / "all_baselines_comparison_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
