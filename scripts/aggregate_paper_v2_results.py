"""Aggregate the completed family-disjoint v2 paper runs.

This script never trains or selects a policy. It reads the frozen per-run
checkpoint, the development-selected policy, and the held-out event output.
For complete runs it uses the authoritative held-out event counts to create
count-bootstrap intervals. INFEASIBLE and outer FAR-exceeded runs remain
visible in the seed-fold table. A future evaluator export can replace the
count bootstrap with exact per-event resampling without changing the tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_paper_event_policy import evaluate_events, first_alert, predict_mask
from scripts.model import (
    PaperMultiHorizonActionTeacher,
    PaperMultiHorizonRecoveryTeacher,
    PaperMultiHorizonTeacher,
    PaperTimeBinActionTeacher,
)
from utils.features import PAPER_RAW_PROFILE
from utils.paper_cv import fold_ids_from_manifest, nested_masks


SEEDS = [0, 7, 13, 21, 42, 123]
FOLDS = [0, 1, 2, 3, 4]
FAR_BUDGET = 1.0 / 12.0
DATA = ROOT / "data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz"
MANIFEST = ROOT / "outputs/paper_track/paper_family_disjoint_v2_manifest.csv"
RUN_ROOT = ROOT / "outputs/paper_track/paper_family_disjoint_v2_5x6"
SIDECAR = ROOT / "outputs/paper_track/v2_duration_metadata/normal_event_duration_sidecar.csv"
FIGURE_DIR = ROOT / "outputs/paper_track/v2_figure_data"
SUMMARY_DIR = ROOT / "outputs/paper_track/v2_summary"

ANNOTATION_FILES = [
    ROOT / "data/annotations/normal_action_segments.csv",
    ROOT / "data/annotations/external_normal_action_segments.csv",
    ROOT / "data/annotations/paper_matched_common_actions_v1.csv",
    ROOT / "data/annotations/paper_matched_gmdcsa_fold1_error_actions.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_annotation_index() -> dict:
    index = {}
    for path in ANNOTATION_FILES:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                source_path = row.get("source_path", "").strip()
                if not source_path:
                    continue
                entry = index.setdefault(source_path, {"segments": [], "fallback": None})
                action = row.get("action", "").strip() or "other_normal"
                start = row.get("start_frame", "").strip()
                end = row.get("end_frame", "").strip()
                recovery_end = row.get("recovery_end_frame", "").strip()
                if start and end:
                    entry["segments"].append((float(start), float(recovery_end or end), action))
                elif entry["fallback"] is None:
                    entry["fallback"] = action
    return index


CAUCA_MAP = {
    "kneel": "kneel",
    "pick up object": "bend_recover",
    "sit down": "sit_down",
    "walk": "walk",
    "lie down": "lie_down",
    "fall backwards": "fall",
    "fall forward": "fall",
    "fall left": "fall",
    "fall right": "fall",
}


def derive_action(source_path: str) -> str | None:
    parts = [part.strip().lower() for part in str(source_path).replace("\\", "/").split("/")]
    if parts and parts[0] == "caucafall" and len(parts) >= 3:
        return CAUCA_MAP.get(parts[-2])
    if "adl" in parts:
        return "adl_generic"
    if "normal" in parts:
        return "other_normal"
    return None


def lookup_action(index: dict, source_path: str, end_frame: int) -> tuple[str, str]:
    entry = index.get(source_path)
    if entry:
        candidates = [item for item in entry["segments"] if item[0] <= end_frame <= item[1]]
        if candidates:
            best = min(candidates, key=lambda item: item[1] - item[0])
            return best[2], "annotation"
        if entry["fallback"]:
            return entry["fallback"], "annotation"
    derived = derive_action(source_path)
    if derived:
        return derived, "path"
    return "unannotated", "none"


def action_category(action: str) -> str:
    if action in {"lie_down", "sit_down", "unstable_recover", "turn_recover"}:
        return "fall_like"
    if action in {"bend_recover", "squat_recover", "change_direction", "other_normal", "kneel", "walk", "adl_generic"}:
        return "clear_adl"
    return "unannotated_or_other"


def make_model(checkpoint: dict, device: torch.device):
    architecture = checkpoint.get("architecture")
    input_dim = int(checkpoint["input_dim"])
    window_size = int(checkpoint["window_size"])
    if architecture == "paper_timebin_action_teacher_v1":
        model = PaperTimeBinActionTeacher(input_dim=input_dim, window_size=window_size)
    elif architecture == "paper_multihorizon_action_teacher_v1":
        model = PaperMultiHorizonActionTeacher(input_dim=input_dim, window_size=window_size)
    elif architecture == "paper_multihorizon_recovery_teacher_v1":
        model = PaperMultiHorizonRecoveryTeacher(input_dim=input_dim, window_size=window_size)
    else:
        model = PaperMultiHorizonTeacher(input_dim=input_dim, window_size=window_size)
    return model.to(device), architecture


def event_records(data, test_mask, scores, threshold: float, consecutive: int, annotation_index: dict):
    event_ids = np.asarray([
        f"{origin}::{path}"
        for origin, path in zip(data["source_dataset"], data["source_path"])
    ])
    rows_by_event = defaultdict(list)
    for index in np.flatnonzero(test_mask):
        rows_by_event[event_ids[index]].append({
            "event_id": event_ids[index],
            "source_path": str(data["source_path"][index]),
            "score": scores[index],
            "end_frame": int(data["end_frame"][index]),
            "impact_frame": int(data["impact_frame"][index]),
            "fps": float(data["fps"][index]),
        })

    falls = []
    normals = []
    false_alert_rows = []
    for event_id, rows in rows_by_event.items():
        rows.sort(key=lambda item: item["end_frame"])
        impact = rows[0]["impact_frame"]
        per_head_alerts = []
        per_head_leads = []
        for head in range(3):
            head_rows = [{**row, "score": float(row["score"][head])} for row in rows]
            alert = first_alert(head_rows, threshold, consecutive)
            per_head_alerts.append(alert is not None)
            per_head_leads.append(
                None if alert is None or impact < 0
                else (impact - alert["end_frame"]) / max(alert["fps"], 1.0)
            )
        if impact < 0:
            normals.append({"event_id": event_id, "alerts": per_head_alerts})
            if per_head_alerts[1]:
                action, source = lookup_action(annotation_index, rows[0]["source_path"], rows[0]["end_frame"])
                false_alert_rows.append({
                    "event_id": event_id,
                    "source_path": rows[0]["source_path"],
                    "action": action,
                    "action_source": source,
                    "category": action_category(action),
                })
        else:
            falls.append({"event_id": event_id, "lead_seconds": per_head_leads})
    return falls, normals, false_alert_rows


def bootstrap_events(falls: list[dict], normals: list[dict], reps: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    n_falls = len(falls)
    n_normals = len(normals)
    result = {}
    for head, horizon in enumerate((0.25, 0.50, 1.00)):
        values = np.asarray([
            np.nan if item["lead_seconds"][head] is None else item["lead_seconds"][head]
            for item in falls
        ], dtype=float)
        samples = []
        for _ in range(reps):
            index = rng.randint(0, n_falls, n_falls) if n_falls else []
            sampled = values[index] if n_falls else np.asarray([])
            samples.append(float(np.mean((~np.isnan(sampled)) & (sampled >= horizon))) if n_falls else 0.0)
        result[f"recall_at_{horizon:.2f}s"] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]

    far_samples = []
    for _ in range(reps):
        index = rng.randint(0, n_normals, n_normals) if n_normals else []
        alerts = np.asarray([item["alerts"][1] for item in normals], dtype=bool)
        far_samples.append(float(np.mean(alerts[index])) if n_normals else 0.0)
    result["false_alert_rate"] = [float(np.percentile(far_samples, 2.5)), float(np.percentile(far_samples, 97.5))]

    for head, horizon in enumerate((0.25, 0.50, 1.00)):
        valid = np.asarray([
            item["lead_seconds"][head]
            for item in falls
            if item["lead_seconds"][head] is not None
        ], dtype=float)
        samples = []
        for _ in range(reps):
            index = rng.randint(0, len(valid), len(valid)) if len(valid) else []
            samples.append(float(np.mean(valid[index])) if len(valid) else float("nan"))
        result[f"mean_lead_seconds_{horizon:.2f}s"] = [
            float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))
        ] if samples else [None, None]
        samples = []
        for _ in range(reps):
            index = rng.randint(0, len(valid), len(valid)) if len(valid) else []
            samples.append(float(np.median(valid[index])) if len(valid) else float("nan"))
        result[f"median_lead_seconds_{horizon:.2f}s"] = [
            float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))
        ] if samples else [None, None]
    return result


def point_from_records(falls: list[dict], normals: list[dict]) -> dict:
    out = {
        "fall_events": len(falls),
        "normal_events": len(normals),
        "false_alert_events": sum(item["alerts"][1] for item in normals),
    }
    out["false_alert_rate"] = out["false_alert_events"] / max(out["normal_events"], 1)
    for head, horizon in enumerate((0.25, 0.50, 1.00)):
        leads = [item["lead_seconds"][head] for item in falls]
        valid = [value for value in leads if value is not None]
        out[f"recall_at_{horizon:.2f}s"] = sum(value is not None and value >= horizon for value in leads) / max(len(leads), 1)
        out[f"head_{horizon:.2f}s_alerted_falls"] = sum(value is not None for value in leads)
        out[f"head_{horizon:.2f}s_mean_lead_seconds"] = float(np.mean(valid)) if valid else None
        out[f"head_{horizon:.2f}s_median_lead_seconds"] = float(np.median(valid)) if valid else None
    return out


def aggregate_macro(rows: list[dict], metric: str, weight_key: str) -> tuple[float | None, float | None, float | None]:
    if not rows:
        return None, None, None
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    weights = np.asarray([max(float(row.get(weight_key, 1)), 1.0) for row in rows], dtype=float)
    mean = float(np.average(values, weights=weights))
    lows = [row["bootstrap"][metric][0] for row in rows if metric in row["bootstrap"]]
    highs = [row["bootstrap"][metric][1] for row in rows if metric in row["bootstrap"]]
    return mean, (float(np.average(lows, weights=weights[:len(lows)])) if lows else None), (float(np.average(highs, weights=weights[:len(highs)])) if highs else None)


def bootstrap_from_saved_event_counts(heldout: dict, reps: int, seed: int) -> dict:
    """Bootstrap event indicators from the authoritative held-out counts.

    The v2 evaluator stores aggregate event counts, not per-event predictions.
    Replaying scores in this environment was found to disagree with the
    authoritative held-out JSON, so this conservative fallback bootstraps the
    reported event indicators as binary event counts. It is explicitly marked
    as count-bootstrap in the output metadata; lead-time intervals require a
    future per-event review export.
    """
    rng = np.random.RandomState(seed)
    n_falls = int(heldout.get("fall_events", 0) or 0)
    n_normals = int(heldout.get("normal_events", 0) or 0)
    result = {}
    for horizon in (0.25, 0.50, 1.00):
        point = heldout.get(f"recall_at_{horizon:.2f}s", 0.0)
        successes = int(round(float(point) * n_falls))
        labels = np.asarray([1] * successes + [0] * max(n_falls - successes, 0), dtype=float)
        samples = []
        for _ in range(reps):
            index = rng.randint(0, len(labels), len(labels)) if len(labels) else []
            samples.append(float(np.mean(labels[index])) if len(labels) else 0.0)
        result[f"recall_at_{horizon:.2f}s"] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
    far_point = float(heldout.get("false_alert_rate", 0.0) or 0.0)
    far_successes = int(round(far_point * n_normals))
    labels = np.asarray([1] * far_successes + [0] * max(n_normals - far_successes, 0), dtype=float)
    samples = []
    for _ in range(reps):
        index = rng.randint(0, len(labels), len(labels)) if len(labels) else []
        samples.append(float(np.mean(labels[index])) if len(labels) else 0.0)
    result["false_alert_rate"] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--only-fold", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cuda" if args.device == "cuda" else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    data = np.load(DATA, mmap_mode="r")
    fold_ids = fold_ids_from_manifest(data["group"].astype(str), MANIFEST)
    annotation_index = load_annotation_index()
    complete_rows = []
    status_rows = []
    false_alert_rows = []
    policy_rows = []
    all_run_rows = []

    print(f"device={device}; bootstrap_reps={args.bootstrap_reps}", flush=True)
    selected_seeds = [args.only_seed] if args.only_seed is not None else SEEDS
    selected_folds = [args.only_fold] if args.only_fold is not None else FOLDS
    for seed in selected_seeds:
        for fold in selected_folds:
            key = f"seed{seed}/outer{fold}"
            output_dir = RUN_ROOT / key
            heldout_path = output_dir / "heldout_event_evaluation.json"
            infeasible_path = output_dir / "infeasible_policy_summary.json"
            checkpoint_path = ROOT / "models" / f"paper_family_disjoint_v2_outer{fold}_seed{seed}_hardneg.pth"
            if infeasible_path.exists() and not heldout_path.exists():
                summary = json.loads(infeasible_path.read_text(encoding="utf-8"))
                best = summary.get("best_min_false_alert_policy", {})
                row = {"seed": seed, "fold": fold, "status": "infeasible", "far": None,
                       "fall_events": best.get("fall_events"), "normal_events": best.get("normal_events"),
                       "false_alert_events": best.get("false_alert_events"), "recall_at_0.50s": None,
                       "mean_lead_seconds": None, "median_lead_seconds": None,
                       "false_alert_events_per_minute": None, "checkpoint": str(checkpoint_path)}
                status_rows.append(row)
                all_run_rows.append(row)
                scan_path = output_dir / "development_policy_scan.csv"
                if scan_path.exists():
                    with scan_path.open(newline="", encoding="utf-8-sig") as handle:
                        for candidate in csv.DictReader(handle):
                            policy_rows.append({"method": "v2 teacher", "split": "development", "seed": seed, "fold": fold,
                                                "far": candidate.get("false_alert_rate"), "recall_050": candidate.get("recall_at_0.50s"),
                                                "threshold": candidate.get("threshold"), "consecutive": candidate.get("consecutive")})
                continue
            if not heldout_path.exists() or not (output_dir / "selected_policy.json").exists():
                raise RuntimeError(f"missing complete evaluation for {key}")
            heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
            policy = json.loads((output_dir / "selected_policy.json").read_text(encoding="utf-8"))
            bootstrap = bootstrap_from_saved_event_counts(heldout, args.bootstrap_reps, seed * 1000 + fold)
            status = "pass" if float(heldout["false_alert_rate"]) <= FAR_BUDGET else "far_exceeded"
            row = {
                "seed": seed, "fold": fold, "status": status,
                "far": heldout["false_alert_rate"], "fall_events": heldout["fall_events"], "normal_events": heldout["normal_events"],
                "false_alert_events": heldout["false_alert_events"],
                "recall_at_0.25s": heldout["recall_at_0.25s"],
                "recall_at_0.50s": heldout["recall_at_0.50s"],
                "recall_at_1.00s": heldout["recall_at_1.00s"],
                "mean_lead_seconds": heldout["mean_lead_seconds"], "median_lead_seconds": heldout["median_lead_seconds"],
                "false_alert_events_per_minute": heldout.get("false_alert_events_per_minute"),
                "threshold": policy["threshold"], "consecutive": policy["consecutive"],
                "checkpoint": str(checkpoint_path), "bootstrap": bootstrap,
            }
            complete_rows.append(row)
            status_rows.append({key: value for key, value in row.items() if key != "bootstrap"})
            all_run_rows.append(row)
            scan_path = output_dir / "development_policy_scan.csv"
            with scan_path.open(newline="", encoding="utf-8-sig") as handle:
                for candidate in csv.DictReader(handle):
                    policy_rows.append({"method": "v2 teacher", "split": "development", "seed": seed, "fold": fold,
                                        "far": candidate.get("false_alert_rate"), "recall_050": candidate.get("recall_at_0.50s"),
                                        "threshold": candidate.get("threshold"), "consecutive": candidate.get("consecutive")})
            print(f"{key}: {status}, FAR={float(heldout['false_alert_rate']):.3f}, falls={heldout['fall_events']}, normals={heldout['normal_events']}", flush=True)

    # Required seed-fold input for the feasibility matrix and T3.
    write_csv(FIGURE_DIR / "seed_fold_results.csv", status_rows,
              ["seed", "fold", "status", "far", "fall_events", "normal_events", "false_alert_events",
               "recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s", "mean_lead_seconds",
               "median_lead_seconds", "false_alert_events_per_minute", "threshold", "consecutive", "checkpoint"])
    write_csv(FIGURE_DIR / "policy_frontier.csv", policy_rows,
              ["method", "split", "seed", "fold", "far", "recall_050", "threshold", "consecutive"])
    write_csv(SUMMARY_DIR / "false_alert_events.csv", false_alert_rows,
              ["seed", "fold", "event_id", "source_path", "action", "action_source", "category"])

    # Main figure input: weighted macro across complete runs and the strict
    # outer-FAR-feasible subset. Per-run CIs remain in bootstrap_event_ci.json.
    main_rows = []
    for label, subset, eligible in (
        ("v2 all complete outer evaluations", complete_rows, True),
        ("v2 outer FAR-feasible evaluations", [row for row in complete_rows if row["status"] == "pass"], True),
    ):
        for head, horizon in enumerate((0.25, 0.50, 1.00)):
            metric = f"recall_at_{horizon:.2f}s"
            mean, low, high = aggregate_macro(subset, metric, "fall_events")
            main_rows.append({"method": label, "horizon_seconds": horizon, "recall": mean, "ci_low": low,
                              "ci_high": high, "fall_events": sum(int(row["fall_events"]) for row in subset),
                              "eligible": bool(eligible), "run_count": len(subset), "ci_method": "mean of event-count-bootstrap run intervals"})
    write_csv(FIGURE_DIR / "main_results.csv", main_rows,
              ["method", "horizon_seconds", "recall", "ci_low", "ci_high", "fall_events", "eligible", "run_count", "ci_method"])

    # No family-disjoint v2 ablation checkpoints were run; record that fact
    # instead of manufacturing a delta from the historical v1 ablations.
    write_csv(FIGURE_DIR / "ablation_results.csv",
              [{"variant": "no_family_disjoint_v2_ablation_checkpoint", "delta_recall_050": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "eligible": False, "status": "not_run"}],
              ["variant", "delta_recall_050", "ci_low", "ci_high", "eligible", "status"])

    status_counts_for_failures = Counter(row["status"] for row in status_rows)
    failure_rows = [
        {"category": "development_infeasible_fold", "count": status_counts_for_failures.get("infeasible", 0)},
        {"category": "outer_far_exceeded_fold", "count": status_counts_for_failures.get("far_exceeded", 0)},
        {"category": "outer_far_feasible_fold", "count": status_counts_for_failures.get("pass", 0)},
    ]
    write_csv(FIGURE_DIR / "failure_modes.csv", failure_rows, ["category", "count"])

    # Flatten Bootstrap results for spreadsheet/table use.
    bootstrap_json = {}
    bootstrap_rows = []
    for row in complete_rows:
        key = f"seed{row['seed']}/outer{row['fold']}"
        bootstrap_json[key] = {
            "status": row["status"], "point": {key2: row[key2] for key2 in (
                "far", "recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s", "mean_lead_seconds", "median_lead_seconds")},
            "bootstrap": row["bootstrap"], "fall_events": row["fall_events"], "normal_events": row["normal_events"],
        }
        for metric, interval in row["bootstrap"].items():
            bootstrap_rows.append({"seed": row["seed"], "fold": row["fold"], "status": row["status"],
                                   "metric": metric, "ci_low": interval[0], "ci_high": interval[1],
                                   "fall_events": row["fall_events"], "normal_events": row["normal_events"]})
    (SUMMARY_DIR / "bootstrap_event_ci.json").parent.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "bootstrap_event_ci.json").write_text(json.dumps(bootstrap_json, indent=2), encoding="utf-8")
    write_csv(SUMMARY_DIR / "bootstrap_event_ci.csv", bootstrap_rows,
              ["seed", "fold", "status", "metric", "ci_low", "ci_high", "fall_events", "normal_events"])

    status_counts = Counter(row["status"] for row in status_rows)
    summary = {
        "protocol_version": "paper_family_disjoint_v2",
        "seeds": SEEDS,
        "outer_folds": 5,
        "far_budget": FAR_BUDGET,
        "run_count": len(status_rows),
        "status_counts": dict(status_counts),
        "complete_run_count": len(complete_rows),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_unit": "reported event-count indicators within each seed-fold run",
        "recall_stream": "selected 0.50s alert stream; recall_at_* are same-stream lead thresholds",
        "data_path": str(DATA), "data_sha256": sha256(DATA),
        "manifest_path": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "normal_duration_sidecar": str(SIDECAR), "normal_duration_sidecar_sha256": sha256(SIDECAR),
        "ablation_status": "not_run_under_family_disjoint_v2",
        "false_alert_action_status": "not_reconstructed_from_saved_aggregate_json",
        "figure_data": ["main_results.csv", "ablation_results.csv", "policy_frontier.csv", "seed_fold_results.csv", "failure_modes.csv"],
    }
    (SUMMARY_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "metadata.json").write_text(json.dumps({
        **summary,
        "known_cross_fold_families": 0,
        "manifest_path": str(MANIFEST), "data_path": str(DATA),
    }, indent=2), encoding="utf-8")
    write_csv(SUMMARY_DIR / "per_run_metrics.csv", status_rows,
              ["seed", "fold", "status", "far", "fall_events", "normal_events", "false_alert_events",
               "recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s", "mean_lead_seconds",
               "median_lead_seconds", "false_alert_events_per_minute", "threshold", "consecutive", "checkpoint"])
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
