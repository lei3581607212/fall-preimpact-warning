"""Reconstruct per-event false-alert actions for frozen v2 outer evaluations.

Only runs whose replayed event counts exactly match the authoritative heldout
JSON are released into the false-alert CSV. Mismatches are retained in the
audit JSON and never silently replace the published aggregate metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_v2_results import (  # noqa: E402
    FOLDS, MANIFEST, RUN_ROOT, SEEDS, DATA,
    load_annotation_index, make_model,
)
from scripts.evaluate_paper_event_policy import predict_mask
from utils.paper_cv import fold_ids_from_manifest, nested_masks


def same_stream_event_records(data, test_mask, scores, threshold, consecutive, annotation_index):
    """Use the frozen 0.5 s stream for false alerts and all lead thresholds."""
    event_ids = np.asarray([f"{origin}::{path}" for origin, path in zip(data["source_dataset"], data["source_path"])])
    rows_by_event = {}
    for index in np.flatnonzero(test_mask):
        rows_by_event.setdefault(event_ids[index], []).append({
            "event_id": event_ids[index], "source_path": str(data["source_path"][index]),
            "score": float(scores[index, 1]), "end_frame": int(data["end_frame"][index]),
            "impact_frame": int(data["impact_frame"][index]), "fps": float(data["fps"][index]),
        })
    falls, normals, false_alert_rows = [], [], []
    from scripts.aggregate_paper_v2_results import first_alert  # local import avoids changing public helpers
    for event_id, rows in rows_by_event.items():
        rows.sort(key=lambda item: item["end_frame"])
        alert = first_alert(rows, threshold, consecutive)
        impact = rows[0]["impact_frame"]
        if impact < 0:
            alerted = alert is not None
            normals.append({"event_id": event_id, "alerts": [alerted]})
            if alerted:
                from scripts.aggregate_paper_v2_results import lookup_action, action_category
                action, source = lookup_action(annotation_index, rows[0]["source_path"], rows[0]["end_frame"])
                false_alert_rows.append({
                    "event_id": event_id, "source_path": rows[0]["source_path"],
                    "action": action, "action_source": source, "category": action_category(action),
                })
        else:
            lead = None if alert is None else (impact - alert["end_frame"]) / max(alert["fps"], 1.0)
            falls.append({"event_id": event_id, "lead_seconds": lead})
    return falls, normals, false_alert_rows


def same_stream_point(falls, normals):
    leads = [item["lead_seconds"] for item in falls]
    return {
        "fall_events": len(falls),
        "normal_events": len(normals),
        "false_alert_events": sum(item["alerts"][0] for item in normals),
        "recall_at_0.25s": sum(value is not None and value >= 0.25 for value in leads) / max(len(leads), 1),
        "recall_at_0.50s": sum(value is not None and value >= 0.50 for value in leads) / max(len(leads), 1),
        "recall_at_1.00s": sum(value is not None and value >= 1.00 for value in leads) / max(len(leads), 1),
    }
from utils.paper_cv import fold_ids_from_manifest, nested_masks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Inference batch size; eval-mode outputs are batch-size invariant.")
    parser.add_argument("--output-csv", type=Path,
                        default=Path("outputs/paper_track/v2_summary/false_alert_events_reconstructed.csv"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("outputs/paper_track/v2_summary/false_alert_reconstruction_audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    data = np.load(DATA, mmap_mode="r")
    fold_ids = fold_ids_from_manifest(data["group"].astype(str), MANIFEST)
    annotations = load_annotation_index()
    seeds = [args.only_seed] if args.only_seed is not None else list(SEEDS)
    folds = [args.only_fold] if args.only_fold is not None else list(FOLDS)
    audit = {"protocol": "paper_family_disjoint_v2", "device": str(device), "runs": [], "matched_run_count": 0, "mismatch_run_count": 0}
    false_rows = []
    action_counts = Counter()
    for seed in seeds:
        for fold in folds:
            run_dir = RUN_ROOT / f"seed{seed}" / f"outer{fold}"
            heldout_path = run_dir / "heldout_event_evaluation.json"
            policy_path = run_dir / "selected_policy.json"
            checkpoint_path = ROOT / "models" / f"paper_family_disjoint_v2_outer{fold}_seed{seed}_hardneg.pth"
            if not heldout_path.exists() or not policy_path.exists():
                continue
            heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model, architecture = make_model(checkpoint, device)
            model.load_state_dict(checkpoint["state_dict"])
            _, _, test = nested_masks(fold_ids, fold)
            scores, _ = predict_mask(
                data, test, model, device, args.batch_size,
                checkpoint.get("feature_profile", "paper_raw_v1"),
                int(checkpoint["input_dim"]), architecture == "paper_timebin_action_teacher_v1",
            )
            falls, normals, rows = same_stream_event_records(
                data, test, scores, float(policy["threshold"]), int(policy["consecutive"]), annotations,
            )
            replay = same_stream_point(falls, normals)
            fields = ("fall_events", "normal_events", "false_alert_events", "recall_at_0.25s", "recall_at_0.50s", "recall_at_1.00s")
            mismatches = {
                field: {"authoritative": heldout.get(field), "replayed": replay.get(field)}
                for field in fields
                if int(heldout.get(field, 0)) != int(replay.get(field, 0)) if field in {"fall_events", "normal_events", "false_alert_events"}
            }
            # Recall values are floating point ratios; compare at event-count precision.
            for field, head in (("recall_at_0.25s", 0), ("recall_at_0.50s", 1), ("recall_at_1.00s", 2)):
                if abs(float(heldout.get(field, 0.0)) - float(replay.get(field, 0.0))) > 1e-8:
                    mismatches[field] = {"authoritative": heldout.get(field), "replayed": replay.get(field)}
            matched = not mismatches
            run_audit = {"seed": seed, "fold": fold, "checkpoint": str(checkpoint_path), "policy": policy,
                         "authoritative": {field: heldout.get(field) for field in fields}, "replayed": replay,
                         "matched": matched, "mismatches": mismatches}
            audit["runs"].append(run_audit)
            if matched:
                audit["matched_run_count"] += 1
                for row in rows:
                    row.update({"seed": seed, "fold": fold})
                    false_rows.append(row)
                    action_counts[row["action"]] += 1
            else:
                audit["mismatch_run_count"] += 1
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "fold", "event_id", "source_path", "action", "action_source", "category"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(false_rows)
    audit["false_alert_rows_released"] = len(false_rows)
    audit["action_counts_released"] = dict(action_counts)
    audit["publication_rule"] = "rows released only when replayed event counts and three recall ratios match authoritative heldout JSON"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("matched_run_count", "mismatch_run_count", "false_alert_rows_released", "action_counts_released")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
