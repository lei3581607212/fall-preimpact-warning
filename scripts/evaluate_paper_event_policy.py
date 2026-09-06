"""Select a multi-horizon teacher alert policy on development data only."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.model import (PaperMultiHorizonTeacher, PaperMultiHorizonActionTeacher, PureTCNBaseline,
                           PaperMultiHorizonRecoveryTeacher, PaperTimeBinActionTeacher,
                           TCNTEBaseline)
from scripts.train_paper_multihorizon_teacher import grouped_split
from utils.features import PAPER_RAW_PROFILE, prepare_features
from utils.paper_cv import fold_ids_from_manifest, nested_masks


def predict(model, x, device, batch_size, timebin=False):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
    values = []
    recovery_values = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs = model(batch.to(device))
            logits = outputs[0]
            if timebin:
                phase = torch.softmax(logits, dim=1)
                values.append(torch.stack((phase[:, 4], phase[:, 3:].sum(dim=1), phase[:, 2:].sum(dim=1)), dim=1).cpu().numpy())
            else:
                values.append(torch.sigmoid(logits).cpu().numpy())
            auxiliary = outputs[2] if len(outputs) > 2 else None
            if auxiliary is not None and auxiliary.ndim == 1:
                recovery_values.append(torch.sigmoid(auxiliary).cpu().numpy())
            else:
                recovery_values.append(np.zeros(len(batch), dtype=np.float32))
    return np.concatenate(values), np.concatenate(recovery_values)


def predict_mask(data, mask, model, device, batch_size, feature_profile, input_dim, timebin):
    """Predict only one split so development feasibility gates held-out access."""
    score_idx = np.flatnonzero(mask)
    all_scores = np.zeros((len(data["X"]), 3), dtype=np.float32)
    recovery_scores = np.zeros(len(data["X"]), dtype=np.float32)
    # Keep feature preparation amortized over large contiguous chunks. The
    # largest v2 split is below 12k windows, so this remains bounded while
    # avoiding repeated Python/NumPy setup for each 2k block.
    chunk_size = 10000
    for start in range(0, len(score_idx), chunk_size):
        idx = score_idx[start:start + chunk_size]
        raw_chunk = data["X"][idx].astype(np.float32, copy=False)
        x_chunk = raw_chunk if raw_chunk.shape[-1] == input_dim else prepare_features(raw_chunk, feature_profile)
        if x_chunk.shape[-1] != input_dim:
            raise RuntimeError(f"Checkpoint expects input_dim={input_dim}, profile produced {x_chunk.shape[-1]}")
        s_chunk, r_chunk = predict(model, x_chunk, device, batch_size, timebin=timebin)
        all_scores[idx] = s_chunk
        recovery_scores[idx] = r_chunk
    return all_scores, recovery_scores


def first_alert(rows, threshold, consecutive):
    evidence = 0
    for row in rows:
        evidence = evidence + 1 if row["score"] >= threshold else 0
        if evidence >= consecutive:
            return row
    return None


def load_normal_durations(path):
    """Load evaluated normal-event exposure without modifying the frozen NPZ."""
    durations = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_id = row["event_id"]
            duration = float(row["evaluated_duration_seconds"])
            if event_id in durations:
                raise RuntimeError(f"Duplicate normal event in duration sidecar: {event_id}")
            if not np.isfinite(duration) or duration <= 0:
                raise RuntimeError(f"Invalid evaluated duration for normal event: {event_id}")
            durations[event_id] = duration
    return durations


def evaluate_events(data, score, mask, threshold, consecutive, normal_duration_by_event=None):
    event_ids = np.asarray([f"{origin}::{path}" for origin, path in zip(data["source_dataset"], data["source_path"])])
    rows_by_event = {}
    for index in np.flatnonzero(mask):
        rows_by_event.setdefault(event_ids[index], []).append({
            "score": float(score[index]), "end_frame": int(data["end_frame"][index]),
            "impact_frame": int(data["impact_frame"][index]), "fps": float(data["fps"][index]),
        })
    normal_total = normal_alerts = 0
    normal_duration_minutes = 0.0
    normal_events_with_valid_duration = 0
    falls = []
    for event_id, rows in rows_by_event.items():
        rows.sort(key=lambda item: item["end_frame"])
        alert = first_alert(rows, threshold, consecutive)
        impact = rows[0]["impact_frame"]
        if impact < 0:
            normal_total += 1
            normal_alerts += int(alert is not None)
            if normal_duration_by_event is not None and event_id in normal_duration_by_event:
                normal_duration_minutes += normal_duration_by_event[event_id] / 60.0
                normal_events_with_valid_duration += 1
            else:
                event_fps = {row["fps"] for row in rows if np.isfinite(row["fps"]) and row["fps"] > 0}
                if len(event_fps) != 1:
                    continue
                fps = event_fps.pop()
                duration_frames = max(row["end_frame"] for row in rows) - min(row["end_frame"] for row in rows) + 1
                normal_duration_minutes += duration_frames / fps / 60.0
                normal_events_with_valid_duration += 1
            continue
        lead = None if alert is None else (impact - alert["end_frame"]) / max(alert["fps"], 1.0)
        falls.append({"event_id": event_id, "alerted": alert is not None, "lead_seconds": lead})
    leads = [item["lead_seconds"] for item in falls if item["lead_seconds"] is not None]
    result = {
        "fall_events": len(falls), "normal_events": normal_total, "false_alert_events": normal_alerts,
        "false_alert_rate": normal_alerts / max(normal_total, 1),
        "false_alert_events_per_minute": (
            normal_alerts / normal_duration_minutes
            if normal_events_with_valid_duration == normal_total and normal_duration_minutes > 0
            else None
        ),
        "normal_duration_minutes": normal_duration_minutes if normal_events_with_valid_duration else None,
        "normal_events_with_valid_duration": normal_events_with_valid_duration,
        "normal_duration_basis": (
            "sidecar_evaluated_windows" if normal_duration_by_event is not None
            else "npz_fps_span_fallback"
        ),
        "alerted_fall_events": sum(item["alerted"] for item in falls),
        "mean_lead_seconds": float(np.mean(leads)) if leads else None,
        "median_lead_seconds": float(np.median(leads)) if leads else None,
    }
    for lead in (0.25, 0.5, 1.0):
        result[f"recall_at_{lead:.2f}s"] = (sum(item["lead_seconds"] is not None and item["lead_seconds"] >= lead
                                                for item in falls) / max(len(falls), 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prediction_sequences/paper_multihorizon_pose_64frame.npz"))
    parser.add_argument("--model", type=Path, default=Path("models/paper_multihorizon_teacher_v1.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_track/event_policy_v1"))
    parser.add_argument("--max-false-alert-rate", type=float, default=1 / 12)
    parser.add_argument("--primary-lead-seconds", type=float, default=0.5, choices=(0.25, 0.5, 1.0))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8,
                        help="PyTorch CPU worker threads used for inference.")
    parser.add_argument("--skip-heldout", action="store_true",
                        help="Select and save a development policy without reading the held-out partition.")
    parser.add_argument("--fold-manifest", type=Path,
                        help="Frozen group-disjoint outer-fold manifest for nested cross-validation.")
    parser.add_argument("--outer-fold", type=int,
                        help="Outer test fold to use with --fold-manifest.")
    parser.add_argument("--normal-duration-manifest", type=Path,
                        help="Normal-event FPS/duration sidecar used only for false-alert events per minute.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    archive = np.load(args.data, mmap_mode="r")
    # Compressed NPZ members cannot be memory-mapped. Materialize each member
    # once so chunked development/test inference does not repeatedly decompress
    # the full feature tensor.
    data = {key: archive[key] for key in archive.files}
    normal_durations = load_normal_durations(args.normal_duration_manifest) if args.normal_duration_manifest else None
    if normal_durations is not None:
        normal_mask = data["is_fall_window"].astype(np.uint8) == 0
        required_normal_events = {
            f"{dataset}::{path}"
            for dataset, path in zip(data["source_dataset"][normal_mask], data["source_path"][normal_mask])
        }
        missing_durations = sorted(required_normal_events - set(normal_durations))
        if missing_durations:
            raise RuntimeError(
                f"Duration sidecar does not cover {len(missing_durations)} normal events; first={missing_durations[0]}"
            )
    if args.fold_manifest:
        if args.outer_fold is None:
            parser.error("--outer-fold is required with --fold-manifest")
        fold_ids = fold_ids_from_manifest(data["group"].astype(str), args.fold_manifest)
        train, development, test = nested_masks(fold_ids, args.outer_fold)
    else:
        train, development, test = grouped_split(data["group"].astype(str), data["is_fall_window"].astype(np.uint8), 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    architecture = checkpoint.get("architecture")
    window_size = int(checkpoint.get("window_size", data["window_size"]))
    feature_profile = checkpoint.get("feature_profile", PAPER_RAW_PROFILE)
    input_dim = int(checkpoint.get("input_dim", 68))
    model = (TCNTEBaseline(input_dim=input_dim, window_size=window_size)
             if architecture in {"tcnte", "tcnte_baseline_v1"}
             else PureTCNBaseline(input_dim=input_dim, window_size=window_size) if architecture == "pure_tcn_baseline_v1"
             else PaperTimeBinActionTeacher(input_dim=input_dim, window_size=window_size) if architecture == "paper_timebin_action_teacher_v1"
             else PaperMultiHorizonActionTeacher(input_dim=input_dim, window_size=window_size) if architecture == "paper_multihorizon_action_teacher_v1"
             else PaperMultiHorizonRecoveryTeacher(input_dim=input_dim, window_size=window_size) if architecture == "paper_multihorizon_recovery_teacher_v1"
             else PaperMultiHorizonTeacher(input_dim=input_dim, window_size=window_size)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    timebin = architecture == "paper_timebin_action_teacher_v1"
    # Policy selection sees development only. The held-out fold is not even
    # inferred until a feasible development policy has been frozen.
    all_scores, recovery_scores = predict_mask(
        data, development, model, device, args.batch_size, feature_profile, input_dim, timebin
    )
    horizon_index = {0.25: 0, 0.5: 1, 1.0: 2}[args.primary_lead_seconds]
    scores = all_scores[:, horizon_index]
    primary_key = f"recall_at_{args.primary_lead_seconds:.2f}s"
    candidates = []
    recovery_thresholds = (0.0, 0.25, 0.5, 0.75) if architecture == "paper_multihorizon_recovery_teacher_v1" else (0.0,)
    for recovery_threshold in recovery_thresholds:
        filtered_scores = np.where(recovery_scores[:, None] >= recovery_threshold, all_scores, all_scores)
        # A recovery threshold of 0.0 disables filtering. Otherwise suppress
        # windows judged likely to recover before event-level trigger logic.
        if recovery_threshold > 0:
            filtered_scores = np.where(recovery_scores[:, None] >= recovery_threshold, 0.0, all_scores)
        scores = filtered_scores[:, horizon_index]
        for threshold in np.arange(0.10, 0.91, 0.05):
            for consecutive in range(1, 5):
                result = evaluate_events(data, scores, development, float(threshold), consecutive, normal_durations)
                result.update({"threshold": round(float(threshold), 2), "consecutive": consecutive,
                               "recovery_threshold": recovery_threshold})
                candidates.append(result)
    eligible = [item for item in candidates if item["false_alert_rate"] <= args.max_false_alert_rate]
    if not eligible:
        # Keep the diagnostic scan when the strict event-level constraint is
        # infeasible on a person-disjoint development split.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "development_policy_scan.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
            writer.writeheader(); writer.writerows(candidates)
        best_far = min(candidates, key=lambda item: (item["false_alert_rate"],
                                                       -item[primary_key], item["consecutive"]))
        (args.output_dir / "infeasible_policy_summary.json").write_text(
            json.dumps({
                "constraint": args.max_false_alert_rate,
                "normal_duration_manifest": (
                    str(args.normal_duration_manifest.resolve()) if args.normal_duration_manifest else None
                ),
                "best_min_false_alert_policy": best_far,
            }, indent=2),
            encoding="utf-8")
        print("No development policy meets the false-alert constraint.")
        print(json.dumps({"constraint": args.max_false_alert_rate,
                          "best_min_false_alert_policy": best_far}, indent=2))
        return
    selected = dict(max(eligible, key=lambda item: (item[primary_key], item["recall_at_1.00s"],
                                                     item["recall_at_0.25s"], item["mean_lead_seconds"] or -1.0)))
    selected["normal_duration_manifest"] = (
        str(args.normal_duration_manifest.resolve()) if args.normal_duration_manifest else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "development_policy_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader(); writer.writerows(candidates)
    (args.output_dir / "selected_policy.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    if args.skip_heldout:
        print(f"Selected development policy: threshold={selected['threshold']}, consecutive={selected['consecutive']}")
        print("Held-out partition was intentionally not inspected.")
        return
    test_scores_all, test_recovery_scores = predict_mask(
        data, test, model, device, args.batch_size, feature_profile, input_dim, timebin
    )
    # Apply the selected trigger policy independently to each risk head. The
    # primary head chooses the policy; the reported horizon recalls must use
    # their corresponding heads, otherwise all three metrics describe one
    # score stream and are not comparable.
    test_scores = test_scores_all[:, horizon_index]
    if selected.get("recovery_threshold", 0.0) > 0:
        test_scores = np.where(test_recovery_scores >= selected["recovery_threshold"], 0.0, test_scores)
    test_result = evaluate_events(
        data, test_scores, test, selected["threshold"], selected["consecutive"], normal_durations
    )
    for horizon, index in ((0.25, 0), (0.5, 1), (1.0, 2)):
        head_result = evaluate_events(data, test_scores_all[:, index], test,
                                      selected["threshold"], selected["consecutive"], normal_durations)
        test_result[f"head_{horizon:.2f}s_recall"] = head_result[f"recall_at_{horizon:.2f}s"]
        test_result[f"head_{horizon:.2f}s_false_alert_rate"] = head_result["false_alert_rate"]
    test_result.update({"threshold": selected["threshold"], "consecutive": selected["consecutive"]})
    test_result["normal_duration_manifest"] = (
        str(args.normal_duration_manifest.resolve()) if args.normal_duration_manifest else None
    )
    (args.output_dir / "heldout_event_evaluation.json").write_text(json.dumps(test_result, indent=2), encoding="utf-8")
    print(f"Selected development policy: threshold={selected['threshold']}, consecutive={selected['consecutive']}")
    print(json.dumps(test_result, indent=2))


if __name__ == "__main__":
    main()
