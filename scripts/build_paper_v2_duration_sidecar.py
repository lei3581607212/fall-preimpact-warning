"""Build auditable normal-event FPS and duration metadata for paper v2."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from functools import reduce
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATA = Path("data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz")
DEFAULT_MANIFEST = Path("outputs/paper_track/paper_family_disjoint_v2_manifest.csv")
DEFAULT_VIDEO_ROOT = Path("data/raw_videos")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_track/v2_duration_metadata")
DEFAULT_ANNOTATIONS = (
    Path("data/annotations/paper_matched_common_actions_v1.csv"),
    Path("data/annotations/paper_matched_gmdcsa_fold1_error_actions.csv"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotation_fps(paths):
    values = defaultdict(list)
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                source_path = (row.get("source_path") or "").strip().replace("\\", "/")
                raw_fps = (row.get("fps") or "").strip()
                if not source_path or not raw_fps:
                    continue
                try:
                    fps = float(raw_fps)
                except ValueError:
                    continue
                if math.isfinite(fps) and fps > 0:
                    values[source_path].append((fps, str(path)))
    resolved = {}
    conflicts = {}
    for source_path, candidates in values.items():
        unique = sorted({round(item[0], 6) for item in candidates})
        if max(unique) - min(unique) > 0.01:
            conflicts[source_path] = unique
            continue
        resolved[source_path] = candidates[0]
    return resolved, conflicts


def inspect_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        opened = capture.isOpened()
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) if opened else 0
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) if opened else 0
    finally:
        capture.release()
    valid = opened and math.isfinite(fps) and fps > 0 and frame_count > 0
    return {
        "opened": opened,
        "valid": valid,
        "fps": fps if valid else None,
        "frame_count": frame_count if valid else None,
        "width": width if valid else None,
        "height": height if valid else None,
    }


def cadence_frames(end_frames: np.ndarray) -> int:
    differences = np.diff(np.unique(np.sort(end_frames.astype(np.int64))))
    positive = [int(value) for value in differences if value > 0]
    return reduce(math.gcd, positive) if positive else 1


def optional_number(value, digits=6):
    return "" if value is None else round(float(value), digits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--annotation-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    annotations = args.annotation_csv or list(DEFAULT_ANNOTATIONS)
    annotation_fps, annotation_conflicts = load_annotation_fps(annotations)

    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_by_group = {row["group"]: row for row in manifest_rows}

    rows = []
    with np.load(args.data, mmap_mode="r") as data:
        normal = data["is_fall_window"].astype(np.uint8) == 0
        source_paths = data["source_path"].astype(str)
        source_datasets = data["source_dataset"].astype(str)
        groups = data["group"].astype(str)
        npz_fps = data["fps"].astype(np.float64)
        end_frames = data["end_frame"].astype(np.int64)
        for source_path in sorted(np.unique(source_paths[normal])):
            mask = normal & (source_paths == source_path)
            event_groups = sorted(np.unique(groups[mask]).tolist())
            event_datasets = sorted(np.unique(source_datasets[mask]).tolist())
            if len(event_groups) != 1 or len(event_datasets) != 1:
                raise RuntimeError(f"Normal event metadata is inconsistent: {source_path}")
            group = event_groups[0]
            dataset = event_datasets[0]
            if group not in manifest_by_group:
                raise RuntimeError(f"Manifest does not cover normal group: {group}")
            manifest_row = manifest_by_group[group]
            event_ends = end_frames[mask]
            cadence = cadence_frames(event_ends)
            stored_fps_values = sorted({round(float(value), 6) for value in npz_fps[mask] if value > 0})
            if len(stored_fps_values) > 1:
                raise RuntimeError(f"NPZ has conflicting FPS values for {source_path}: {stored_fps_values}")
            stored_fps = stored_fps_values[0] if stored_fps_values else None

            relative_path = Path(*source_path.split("/"))
            video_path = args.video_root / relative_path
            video_exists = video_path.is_file()
            video = inspect_video(video_path) if video_exists else {
                "opened": False, "valid": False, "fps": None, "frame_count": None, "width": None, "height": None
            }
            annotation = annotation_fps.get(source_path)
            if video["valid"]:
                resolved_fps = video["fps"]
                fps_source = "video_container_opencv"
                fps_evidence = str(video_path)
                status = "VERIFIED_VIDEO"
            elif stored_fps is not None:
                resolved_fps = stored_fps
                fps_source = "frozen_npz"
                fps_evidence = str(args.data)
                status = "NPZ_FPS_ONLY"
            elif annotation is not None:
                resolved_fps = annotation[0]
                fps_source = "review_annotation"
                fps_evidence = str(Path(annotation[1]))
                status = "ANNOTATION_FPS_ONLY"
            else:
                resolved_fps = None
                fps_source = "unresolved"
                fps_evidence = ""
                status = "UNRESOLVED"

            if stored_fps is not None and resolved_fps is not None and abs(stored_fps - resolved_fps) > 0.01:
                raise RuntimeError(f"Frozen NPZ FPS conflicts with recovered FPS for {source_path}")
            raw_duration = (
                video["frame_count"] / resolved_fps
                if video["valid"] and resolved_fps is not None
                else None
            )
            evaluated_duration = (
                len(event_ends) * cadence / resolved_fps if resolved_fps is not None else None
            )
            evaluated_span = (
                (int(event_ends.max()) - int(event_ends.min()) + cadence) / resolved_fps
                if resolved_fps is not None else None
            )
            end_within_video = (
                int(event_ends.max()) < video["frame_count"] if video["valid"] else None
            )
            rows.append({
                "event_id": f"{dataset}::{source_path}",
                "source_dataset": dataset,
                "source_path": source_path,
                "group": group,
                "physical_source_family": manifest_row["physical_source_family"],
                "outer_fold": int(manifest_row["outer_fold"]),
                "window_count": int(mask.sum()),
                "first_end_frame": int(event_ends.min()),
                "last_end_frame": int(event_ends.max()),
                "evaluation_cadence_frames": cadence,
                "npz_fps": optional_number(stored_fps),
                "resolved_fps": optional_number(resolved_fps),
                "fps_source": fps_source,
                "fps_evidence": fps_evidence,
                "video_exists": video_exists,
                "video_opened": video["opened"],
                "raw_video_frame_count": "" if video["frame_count"] is None else video["frame_count"],
                "raw_video_width": "" if video["width"] is None else video["width"],
                "raw_video_height": "" if video["height"] is None else video["height"],
                "raw_video_duration_seconds": optional_number(raw_duration),
                "evaluated_duration_seconds": optional_number(evaluated_duration),
                "evaluated_span_seconds": optional_number(evaluated_span),
                "last_end_within_video": "" if end_within_video is None else end_within_video,
                "status": status,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = args.output_dir / "normal_event_duration_sidecar.csv"
    with sidecar_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unresolved = [row for row in rows if row["status"] == "UNRESOLVED"]
    invalid_duration = [row for row in rows if row["evaluated_duration_seconds"] == ""]
    out_of_range = [row for row in rows if row["last_end_within_video"] is False]
    dataset_summary = {}
    for dataset in sorted({row["source_dataset"] for row in rows}):
        selected = [row for row in rows if row["source_dataset"] == dataset]
        dataset_summary[dataset] = {
            "events": len(selected),
            "raw_videos_verified": sum(row["status"] == "VERIFIED_VIDEO" for row in selected),
            "evaluated_duration_minutes": sum(float(row["evaluated_duration_seconds"]) for row in selected) / 60.0,
            "raw_video_duration_minutes": sum(float(row["raw_video_duration_seconds"] or 0) for row in selected) / 60.0,
        }
    status_counts = Counter(row["status"] for row in rows)
    audit = {
        "status": "PASS" if not unresolved and not invalid_duration and not out_of_range and not annotation_conflicts else "FAIL",
        "protocol": "paper_family_disjoint_v2",
        "duration_definition": (
            "evaluated_duration_seconds = window_count * gcd(positive end-frame differences) / resolved_fps; "
            "this counts only scored prediction opportunities and excludes gaps absent from the frozen NPZ"
        ),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "video_root": str(args.video_root.resolve()),
        "annotation_sources": [str(path.resolve()) for path in annotations],
        "events": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "resolved_fps_events": len(rows) - len(unresolved),
        "verified_raw_video_events": sum(row["status"] == "VERIFIED_VIDEO" for row in rows),
        "evaluated_duration_minutes": sum(float(row["evaluated_duration_seconds"] or 0) for row in rows) / 60.0,
        "raw_video_duration_minutes_verified_only": sum(float(row["raw_video_duration_seconds"] or 0) for row in rows) / 60.0,
        "dataset_summary": dataset_summary,
        "unresolved_events": [row["event_id"] for row in unresolved],
        "invalid_duration_events": [row["event_id"] for row in invalid_duration],
        "out_of_range_events": [row["event_id"] for row in out_of_range],
        "annotation_fps_conflicts": annotation_conflicts,
        "sidecar": str(sidecar_path.resolve()),
        "sidecar_sha256": sha256(sidecar_path),
    }
    audit_path = args.output_dir / "normal_event_duration_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
