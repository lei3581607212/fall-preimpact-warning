"""Audit the frozen family-disjoint v2 dataset before paper-track training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paper_cv import is_derived_source_alias, nested_masks, physical_source_family


REQUIRED_FIELDS = {
    "X",
    "group",
    "source_path",
    "source_dataset",
    "end_frame",
    "impact_frame",
    "fps",
    "phase",
    "is_fall_window",
    "time_to_impact_seconds",
    "window_size",
    "risk_025ms",
    "risk_050ms",
    "risk_100ms",
}
HORIZONS = ((0.25, "risk_025ms"), (0.50, "risk_050ms"), (1.00, "risk_100ms"))
MANIFEST_FIELDS = {
    "group",
    "physical_source_family",
    "source_dataset",
    "outer_fold",
    "event_type",
    "windows",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_ids(data) -> np.ndarray:
    return np.asarray(
        [f"{dataset}::{path}" for dataset, path in zip(data["source_dataset"], data["source_path"])],
        dtype=str,
    )


def count_units(values: np.ndarray, mask: np.ndarray) -> int:
    return int(len(np.unique(values[mask])))


def add_check(checks: list, name: str, passed: bool, **details) -> None:
    item = {"name": name, "passed": bool(passed)}
    item.update(details)
    checks.append(item)


def add_warning(warnings: list, name: str, present: bool, **details) -> None:
    item = {"name": name, "present": bool(present)}
    item.update(details)
    warnings.append(item)


def split_row(data, mask, fold, split, dataset, groups, events, is_fall):
    selected = mask if dataset == "ALL" else mask & (data["source_dataset"].astype(str) == dataset)
    selected_groups = np.unique(groups[selected])
    selected_events = np.unique(events[selected])
    group_fall = {group: int(is_fall[np.flatnonzero(groups == group)[0]]) for group in selected_groups}
    event_fall = {event: int(is_fall[np.flatnonzero(events == event)[0]]) for event in selected_events}
    row = {
        "outer_fold": fold,
        "split": split,
        "source_dataset": dataset,
        "windows": int(selected.sum()),
        "groups": len(selected_groups),
        "fall_groups": sum(group_fall.values()),
        "normal_groups": len(group_fall) - sum(group_fall.values()),
        "source_events": len(selected_events),
        "fall_events": sum(event_fall.values()),
        "normal_events": len(event_fall) - sum(event_fall.values()),
    }
    for _, key in HORIZONS:
        row[f"positive_{key}"] = int(np.asarray(data[key])[selected].sum())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/paper_track/paper_family_disjoint_v2_manifest.csv"),
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=Path("outputs/paper_track/paper_family_disjoint_v2_audit.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_track/v2_preflight"))
    args = parser.parse_args()

    checks = []
    warnings = []
    with np.load(args.data, mmap_mode="r") as data:
        fields = set(data.files)
        missing = sorted(REQUIRED_FIELDS - fields)
        add_check(checks, "required_npz_fields", not missing, missing=missing)
        if missing:
            raise RuntimeError(f"NPZ is missing required fields: {missing}")

        n = int(len(data["X"]))
        per_window = sorted(key for key in REQUIRED_FIELDS - {"window_size"} if len(data[key]) != n)
        add_check(checks, "per_window_lengths", not per_window, windows=n, mismatched_fields=per_window)
        add_check(
            checks,
            "input_shape",
            data["X"].ndim == 3 and data["X"].shape[1:] == (64, 68) and int(data["window_size"]) == 64,
            shape=list(data["X"].shape),
            window_size=int(data["window_size"]),
        )
        add_check(checks, "input_dtype", data["X"].dtype == np.float32, dtype=str(data["X"].dtype))
        nonfinite_x = int(np.size(data["X"]) - np.isfinite(data["X"]).sum())
        add_check(checks, "input_finite", nonfinite_x == 0, nonfinite_values=nonfinite_x)

        groups = data["group"].astype(str)
        events = event_ids(data)
        datasets = data["source_dataset"].astype(str)
        is_fall = data["is_fall_window"].astype(np.uint8)
        fall = is_fall == 1
        normal = ~fall
        end_frame = data["end_frame"].astype(np.int64)
        impact_frame = data["impact_frame"].astype(np.int64)
        fps = data["fps"].astype(np.float64)
        remaining = data["time_to_impact_seconds"].astype(np.float64)

        add_check(checks, "binary_event_type", set(np.unique(is_fall)).issubset({0, 1}), values=np.unique(is_fall).tolist())
        invalid_fps = ~np.isfinite(fps) | (fps <= 0)
        bad_fall_fps = int((invalid_fps & fall).sum())
        bad_normal_fps = int((invalid_fps & normal).sum())
        add_check(checks, "fall_positive_finite_fps", bad_fall_fps == 0, invalid_windows=bad_fall_fps)
        add_warning(
            warnings,
            "normal_fps_missing_for_false_alerts_per_minute",
            bad_normal_fps > 0,
            invalid_windows=bad_normal_fps,
            affected_source_events=count_units(events, invalid_fps & normal),
            impact="Event false-alert rate remains valid; false alerts per minute cannot be reported for affected events.",
        )
        bad_normal = int(((impact_frame[normal] != -1) | ~np.isnan(remaining[normal])).sum())
        add_check(checks, "normal_sentinels", bad_normal == 0, invalid_windows=bad_normal)
        bad_fall = int(((impact_frame[fall] < 0) | ~np.isfinite(remaining[fall]) | (remaining[fall] <= 0)).sum())
        add_check(checks, "fall_timing_valid", bad_fall == 0, invalid_windows=bad_fall)
        noncausal = int((end_frame[fall] >= impact_frame[fall]).sum())
        add_check(checks, "causal_window_end", noncausal == 0, invalid_windows=noncausal)
        expected_remaining = (impact_frame[fall] - end_frame[fall]) / fps[fall]
        timing_mismatch = int((~np.isclose(remaining[fall], expected_remaining, rtol=1e-5, atol=1e-6)).sum())
        max_timing_error = float(np.max(np.abs(remaining[fall] - expected_remaining))) if fall.any() else 0.0
        add_check(
            checks,
            "time_to_impact_formula",
            timing_mismatch == 0,
            mismatched_windows=timing_mismatch,
            max_absolute_error_seconds=max_timing_error,
        )

        label_counts = {}
        for horizon, key in HORIZONS:
            actual = data[key].astype(np.uint8)
            binary = set(np.unique(actual)).issubset({0, 1})
            expected = fall & (remaining > 0) & (remaining <= horizon)
            mismatch = int((actual.astype(bool) != expected).sum())
            add_check(checks, f"{key}_definition", binary and mismatch == 0, positives=int(actual.sum()), mismatched_windows=mismatch)
            label_counts[key] = int(actual.sum())
        nested_025_050 = int((data["risk_025ms"] > data["risk_050ms"]).sum())
        nested_050_100 = int((data["risk_050ms"] > data["risk_100ms"]).sum())
        add_check(
            checks,
            "nested_risk_targets",
            nested_025_050 == 0 and nested_050_100 == 0,
            violations_025_over_050=nested_025_050,
            violations_050_over_100=nested_050_100,
        )
        group_mixed = []
        dataset_mixed = []
        event_mixed = []
        for group in np.unique(groups):
            idx = groups == group
            if len(np.unique(is_fall[idx])) != 1:
                group_mixed.append(group)
            if len(np.unique(datasets[idx])) != 1:
                dataset_mixed.append(group)
        for event in np.unique(events):
            if len(np.unique(is_fall[events == event])) != 1:
                event_mixed.append(event)
        add_check(checks, "group_event_consistency", not group_mixed, mixed_groups=group_mixed[:10])
        add_check(checks, "group_dataset_consistency", not dataset_mixed, mixed_groups=dataset_mixed[:10])
        add_check(checks, "source_event_consistency", not event_mixed, mixed_events=event_mixed[:10])
        derived_aliases = sorted(group for group in np.unique(groups) if is_derived_source_alias(group))
        derived_paths = sorted(path for path in np.unique(data["source_path"].astype(str)) if "MultiFallEvents/MulticamTrain_chute" in path)
        add_check(
            checks,
            "derived_multicam_aliases_excluded",
            not derived_aliases and not derived_paths,
            alias_groups=derived_aliases,
            alias_paths=derived_paths[:10],
        )

        with args.manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            manifest_columns = set(reader.fieldnames or [])
            rows = list(reader)
        missing_manifest = sorted(MANIFEST_FIELDS - manifest_columns)
        add_check(checks, "required_manifest_fields", not missing_manifest, missing=missing_manifest)
        if missing_manifest:
            raise RuntimeError(f"Manifest is missing required fields: {missing_manifest}")
        manifest_groups = [row["group"] for row in rows]
        duplicates = sorted(group for group, count in Counter(manifest_groups).items() if count > 1)
        add_check(checks, "manifest_group_unique", not duplicates, duplicates=duplicates[:10])
        missing_manifest_groups = sorted(set(groups) - set(manifest_groups))
        extra_manifest_groups = sorted(set(manifest_groups) - set(groups))
        add_check(
            checks,
            "manifest_exact_coverage",
            not missing_manifest_groups and not extra_manifest_groups,
            missing_groups=missing_manifest_groups[:10],
            extra_groups=extra_manifest_groups[:10],
        )
        manifest_by_group = {row["group"]: row for row in rows}
        metadata_errors = []
        for group in np.unique(groups):
            row = manifest_by_group.get(group)
            if row is None:
                continue
            idx = groups == group
            expected_family = physical_source_family(group)
            expected_type = "fall" if is_fall[np.flatnonzero(idx)[0]] else "normal"
            expected_dataset = datasets[np.flatnonzero(idx)[0]]
            if (
                row["physical_source_family"] != expected_family
                or row["event_type"] != expected_type
                or row["source_dataset"] != expected_dataset
                or int(row["windows"]) != int(idx.sum())
            ):
                metadata_errors.append(group)
        add_check(checks, "manifest_metadata_matches_npz", not metadata_errors, mismatched_groups=metadata_errors[:10])
        fold_by_group = {row["group"]: int(row["outer_fold"]) for row in rows}
        fold_ids = np.asarray([fold_by_group[group] for group in groups], dtype=np.int16)
        fold_values = sorted(np.unique(fold_ids).tolist())
        add_check(checks, "five_outer_folds", fold_values == [0, 1, 2, 3, 4], folds=fold_values)
        family_folds = {}
        for row in rows:
            family_folds.setdefault(row["physical_source_family"], set()).add(int(row["outer_fold"]))
        cross_fold = sorted(family for family, assigned in family_folds.items() if len(assigned) != 1)
        add_check(checks, "physical_family_single_fold", not cross_fold, cross_fold_families=cross_fold[:10])

        statistics = []
        split_intersections = []
        for outer_fold in range(5):
            train, development, test = nested_masks(fold_ids, outer_fold)
            masks = {"train": train, "development": development, "test": test}
            families = {
                name: {manifest_by_group[group]["physical_source_family"] for group in np.unique(groups[mask])}
                for name, mask in masks.items()
            }
            intersections = {
                "train_development": sorted(families["train"] & families["development"]),
                "train_test": sorted(families["train"] & families["test"]),
                "development_test": sorted(families["development"] & families["test"]),
            }
            split_intersections.append({"outer_fold": outer_fold, **intersections})
            for split, mask in masks.items():
                statistics.append(split_row(data, mask, outer_fold, split, "ALL", groups, events, is_fall))
                for dataset in sorted(np.unique(datasets[mask])):
                    statistics.append(split_row(data, mask, outer_fold, split, dataset, groups, events, is_fall))
        any_intersection = any(values for item in split_intersections for key, values in item.items() if key != "outer_fold")
        add_check(checks, "nested_split_family_disjoint", not any_intersection, intersections=split_intersections)

        recorded_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
        observed_hashes = {"output_data_sha256": sha256(args.data), "manifest_sha256": sha256(args.manifest)}
        hash_mismatch = {
            key: {"expected": recorded_audit.get(key), "observed": value}
            for key, value in observed_hashes.items()
            if recorded_audit.get(key) != value
        }
        add_check(checks, "frozen_artifact_hashes", not hash_mismatch, mismatches=hash_mismatch)
        add_check(
            checks,
            "frozen_protocol_metadata",
            recorded_audit.get("protocol") == "paper_family_disjoint_v2"
            and recorded_audit.get("weight_application") == "loss_only"
            and recorded_audit.get("known_cross_fold_families") == 0,
            protocol=recorded_audit.get("protocol"),
            weight_application=recorded_audit.get("weight_application"),
            known_cross_fold_families=recorded_audit.get("known_cross_fold_families"),
        )

        summary = {
            "windows": n,
            "groups": count_units(groups, np.ones(n, dtype=bool)),
            "source_events": count_units(events, np.ones(n, dtype=bool)),
            "fall_groups": count_units(groups, fall),
            "normal_groups": count_units(groups, normal),
            "fall_source_events": count_units(events, fall),
            "normal_source_events": count_units(events, normal),
            "risk_positive_windows": label_counts,
            "source_dataset_windows": {key: int(value) for key, value in sorted(Counter(datasets).items())},
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fold_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    passed = all(item["passed"] for item in checks)
    report = {
        "status": "PASS" if passed else "FAIL",
        "protocol": "paper_family_disjoint_v2",
        "weight_application": "loss_only",
        "data": str(args.data.resolve()),
        "manifest": str(args.manifest.resolve()),
        "hashes": observed_hashes,
        "summary": summary,
        "checks": checks,
        "warnings": warnings,
        "fold_statistics_csv": str(csv_path.resolve()),
    }
    json_path = args.output_dir / "label_event_audit.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "summary": summary,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "warnings": [item["name"] for item in warnings if item["present"]],
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
