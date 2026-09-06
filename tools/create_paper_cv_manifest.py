"""Create the frozen group-disjoint outer-fold manifest for the paper track."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paper_cv import (is_derived_source_alias, physical_source_family,
                            stratified_family_folds, stratified_group_folds)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "prediction_sequences" / "paper_multihorizon_pose_64frame_matched.npz"
V1_MANIFEST = ROOT / "outputs" / "paper_track" / "paper_matched_5fold_manifest.csv"
V2_DATA = ROOT / "data" / "prediction_sequences" / "paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz"
V2_MANIFEST = ROOT / "outputs" / "paper_track" / "paper_family_disjoint_v2_manifest.csv"
V2_EXCLUSIONS = ROOT / "outputs" / "paper_track" / "paper_family_disjoint_v2_exclusions.csv"
V2_AUDIT = ROOT / "outputs" / "paper_track" / "paper_family_disjoint_v2_audit.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(groups, is_fall, fold_ids, source_dataset=None, include_family=False):
    rows = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        family = physical_source_family(group)
        row = {
            "group": group,
            "outer_fold": int(fold_ids[indices[0]]),
            "event_type": "fall" if int(is_fall[indices].max()) else "normal",
            "windows": int(len(indices)),
        }
        if include_family:
            row = {
                "group": group,
                "physical_source_family": family,
                "source_dataset": str(source_dataset[indices[0]]),
                **{key: value for key, value in row.items() if key != "group"},
            }
        rows.append(row)
    return rows


def fold_summary(rows, folds):
    summaries = []
    for fold in range(folds):
        selected = [row for row in rows if row["outer_fold"] == fold]
        dataset_counts = {}
        for row in selected:
            if "source_dataset" in row:
                dataset = row["source_dataset"]
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        summaries.append(
            {
                "outer_fold": fold,
                "groups": len(selected),
                "fall_groups": sum(row["event_type"] == "fall" for row in selected),
                "normal_groups": sum(row["event_type"] == "normal" for row in selected),
                "windows": sum(row["windows"] for row in selected),
                "source_dataset_groups": dataset_counts,
            }
        )
    return summaries


def create_family_disjoint_v2(args):
    if args.output_data.resolve() == args.data.resolve():
        raise ValueError("v2 output data must not overwrite the input NPZ")
    with np.load(args.data) as data:
        groups = data["group"].astype(str)
        derived_mask = np.asarray([is_derived_source_alias(group) for group in groups])
        derived_groups = sorted(set(groups[derived_mask]))
        if len(derived_groups) != args.expected_derived_aliases:
            raise RuntimeError(
                f"expected {args.expected_derived_aliases} derived Multicam aliases, "
                f"found {len(derived_groups)}"
            )

        all_groups = set(groups)
        exclusion_rows = []
        for group in derived_groups:
            family = physical_source_family(group)
            canonical = sorted(
                candidate for candidate in all_groups
                if candidate != group and physical_source_family(candidate) == family
            )
            if len(canonical) != 1:
                raise RuntimeError(f"expected one retained canonical group for {group}, found {canonical}")
            indices = np.flatnonzero(groups == group)
            exclusion_rows.append(
                {
                    "excluded_group": group,
                    "physical_source_family": family,
                    "retained_canonical_group": canonical[0],
                    "excluded_windows": int(len(indices)),
                    "excluded_source_paths": int(len(set(data["source_path"][indices].astype(str)))),
                    "reason": "derived event clip duplicates a retained full Multicam recording",
                }
            )

        keep = ~derived_mask
        payload = {
            key: (data[key] if data[key].ndim == 0 else data[key][keep])
            for key in data.files
        }
        retained_groups = payload["group"].astype(str)
        retained_families = np.asarray(
            [physical_source_family(group) for group in retained_groups]
        )
        if len(set(retained_groups)) != len(set(retained_families)):
            raise RuntimeError("retained v2 data still contains more than one group per physical family")
        is_fall = payload["is_fall_window"].astype(np.uint8)
        source_dataset = payload["source_dataset"].astype(str)
        fold_ids = stratified_family_folds(
            retained_families, is_fall, source_dataset, args.folds, args.seed
        )

        args.output_data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_data, **payload)

    rows = manifest_rows(
        retained_groups, is_fall, fold_ids, source_dataset=source_dataset,
        include_family=True,
    )
    family_folds = {}
    for row in rows:
        family_folds.setdefault(row["physical_source_family"], set()).add(row["outer_fold"])
    cross_fold = sorted(family for family, values in family_folds.items() if len(values) > 1)
    if cross_fold:
        raise RuntimeError(f"v2 manifest still has cross-fold physical families: {cross_fold[:3]}")
    summaries = fold_summary(rows, args.folds)
    if any(summary["fall_groups"] == 0 or summary["normal_groups"] == 0 for summary in summaries):
        raise RuntimeError("every v2 fold must contain both fall and normal groups")

    write_csv(args.output, rows, list(rows[0]))
    write_csv(args.exclusions, exclusion_rows, list(exclusion_rows[0]))
    audit = {
        "protocol": "paper_family_disjoint_v2",
        "seed": args.seed,
        "outer_folds": args.folds,
        "weight_application": "loss_only",
        "input_data": str(args.data),
        "input_data_sha256": sha256(args.data),
        "output_data": str(args.output_data),
        "output_data_sha256": sha256(args.output_data),
        "manifest": str(args.output),
        "manifest_sha256": sha256(args.output),
        "exclusions": str(args.exclusions),
        "exclusions_sha256": sha256(args.exclusions),
        "original_windows": int(len(groups)),
        "retained_windows": int(len(retained_groups)),
        "excluded_windows": int(derived_mask.sum()),
        "original_groups": int(len(set(groups))),
        "retained_groups": int(len(set(retained_groups))),
        "excluded_derived_alias_groups": len(derived_groups),
        "known_alias_families": len(exclusion_rows),
        "known_cross_fold_families": len(cross_fold),
        "fold_summaries": summaries,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", choices=("grouped_v1", "family_disjoint_v2"),
                        default="grouped_v1")
    parser.add_argument("--output-data", type=Path, default=V2_DATA)
    parser.add_argument("--exclusions", type=Path, default=V2_EXCLUSIONS)
    parser.add_argument("--audit", type=Path, default=V2_AUDIT)
    parser.add_argument("--expected-derived-aliases", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.data = args.data.resolve()
    if args.protocol == "family_disjoint_v2":
        args.output = (args.output or V2_MANIFEST).resolve()
        args.output_data = args.output_data.resolve()
        args.exclusions = args.exclusions.resolve()
        args.audit = args.audit.resolve()
        rows, summaries = create_family_disjoint_v2(args)
        print(
            f"protocol=paper_family_disjoint_v2 input={args.data} "
            f"output_data={args.output_data} manifest={args.output} seed={args.seed} folds={args.folds}"
        )
    else:
        args.output = (args.output or V1_MANIFEST).resolve()
        data = np.load(args.data)
        groups = data["group"].astype(str)
        is_fall = data["is_fall_window"].astype(np.uint8)
        fold_ids = stratified_group_folds(groups, is_fall, args.folds, args.seed)
        rows = manifest_rows(groups, is_fall, fold_ids)
        summaries = fold_summary(rows, args.folds)
        write_csv(args.output, rows, ("group", "outer_fold", "event_type", "windows"))
    for summary in summaries:
        print(
            f"fold={summary['outer_fold']}: groups={summary['groups']}, "
            f"fall_groups={summary['fall_groups']}, normal_groups={summary['normal_groups']}, "
            f"windows={summary['windows']}"
        )
    print(f"Saved frozen manifest with {len(rows)} groups to {args.output}")


if __name__ == "__main__":
    main()
