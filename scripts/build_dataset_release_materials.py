"""Build reviewer-facing dataset and public-release manifests.

The generated files contain no raw video, frames, pose arrays, or internal file
paths. Private group keys are replaced with deterministic SHA-256 identifiers.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1_DATA = ROOT / "data" / "prediction_sequences" / "paper_multihorizon_pose_64frame_matched.npz"
V1_FOLD_MANIFEST = ROOT / "outputs" / "paper_track" / "paper_matched_5fold_manifest.csv"
V2_DATA = ROOT / "data" / "prediction_sequences" / "paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz"
V2_FOLD_MANIFEST = ROOT / "outputs" / "paper_track" / "paper_family_disjoint_v2_manifest.csv"
V2_AUDIT = ROOT / "outputs" / "paper_track" / "paper_family_disjoint_v2_audit.json"
DATA = V2_DATA
FOLD_MANIFEST = V2_FOLD_MANIFEST
SOURCE_MANIFEST = ROOT / "submission" / "dataset_release_manifest.csv"
PUBLIC_CATALOG = ROOT / "submission" / "public_dataset_catalog.csv"
PUBLIC_FOLD_MANIFEST = ROOT / "submission" / "github_release" / "paper_matched_5fold_manifest_public.csv"
PUBLIC_V2_FOLD_MANIFEST = ROOT / "submission" / "github_release" / "paper_family_disjoint_v2_manifest_public.csv"
DUPLICATE_AUDIT = ROOT / "submission" / "dataset_duplicate_family_audit.csv"


SOURCE_METADATA = {
    "CAUCAFall": {
        "official_name": "Dataset CAUCAFall",
        "access": "public",
        "upstream_modality": "RGB video and frame-level segmentation labels",
        "official_url": "https://data.mendeley.com/datasets/7w7fccy7ky/4",
        "persistent_id": "10.17632/7w7fccy7ky.4",
        "license_status": "CC BY 4.0 verified in DataCite metadata (2026-09-04)",
        "redistribution": "Derived release requires attribution and a final file-level review",
    },
    "FieldCollection": {
        "official_name": "FieldCollection (private, team-enacted)",
        "access": "private",
        "upstream_modality": "RGB video",
        "official_url": "not public",
        "persistent_id": "not assigned",
        "license_status": "participant consent and institutional ethics decision pending",
        "redistribution": "Do not release raw video; publish only anonymized manifest/statistics until authorized",
    },
    "GMDCSA24": {
        "official_name": "GMDCSA-24: A Dataset for Human Fall Detection in Videos",
        "access": "public",
        "upstream_modality": "RGB video",
        "official_url": "https://zenodo.org/records/11489420",
        "persistent_id": "10.5281/zenodo.11489420; article 10.1016/j.dib.2024.110892",
        "license_status": "CC BY 4.0 verified in Zenodo/Crossref metadata (2026-09-04)",
        "redistribution": "Derived release requires attribution and a final file-level review",
    },
    "URFD": {
        "official_name": "UR Fall Detection Dataset (URFD)",
        "access": "public download historically available",
        "upstream_modality": "synchronized RGB/depth video and accelerometer; this work uses video-derived pose only",
        "official_url": "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html",
        "persistent_id": "article 10.1016/j.cmpb.2014.09.005",
        "license_status": "legacy official URL was unreachable on 2026-09-04; redistribution terms not verified",
        "redistribution": "Do not redistribute raw or derived samples until written terms are confirmed",
    },
    "TSTModified": {
        "official_name": "TST Fall detection dataset v2 (locally modified copy)",
        "access": "public/registered access through IEEE DataPort",
        "upstream_modality": "Kinect v2 depth frames/skeleton joints and IMU; this work uses video-derived pose only",
        "official_url": "https://ieee-dataport.org/documents/tst-fall-detection-dataset-v2",
        "persistent_id": "10.21227/H2QP48",
        "license_status": "official record verified; redistribution terms require account-level confirmation",
        "redistribution": "Publish provenance and processing code only unless IEEE DataPort terms permit derivatives",
    },
    "MultiFallEvents": {
        "official_name": "MultiFallEvents (project composite from MCFD/KU Leuven sources)",
        "access": "derived project collection",
        "upstream_modality": "RGB multi-camera video",
        "official_url": "https://www.iro.umontreal.ca/~labimage/Dataset/; https://iiw.kuleuven.be/onderzoek/advise/datasets",
        "persistent_id": "Auvinet et al., DIRO Technical Report 1350 (2010)",
        "license_status": "source pages verified; redistribution terms not verified",
        "redistribution": "Do not redistribute clips; publish composite manifest and extraction code only",
    },
    "MulticamTrain": {
        "official_name": "Multiple Cameras Fall Dataset (project training subset)",
        "access": "public download historically available",
        "upstream_modality": "RGB multi-camera video",
        "official_url": "https://www.iro.umontreal.ca/~labimage/Dataset/",
        "persistent_id": "Auvinet et al., DIRO Technical Report 1350 (2010)",
        "license_status": "official source page verified; redistribution terms not verified",
        "redistribution": "Publish download instructions, checksums, and processing code; do not mirror videos",
    },
    "MCFDHardNeg": {
        "official_name": "Multiple Cameras Fall Dataset (verified normal interval)",
        "access": "public download historically available",
        "upstream_modality": "RGB multi-camera video",
        "official_url": "https://www.iro.umontreal.ca/~labimage/Dataset/",
        "persistent_id": "Auvinet et al., DIRO Technical Report 1350 (2010)",
        "license_status": "official source page verified; redistribution terms not verified",
        "redistribution": "Publish interval annotation and processing code; do not mirror videos",
    },
}


PUBLIC_DATASETS = [
    {
        "official_name": "Dataset CAUCAFall",
        "project_source_keys": "CAUCAFall",
        "modality": "RGB video; segmentation labels",
        "project_use": "directly included in grouped 5-fold CV",
        "official_url": "https://data.mendeley.com/datasets/7w7fccy7ky/4",
        "persistent_id": "10.17632/7w7fccy7ky.4",
        "repository": "Mendeley Data",
        "access_or_license": "CC BY 4.0 verified",
        "release_decision": "metadata/scripts now; derived data after final attribution review",
    },
    {
        "official_name": "GMDCSA-24",
        "project_source_keys": "GMDCSA24",
        "modality": "RGB video",
        "project_use": "directly included in grouped 5-fold CV",
        "official_url": "https://zenodo.org/records/11489420",
        "persistent_id": "10.5281/zenodo.11489420",
        "repository": "Zenodo",
        "access_or_license": "CC BY 4.0 verified",
        "release_decision": "metadata/scripts now; derived data after final attribution review",
    },
    {
        "official_name": "UR Fall Detection Dataset",
        "project_source_keys": "URFD",
        "modality": "RGB/depth video and accelerometer",
        "project_use": "video-derived 2D pose included in grouped 5-fold CV",
        "official_url": "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html",
        "persistent_id": "article 10.1016/j.cmpb.2014.09.005",
        "repository": "University of Rzeszow legacy site",
        "access_or_license": "official URL currently unreachable; terms need author confirmation",
        "release_decision": "no mirroring; publish code, provenance, and checksums only",
    },
    {
        "official_name": "TST Fall detection dataset v2",
        "project_source_keys": "TSTModified",
        "modality": "Kinect v2 depth/skeleton and IMU",
        "project_use": "131 selected depth-video sources mapped to 11 groups; 2D pose only enters model",
        "official_url": "https://ieee-dataport.org/documents/tst-fall-detection-dataset-v2",
        "persistent_id": "10.21227/H2QP48",
        "repository": "IEEE DataPort",
        "access_or_license": "official record verified; account terms require confirmation",
        "release_decision": "no mirroring; publish source selection and processing code only",
    },
    {
        "official_name": "Multiple Cameras Fall Dataset",
        "project_source_keys": "MulticamTrain; MCFDHardNeg; part of MultiFallEvents",
        "modality": "RGB multi-camera video",
        "project_use": "fall clips plus one manually verified normal interval",
        "official_url": "https://www.iro.umontreal.ca/~labimage/Dataset/",
        "persistent_id": "Auvinet et al., DIRO Technical Report 1350 (2010)",
        "repository": "University of Montreal",
        "access_or_license": "official source verified; redistribution terms not stated in local evidence",
        "release_decision": "no mirroring; publish interval/event manifest and extraction code only",
    },
    {
        "official_name": "KU Leuven high-quality fall simulation data",
        "project_source_keys": "part of MultiFallEvents",
        "modality": "2D-camera fall and ADL recordings",
        "project_use": "selected events included through project composite",
        "official_url": "https://iiw.kuleuven.be/onderzoek/advise/datasets",
        "persistent_id": "official institutional dataset page",
        "repository": "KU Leuven",
        "access_or_license": "official page verified; redistribution terms require confirmation",
        "release_decision": "no mirroring; publish provenance and extraction code only",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def public_group_id(group: str) -> str:
    return "grp_" + hashlib.sha256(group.encode("utf-8")).hexdigest()[:20]


def source_family_id(group: str) -> str:
    """Collapse known full-video/event-clip aliases to one physical-event family."""
    multi_event = re.search(r"MultiFallEvents/MulticamTrain_(chute\d+)", group)
    multicam_video = re.search(r"MulticamTrain/(?:train|val)/Fall/(chute\d+)/", group)
    match = multi_event or multicam_video
    if match:
        return f"mcfd_{match.group(1).lower()}"
    return public_group_id(group)


def main() -> int:
    data = np.load(DATA)
    groups = data["group"].astype(str)
    datasets = data["source_dataset"].astype(str)
    source_paths = data["source_path"].astype(str)

    group_to_dataset: dict[str, str] = {}
    dataset_groups: dict[str, set[str]] = defaultdict(set)
    dataset_paths: dict[str, set[str]] = defaultdict(set)
    dataset_windows = Counter(datasets.tolist())
    for group, dataset, source_path in zip(groups, datasets, source_paths):
        previous = group_to_dataset.setdefault(group, dataset)
        if previous != dataset:
            raise ValueError(f"group spans datasets: {group}: {previous} vs {dataset}")
        dataset_groups[dataset].add(group)
        dataset_paths[dataset].add(source_path)

    with FOLD_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        fold_rows = list(csv.DictReader(handle))
    if len(fold_rows) != 284:
        raise ValueError(f"expected 284 v2 fold-manifest rows, found {len(fold_rows)}")
    if {row["group"] for row in fold_rows} != set(group_to_dataset):
        raise ValueError("NPZ groups and 5-fold manifest groups differ")

    event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in fold_rows:
        event_counts[group_to_dataset[row["group"]]][row["event_type"]] += 1

    source_rows: list[dict[str, object]] = []
    for source_key in sorted(SOURCE_METADATA):
        metadata = SOURCE_METADATA[source_key]
        source_rows.append(
            {
                "source_key": source_key,
                "official_name": metadata["official_name"],
                "access": metadata["access"],
                "cv_groups": len(dataset_groups[source_key]),
                "source_videos_used": len(dataset_paths[source_key]),
                "fall_groups": event_counts[source_key]["fall"],
                "normal_groups": event_counts[source_key]["normal"],
                "windows": dataset_windows[source_key],
                "upstream_modality": metadata["upstream_modality"],
                "model_input": "standardized causal 2D pose and derived features only",
                "official_url": metadata["official_url"],
                "persistent_id": metadata["persistent_id"],
                "license_or_access_audit": metadata["license_status"],
                "redistribution_decision": metadata["redistribution"],
            }
        )
    if sum(int(row["cv_groups"]) for row in source_rows) != 284:
        raise ValueError("source-level group counts do not sum to 284")
    if sum(int(row["source_videos_used"]) for row in source_rows) != 530:
        raise ValueError("source-level video counts do not sum to 530")
    if sum(int(row["windows"]) for row in source_rows) != 47381:
        raise ValueError("source-level window counts do not sum to 47,381")

    write_csv(SOURCE_MANIFEST, source_rows, list(source_rows[0]))
    write_csv(PUBLIC_CATALOG, PUBLIC_DATASETS, list(PUBLIC_DATASETS[0]))

    public_rows = [
        {
            "public_group_id": public_group_id(row["group"]),
            "public_family_id": "fam_" + hashlib.sha256(
                row.get("physical_source_family", source_family_id(row["group"])).encode("utf-8")
            ).hexdigest()[:20],
            "source_dataset": group_to_dataset[row["group"]],
            "outer_fold": row["outer_fold"],
            "event_type": row["event_type"],
            "windows": row["windows"],
        }
        for row in fold_rows
    ]
    if len({row["public_group_id"] for row in public_rows}) != len(public_rows):
        raise ValueError("public group identifier collision")
    write_csv(PUBLIC_V2_FOLD_MANIFEST, public_rows, list(public_rows[0]))

    families: dict[str, list[dict[str, str]]] = defaultdict(list)
    with V1_FOLD_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        legacy_fold_rows = list(csv.DictReader(handle))
    for row in legacy_fold_rows:
        family = source_family_id(row["group"])
        if family.startswith("mcfd_chute"):
            families[family].append(row)
    duplicate_rows: list[dict[str, object]] = []
    for family, members in sorted(families.items()):
        if len(members) != 2:
            raise ValueError(f"expected two aliases for {family}, found {len(members)}")
        member_folds = sorted({row["outer_fold"] for row in members})
        duplicate_rows.append(
            {
                "source_family_id": family,
                "alias_1_group": members[0]["group"],
                "alias_1_outer_fold": members[0]["outer_fold"],
                "alias_2_group": members[1]["group"],
                "alias_2_outer_fold": members[1]["outer_fold"],
                "cross_fold": "yes" if len(member_folds) > 1 else "no",
                "required_action": "merge aliases into one group and rerun nested CV",
            }
        )
    write_csv(DUPLICATE_AUDIT, duplicate_rows, list(duplicate_rows[0]))

    print(f"wrote {SOURCE_MANIFEST.relative_to(ROOT)} ({len(source_rows)} sources)")
    print(f"wrote {PUBLIC_CATALOG.relative_to(ROOT)} ({len(PUBLIC_DATASETS)} public datasets)")
    print(f"wrote {PUBLIC_V2_FOLD_MANIFEST.relative_to(ROOT)} ({len(public_rows)} groups)")
    print(
        f"wrote {DUPLICATE_AUDIT.relative_to(ROOT)} "
        f"({len(duplicate_rows)} alias families; "
        f"{sum(row['cross_fold'] == 'yes' for row in duplicate_rows)} cross-fold)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
