"""Deterministic group-disjoint folds for paper-track experiments."""

from __future__ import annotations

import numpy as np
import csv
import re
from collections import defaultdict
from pathlib import Path


_DERIVED_MULTICAM_ALIAS = re.compile(
    r"^fall::MultiFallEvents/MulticamTrain_(chute\d+)$"
)
_CANONICAL_MULTICAM_SOURCE = re.compile(
    r"^fall::MulticamTrain/(?:train|val)/Fall/(chute\d+)/cam4\.avi$"
)


def physical_source_family(group):
    """Return the physical recording family for known path aliases."""
    value = str(group)
    match = _DERIVED_MULTICAM_ALIAS.fullmatch(value)
    if match is None:
        match = _CANONICAL_MULTICAM_SOURCE.fullmatch(value)
    if match is not None:
        return f"mcfd::{match.group(1).lower()}"
    return value


def is_derived_source_alias(group):
    """Identify event clips derived from a retained full Multicam recording."""
    return _DERIVED_MULTICAM_ALIAS.fullmatch(str(group)) is not None


def stratified_group_folds(groups, is_fall, folds=5, seed=42):
    """Assign complete fall and normal groups to balanced outer folds."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    groups = np.asarray(groups).astype(str)
    is_fall = np.asarray(is_fall).astype(np.uint8)
    assignments = {}
    rng = np.random.default_rng(seed)
    for category in (0, 1):
        candidates = np.unique(groups[is_fall == category]).copy()
        rng.shuffle(candidates)
        for index, group in enumerate(candidates):
            assignments[group] = index % folds
    return np.asarray([assignments[group] for group in groups], dtype=np.int16)


def stratified_family_folds(families, is_fall, source_dataset, folds=5, seed=42):
    """Assign physical families using dataset/event strata and window balance."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    families = np.asarray(families).astype(str)
    is_fall = np.asarray(is_fall).astype(np.uint8)
    source_dataset = np.asarray(source_dataset).astype(str)
    if not (len(families) == len(is_fall) == len(source_dataset)):
        raise ValueError("families, is_fall, and source_dataset must have equal length")

    units = []
    for family in np.unique(families):
        indices = np.flatnonzero(families == family)
        event_values = np.unique(is_fall[indices])
        dataset_values = np.unique(source_dataset[indices])
        if len(event_values) != 1:
            raise ValueError(f"physical family mixes fall and normal events: {family}")
        if len(dataset_values) != 1:
            raise ValueError(f"physical family spans source datasets: {family}")
        units.append(
            {
                "family": family,
                "event": int(event_values[0]),
                "dataset": str(dataset_values[0]),
                "windows": int(len(indices)),
            }
        )

    strata = defaultdict(list)
    for unit in units:
        strata[(unit["dataset"], unit["event"])].append(unit)

    rng = np.random.default_rng(seed)
    assignments = {}
    event_counts = np.zeros((2, folds), dtype=np.int64)
    group_counts = np.zeros(folds, dtype=np.int64)
    window_counts = np.zeros(folds, dtype=np.int64)
    for stratum in sorted(strata):
        candidates = strata[stratum]
        rng.shuffle(candidates)
        candidates.sort(key=lambda unit: unit["windows"], reverse=True)
        stratum_counts = np.zeros(folds, dtype=np.int64)
        fold_rank = np.empty(folds, dtype=np.int64)
        fold_rank[rng.permutation(folds)] = np.arange(folds)
        for unit in candidates:
            event = unit["event"]
            selected = min(
                range(folds),
                key=lambda fold: (
                    stratum_counts[fold],
                    event_counts[event, fold],
                    group_counts[fold],
                    window_counts[fold],
                    fold_rank[fold],
                ),
            )
            assignments[unit["family"]] = selected
            stratum_counts[selected] += 1
            event_counts[event, selected] += 1
            group_counts[selected] += 1
            window_counts[selected] += unit["windows"]

    return np.asarray([assignments[family] for family in families], dtype=np.int16)


def nested_masks(fold_ids, outer_fold, development_fold_offset=1):
    """Return train/development/test masks without crossing complete groups."""
    fold_ids = np.asarray(fold_ids)
    folds = int(fold_ids.max()) + 1
    if folds < 3:
        raise ValueError("nested evaluation requires at least 3 outer folds")
    test = fold_ids == outer_fold
    development = fold_ids == ((outer_fold + development_fold_offset) % folds)
    train = ~(test | development)
    return train, development, test


def fold_ids_from_manifest(groups, manifest_path):
    """Load frozen outer-fold IDs for every window group."""
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
        assignments = {row["group"]: int(row["outer_fold"]) for row in csv.DictReader(handle)}
    groups = np.asarray(groups).astype(str)
    missing = sorted(set(groups).difference(assignments))
    if missing:
        raise RuntimeError(f"Fold manifest does not cover {len(missing)} groups; first={missing[0]}")
    return np.asarray([assignments[group] for group in groups], dtype=np.int16)
