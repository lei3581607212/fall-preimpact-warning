"""Build a new v2 NPZ with the frozen 80-D paper physics profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.features import PAPER_PHYSICS_PROFILE, prepare_features


DEFAULT_INPUT = ROOT / "data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz"
DEFAULT_OUTPUT = ROOT / "data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2_physics80.npz"
DEFAULT_AUDIT = ROOT / "outputs/paper_track/v2_preflight/precomputed_physics80_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    source = np.load(args.input)
    features = prepare_features(source["X"].astype(np.float32, copy=False), PAPER_PHYSICS_PROFILE)
    if features.shape != (len(source["X"]), int(source["window_size"]), 80):
        raise RuntimeError(f"unexpected precomputed shape: {features.shape}")
    payload = {key: (features if key == "X" else source[key]) for key in source.files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(args.output)
    check = np.load(args.output, mmap_mode="r")
    for key in source.files:
        if key == "X":
            continue
        left, right = source[key], check[key]
        equal = np.array_equal(left, right, equal_nan=True) if np.issubdtype(left.dtype, np.inexact) else np.array_equal(left, right)
        if not equal:
            raise RuntimeError(f"metadata mismatch after precomputation: {key}")
    audit = {
        "protocol": "paper_family_disjoint_v2",
        "feature_profile": PAPER_PHYSICS_PROFILE,
        "input_path": str(args.input),
        "input_sha256": sha256(args.input),
        "output_path": str(args.output),
        "output_sha256": sha256(args.output),
        "shape": list(check["X"].shape),
        "dtype": str(check["X"].dtype),
        "non_feature_arrays_identical": True,
        "finite": bool(np.isfinite(check["X"]).all()),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
