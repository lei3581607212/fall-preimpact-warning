"""Run matched family-disjoint v2 feature ablations with a resumable state file.

The full physics model is the frozen v2 reference.  This runner trains only
new feature-profile variants and evaluates every variant with the same seed,
outer fold, development policy grid, FAR budget, and duration sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


SEEDS = (0, 7, 13, 21, 42, 123)
FOLDS = (0, 1, 2, 3, 4)
DATA = Path("data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz")
MANIFEST = Path("outputs/paper_track/paper_family_disjoint_v2_manifest.csv")
SIDECAR = Path("outputs/paper_track/v2_duration_metadata/normal_event_duration_sidecar.csv")
ROOT = Path("outputs/paper_track/paper_family_disjoint_v2_matched_ablation_5x6")

# Full is a frozen reference, raw removes all 12 physics/quality descriptors,
# and no_quality keeps the eight body-motion descriptors but removes the four
# pose-validity/motion-quality descriptors.
VARIANTS = {
    "full": {"feature_profile": "paper_physics_v1", "input_dim": 80, "frozen": True},
    "no_physics": {"feature_profile": "paper_raw_v1", "input_dim": 68, "frozen": False},
    "no_quality": {"feature_profile": "paper_physics_no_quality_v1", "input_dim": 76, "frozen": False},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    temporary.replace(path)


def command_record(command: list, cwd: Path) -> dict:
    return {"command": [str(item) for item in command], "cwd": str(cwd), "started_at": utc_now()}


def run_command(command: list, log_path: Path, command_path: Path, cwd: Path) -> None:
    record = command_record(command, cwd)
    print("COMMAND:", subprocess.list2cmdline(record["command"]), flush=True)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(command_path, record)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            record["command"], cwd=str(cwd), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    record.update({"finished_at": utc_now(), "return_code": code, "log": str(log_path)})
    write_json(command_path, record)
    if code != 0:
        raise RuntimeError(f"command failed ({code}): {subprocess.list2cmdline(record['command'])}")


def train_command(python: Path, variant: str, seed: int, fold: int, model: Path,
                  epochs: int, learning_rate: float, initial: Path | None = None) -> list:
    profile = VARIANTS[variant]["feature_profile"]
    command = [
        str(python), "scripts/train_paper_multihorizon_teacher.py",
        "--data", str(DATA), "--model", str(model),
        "--epochs", str(epochs), "--batch-size", "256",
        "--learning-rate", str(learning_rate), "--seed", str(seed),
        "--fold-manifest", str(MANIFEST), "--outer-fold", str(fold),
        "--threads", "8", "--feature-profile", profile, "--skip-test",
        "--group-balanced", "--risk-head-weights", "1.0", "1.5", "2.0",
        "--early-lead-sampling", "2.0", "--action-aux-weight", "0.2",
        "--window-weight-application", "loss_only",
    ]
    if initial is not None:
        command.extend([
            "--initial-model", str(initial), "--hard-negative-threshold", "0.5",
            "--hard-negative-weight", "3.0", "--hard-negative-aggregation", "max_all",
        ])
    return command


def eval_command(python: Path, fold: int, model: Path, output_dir: Path) -> list:
    return [
        str(python), "scripts/evaluate_paper_event_policy.py",
        "--data", str(DATA), "--model", str(model), "--output-dir", str(output_dir),
        "--max-false-alert-rate", str(1 / 12), "--primary-lead-seconds", "0.5",
        "--batch-size", "256", "--threads", "8", "--fold-manifest", str(MANIFEST),
        "--outer-fold", str(fold), "--normal-duration-manifest", str(SIDECAR),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["no_physics", "no_quality"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    python = args.python.resolve()
    if not DATA.exists() or not MANIFEST.exists() or not SIDECAR.exists():
        raise FileNotFoundError("v2 data, manifest, or duration sidecar is missing")
    if sorted(args.seeds) != list(SEEDS) or sorted(args.folds) != list(FOLDS):
        raise ValueError("matched ablation requires the complete frozen 6-seed x 5-fold protocol")
    if args.dry_run:
        for variant in args.variants:
            if VARIANTS[variant]["frozen"]:
                continue
            for seed in SEEDS:
                for fold in FOLDS:
                    base = Path("models") / f"paper_family_disjoint_v2_ablation_{variant}_outer{fold}_seed{seed}_base.pth"
                    hardneg = Path("models") / f"paper_family_disjoint_v2_ablation_{variant}_outer{fold}_seed{seed}_hardneg.pth"
                    output_dir = ROOT / variant / f"seed{seed}" / f"outer{fold}"
                    print(subprocess.list2cmdline(train_command(python, variant, seed, fold, base, 8, 2e-4)))
                    print(subprocess.list2cmdline(train_command(python, variant, seed, fold, hardneg, 6, 5e-5, initial=base)))
                    print(subprocess.list2cmdline(eval_command(python, fold, hardneg, output_dir)))
        return 0
    for variant in args.variants:
        if VARIANTS[variant]["frozen"]:
            continue
    ROOT.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "run_state.json"
    state = {
        "status": "RUNNING", "protocol": "paper_family_disjoint_v2",
        "purpose": "matched_feature_ablation", "seeds": list(SEEDS), "folds": list(FOLDS),
        "variants": {key: VARIANTS[key] for key in args.variants},
        "far_budget": 1 / 12, "weight_application": "loss_only", "primary_lead_seconds": 0.5,
        "data": {"path": str(DATA), "sha256": sha256(DATA)},
        "manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)},
        "duration_sidecar": {"path": str(SIDECAR), "sha256": sha256(SIDECAR)},
        "python": str(python), "started_at": utc_now(), "runs": {},
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("data", {}).get("sha256") != state["data"]["sha256"] or previous.get("manifest", {}).get("sha256") != state["manifest"]["sha256"]:
            raise RuntimeError("existing ablation state does not match frozen v2 data/manifest")
        state = previous
        state["status"] = "RUNNING"
    write_json(state_path, state)
    try:
        for variant in args.variants:
            if VARIANTS[variant]["frozen"]:
                continue
            for seed in SEEDS:
                for fold in FOLDS:
                    key = f"{variant}/seed{seed}/outer{fold}"
                    run_dir = ROOT / variant / f"seed{seed}" / f"outer{fold}"
                    base = Path("models") / f"paper_family_disjoint_v2_ablation_{variant}_outer{fold}_seed{seed}_base.pth"
                    hardneg = Path("models") / f"paper_family_disjoint_v2_ablation_{variant}_outer{fold}_seed{seed}_hardneg.pth"
                    existing = state["runs"].get(key, {})
                    if existing.get("status") in {"COMPLETE", "INFEASIBLE"}:
                        continue
                    state["runs"][key] = {"status": "RUNNING", "variant": variant, "seed": seed, "outer_fold": fold, "feature_profile": VARIANTS[variant]["feature_profile"], "started_at": utc_now()}
                    write_json(state_path, state)
                    if not base.exists():
                        run_command(train_command(python, variant, seed, fold, base, 8, 2e-4), run_dir / "base_train.log", run_dir / "base_train_command.json", project_root)
                    if not hardneg.exists():
                        run_command(train_command(python, variant, seed, fold, hardneg, 6, 5e-5, initial=base), run_dir / "hardneg_train.log", run_dir / "hardneg_train_command.json", project_root)
                    run_command(eval_command(python, fold, hardneg, run_dir), run_dir / "event_policy.log", run_dir / "event_policy_command.json", project_root)
                    status = "INFEASIBLE" if (run_dir / "infeasible_policy_summary.json").exists() else "COMPLETE"
                    state["runs"][key].update({"status": status, "base_checkpoint": str(base), "hardneg_checkpoint": str(hardneg), "finished_at": utc_now()})
                    write_json(state_path, state)
        state["status"] = "COMPLETE"
        state["finished_at"] = utc_now()
        write_json(state_path, state)
    except Exception:
        state["status"] = "FAILED"
        state["failed_at"] = utc_now()
        write_json(state_path, state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
