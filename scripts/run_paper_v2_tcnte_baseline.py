"""Run a same-protocol TCNTE or pure-TCN baseline on the frozen v2 split."""

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
PRECOMPUTED = Path("data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2_physics80.npz")
MANIFEST = Path("outputs/paper_track/paper_family_disjoint_v2_manifest.csv")
SIDECAR = Path("outputs/paper_track/v2_duration_metadata/normal_event_duration_sidecar.csv")


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


def run_command(command: list, log_path: Path, command_path: Path, cwd: Path) -> None:
    record = {"command": [str(item) for item in command], "cwd": str(cwd), "started_at": utc_now()}
    print("COMMAND:", subprocess.list2cmdline(record["command"]), flush=True)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(command_path, record)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(record["command"], cwd=str(cwd), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
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


def train_command(python: Path, seed: int, fold: int, model: Path, epochs: int,
                  learning_rate: float, baseline: str = "tcnte", initial: Path | None = None) -> list:
    command = [
        str(python), "scripts/train_paper_multihorizon_teacher.py", "--data", str(DATA),
        "--precomputed-features", str(PRECOMPUTED),
        "--model", str(model), "--epochs", str(epochs), "--batch-size", "256",
        "--learning-rate", str(learning_rate), "--seed", str(seed), "--fold-manifest", str(MANIFEST),
        "--outer-fold", str(fold), "--threads", "8", "--feature-profile", "paper_physics_v1",
        "--skip-test", "--group-balanced", "--risk-head-weights", "1.0", "1.5", "2.0",
        "--early-lead-sampling", "2.0", "--action-aux-weight", "0.2", "--architecture", baseline,
        "--window-weight-application", "loss_only",
    ]
    if initial is not None:
        command.extend(["--initial-model", str(initial), "--hard-negative-threshold", "0.5",
                         "--hard-negative-weight", "3.0", "--hard-negative-aggregation", "max_all"])
    return command


def eval_command(python: Path, fold: int, model: Path, output_dir: Path) -> list:
    return [str(python), "scripts/evaluate_paper_event_policy.py", "--data", str(PRECOMPUTED),
            "--model", str(model), "--output-dir", str(output_dir), "--max-false-alert-rate", str(1 / 12),
            "--primary-lead-seconds", "0.5", "--batch-size", "256", "--threads", "8",
            "--fold-manifest", str(MANIFEST), "--outer-fold", str(fold),
            "--normal-duration-manifest", str(SIDECAR)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--baseline", choices=("tcnte", "tcn"), default="tcnte")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    python = args.python.resolve()
    baseline = args.baseline
    run_root = Path(f"outputs/paper_track/paper_family_disjoint_v2_{baseline}_baseline_5x6")
    for path in (DATA, PRECOMPUTED, MANIFEST, SIDECAR):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.dry_run:
        for seed in SEEDS:
            for fold in FOLDS:
                base = Path("models") / f"paper_family_disjoint_v2_{baseline}_outer{fold}_seed{seed}_base.pth"
                hard = Path("models") / f"paper_family_disjoint_v2_{baseline}_outer{fold}_seed{seed}_hardneg.pth"
                out = run_root / f"seed{seed}" / f"outer{fold}"
                print(subprocess.list2cmdline(train_command(python, seed, fold, base, 8, 2e-4, baseline)))
                print(subprocess.list2cmdline(train_command(python, seed, fold, hard, 6, 5e-5, baseline, base)))
                print(subprocess.list2cmdline(eval_command(python, fold, hard, out)))
        return 0
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "run_state.json"
    state = {
        "status": "RUNNING", "protocol": "paper_family_disjoint_v2", "baseline": baseline,
        "seeds": list(SEEDS), "folds": list(FOLDS), "feature_profile": "paper_physics_v1",
        "input_dim": 80, "weight_application": "loss_only", "far_budget": 1 / 12,
        "data": {"path": str(DATA), "sha256": sha256(DATA)},
        "precomputed_features": {"path": str(PRECOMPUTED), "sha256": sha256(PRECOMPUTED)},
        "manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)},
        "duration_sidecar": {"path": str(SIDECAR), "sha256": sha256(SIDECAR)},
        "python": str(python), "started_at": utc_now(), "runs": {},
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("data", {}).get("sha256") != state["data"]["sha256"] or previous.get("manifest", {}).get("sha256") != state["manifest"]["sha256"]:
            raise RuntimeError(f"existing {baseline} state does not match frozen v2 data/manifest")
        previous_precomputed = previous.get("precomputed_features", {}).get("sha256")
        if previous_precomputed and previous_precomputed != state["precomputed_features"]["sha256"]:
            raise RuntimeError(f"existing {baseline} state does not match precomputed v2 features")
        if previous.get("baseline") != baseline:
            raise RuntimeError("existing baseline state has a different architecture")
        state = previous
        state["status"] = "RUNNING"
        state["precomputed_features"] = {
            "path": str(PRECOMPUTED), "sha256": sha256(PRECOMPUTED)
        }
    write_json(state_path, state)
    try:
        for seed in SEEDS:
            for fold in FOLDS:
                key = f"seed{seed}/outer{fold}"
                run_dir = run_root / f"seed{seed}" / f"outer{fold}"
                base = Path("models") / f"paper_family_disjoint_v2_{baseline}_outer{fold}_seed{seed}_base.pth"
                hard = Path("models") / f"paper_family_disjoint_v2_{baseline}_outer{fold}_seed{seed}_hardneg.pth"
                existing = state["runs"].get(key, {})
                if existing.get("status") in {"COMPLETE", "INFEASIBLE"}:
                    continue
                state["runs"][key] = {"status": "RUNNING", "seed": seed, "outer_fold": fold, "started_at": utc_now()}
                write_json(state_path, state)
                if not base.exists():
                    run_command(train_command(python, seed, fold, base, 8, 2e-4, baseline), run_dir / "base_train.log", run_dir / "base_train_command.json", root)
                if not hard.exists():
                    run_command(train_command(python, seed, fold, hard, 6, 5e-5, baseline, base), run_dir / "hardneg_train.log", run_dir / "hardneg_train_command.json", root)
                run_command(eval_command(python, fold, hard, run_dir), run_dir / "event_policy.log", run_dir / "event_policy_command.json", root)
                status = "INFEASIBLE" if (run_dir / "infeasible_policy_summary.json").exists() else "COMPLETE"
                state["runs"][key].update({"status": status, "base_checkpoint": str(base), "hardneg_checkpoint": str(hard), "finished_at": utc_now()})
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
