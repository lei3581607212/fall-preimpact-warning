"""Run the frozen family-disjoint v2 paper experiment with resumable stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import torch


DEFAULT_SEEDS = (0, 7, 13, 21, 42, 123)
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
DATA = Path("data/prediction_sequences/paper_multihorizon_pose_64frame_matched_family_disjoint_v2.npz")
MANIFEST = Path("outputs/paper_track/paper_family_disjoint_v2_manifest.csv")
RUN_ROOT = Path("outputs/paper_track/paper_family_disjoint_v2_5x6")
DURATION_SIDECAR = Path("outputs/paper_track/v2_duration_metadata/normal_event_duration_sidecar.csv")


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


def run_command(command, log_path: Path, command_path: Path, cwd: Path, dry_run: bool) -> None:
    record = {"command": [str(item) for item in command], "cwd": str(cwd), "started_at": utc_now()}
    print("COMMAND:", subprocess.list2cmdline(record["command"]), flush=True)
    if dry_run:
        return
    write_json(command_path, record)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            record["command"],
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    record.update({"finished_at": utc_now(), "return_code": return_code, "log": str(log_path)})
    write_json(command_path, record)
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {subprocess.list2cmdline(record['command'])}")


def expected_checkpoint(seed: int, fold: int, epochs: int, learning_rate: float, hardneg: bool):
    return {
        "split_protocol": "paper_family_disjoint_v2",
        "feature_profile": "paper_physics_v1",
        "input_dim": 80,
        "window_size": 64,
        "window_weight_application": "loss_only",
        "hard_negative_aggregation": "max_all" if hardneg else "1s_head",
        "seed": seed,
        "outer_fold": fold,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "skip_test": True,
        "group_balanced": True,
        "risk_head_weights": [1.0, 1.5, 2.0],
        "early_lead_sampling": 2.0,
        "action_aux_weight": 0.2,
        "hard_negative_threshold": 0.5 if hardneg else None,
        "hard_negative_weight": 3.0 if hardneg else None,
    }


def validate_checkpoint(path: Path, expected) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("training_config", {})
    observed = {
        "split_protocol": checkpoint.get("split_protocol"),
        "feature_profile": checkpoint.get("feature_profile"),
        "input_dim": checkpoint.get("input_dim"),
        "window_size": checkpoint.get("window_size"),
        "window_weight_application": checkpoint.get("window_weight_application"),
        "hard_negative_aggregation": checkpoint.get("hard_negative_aggregation"),
        "seed": config.get("seed"),
        "outer_fold": config.get("outer_fold"),
        "epochs": config.get("epochs"),
        "learning_rate": config.get("learning_rate"),
        "skip_test": config.get("skip_test"),
        "group_balanced": config.get("group_balanced"),
        "risk_head_weights": config.get("risk_head_weights"),
        "early_lead_sampling": config.get("early_lead_sampling"),
        "action_aux_weight": config.get("action_aux_weight"),
        "hard_negative_threshold": config.get("hard_negative_threshold") if expected["hard_negative_threshold"] is not None else None,
        "hard_negative_weight": config.get("hard_negative_weight") if expected["hard_negative_weight"] is not None else None,
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing checkpoint contract mismatch for {path}: {json.dumps(mismatches)}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "development_mean_average_precision": checkpoint.get("development_mean_average_precision"),
        "mined_hard_normal_windows": checkpoint.get("mined_hard_normal_windows"),
        "contract": observed,
    }


def training_command(python: Path, seed: int, fold: int, model: Path, epochs: int, learning_rate: float, initial=None):
    command = [
        python,
        "scripts/train_paper_multihorizon_teacher.py",
        "--data", DATA,
        "--model", model,
        "--epochs", str(epochs),
        "--batch-size", "256",
        "--learning-rate", str(learning_rate),
        "--seed", str(seed),
        "--fold-manifest", MANIFEST,
        "--outer-fold", str(fold),
        "--threads", "8",
        "--feature-profile", "paper_physics_v1",
        "--skip-test",
        "--group-balanced",
        "--risk-head-weights", "1.0", "1.5", "2.0",
        "--early-lead-sampling", "2.0",
        "--action-aux-weight", "0.2",
        "--window-weight-application", "loss_only",
    ]
    if initial is not None:
        command.extend([
            "--initial-model", initial,
            "--hard-negative-threshold", "0.5",
            "--hard-negative-weight", "3.0",
            "--hard-negative-aggregation", "max_all",
        ])
    return command


def evaluation_command(python: Path, fold: int, model: Path, output_dir: Path):
    return [
        python,
        "scripts/evaluate_paper_event_policy.py",
        "--data", DATA,
        "--model", model,
        "--output-dir", output_dir,
        "--max-false-alert-rate", str(1 / 12),
        "--primary-lead-seconds", "0.5",
        "--batch-size", "256",
        "--threads", "8",
        "--fold-manifest", MANIFEST,
        "--outer-fold", str(fold),
        "--normal-duration-manifest", DURATION_SIDECAR,
    ]


def evaluation_status(output_dir: Path, checkpoint_hash: str, duration_sidecar_hash: str):
    metadata_path = output_dir / "evaluation_metadata.json"
    if not metadata_path.exists() or not (output_dir / "development_policy_scan.csv").exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        return None
    if metadata.get("normal_duration_sidecar_sha256") != duration_sidecar_hash:
        return None
    if (output_dir / "infeasible_policy_summary.json").exists():
        return "INFEASIBLE"
    if (output_dir / "selected_policy.json").exists() and (output_dir / "heldout_event_evaluation.json").exists():
        return "COMPLETE"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    python = args.python.resolve()
    if sorted(args.seeds) != sorted(DEFAULT_SEEDS) or sorted(args.folds) != sorted(DEFAULT_FOLDS):
        print("WARNING: non-primary seed/fold selection; this is not the complete 5x6 protocol.", flush=True)

    if args.dry_run:
        run_command([python, "scripts/audit_paper_v2_preflight.py"], Path(), Path(), project_root, True)
        run_command([python, "scripts/build_paper_v2_duration_sidecar.py"], Path(), Path(), project_root, True)
        for seed in args.seeds:
            for fold in args.folds:
                output_dir = RUN_ROOT / f"seed{seed}" / f"outer{fold}"
                base = Path("models") / f"paper_family_disjoint_v2_outer{fold}_seed{seed}_base.pth"
                hardneg = Path("models") / f"paper_family_disjoint_v2_outer{fold}_seed{seed}_hardneg.pth"
                run_command(training_command(python, seed, fold, base, 8, 2e-4), Path(), Path(), project_root, True)
                run_command(
                    training_command(python, seed, fold, hardneg, 6, 5e-5, initial=base),
                    Path(), Path(), project_root, True,
                )
                run_command(evaluation_command(python, fold, hardneg, output_dir), Path(), Path(), project_root, True)
        return

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_state_path = RUN_ROOT / "run_state.json"
    state = {
        "status": "RUNNING",
        "protocol": "paper_family_disjoint_v2",
        "weight_application": "loss_only",
        "seeds": args.seeds,
        "folds": args.folds,
        "started_at": utc_now(),
        "python": str(python),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "artifacts": {
            "data": str(DATA),
            "data_sha256": sha256(DATA),
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256(MANIFEST),
            "normal_duration_sidecar": str(DURATION_SIDECAR),
            "normal_duration_sidecar_sha256": sha256(DURATION_SIDECAR),
        },
        "runs": {},
    }
    write_json(run_state_path, state)
    try:
        preflight = [python, "scripts/audit_paper_v2_preflight.py"]
        run_command(preflight, RUN_ROOT / "preflight.log", RUN_ROOT / "preflight_command.json", project_root, args.dry_run)
        duration_preflight = [python, "scripts/build_paper_v2_duration_sidecar.py"]
        run_command(
            duration_preflight,
            RUN_ROOT / "duration_preflight.log",
            RUN_ROOT / "duration_preflight_command.json",
            project_root,
            args.dry_run,
        )
        duration_sidecar_hash = sha256(DURATION_SIDECAR)
        for seed in args.seeds:
            for fold in args.folds:
                key = f"seed{seed}/outer{fold}"
                output_dir = RUN_ROOT / f"seed{seed}" / f"outer{fold}"
                output_dir.mkdir(parents=True, exist_ok=True)
                base = Path("models") / f"paper_family_disjoint_v2_outer{fold}_seed{seed}_base.pth"
                hardneg = Path("models") / f"paper_family_disjoint_v2_outer{fold}_seed{seed}_hardneg.pth"
                state["runs"][key] = {"status": "RUNNING", "started_at": utc_now()}
                write_json(run_state_path, state)

                base_expected = expected_checkpoint(seed, fold, 8, 2e-4, False)
                if base.exists() and not args.dry_run:
                    print(f"RESUME: validating {base}", flush=True)
                else:
                    run_command(
                        training_command(python, seed, fold, base, 8, 2e-4),
                        output_dir / "base_train.log",
                        output_dir / "base_train_command.json",
                        project_root,
                        args.dry_run,
                    )
                if not args.dry_run:
                    state["runs"][key]["base_checkpoint"] = validate_checkpoint(base, base_expected)
                    write_json(run_state_path, state)

                hardneg_expected = expected_checkpoint(seed, fold, 6, 5e-5, True)
                if hardneg.exists() and not args.dry_run:
                    print(f"RESUME: validating {hardneg}", flush=True)
                else:
                    run_command(
                        training_command(python, seed, fold, hardneg, 6, 5e-5, initial=base),
                        output_dir / "hardneg_train.log",
                        output_dir / "hardneg_train_command.json",
                        project_root,
                        args.dry_run,
                    )
                if not args.dry_run:
                    hardneg_record = validate_checkpoint(hardneg, hardneg_expected)
                    state["runs"][key]["hardneg_checkpoint"] = hardneg_record
                    write_json(run_state_path, state)
                    status = evaluation_status(output_dir, hardneg_record["sha256"], duration_sidecar_hash)
                else:
                    status = None
                if status is None:
                    run_command(
                        evaluation_command(python, fold, hardneg, output_dir),
                        output_dir / "event_policy.log",
                        output_dir / "event_policy_command.json",
                        project_root,
                        args.dry_run,
                    )
                    if not args.dry_run:
                        status = "INFEASIBLE" if (output_dir / "infeasible_policy_summary.json").exists() else "COMPLETE"
                        write_json(
                            output_dir / "evaluation_metadata.json",
                            {
                                "status": status,
                                "checkpoint": str(hardneg),
                                "checkpoint_sha256": hardneg_record["sha256"],
                                "normal_duration_sidecar": str(DURATION_SIDECAR),
                                "normal_duration_sidecar_sha256": duration_sidecar_hash,
                                "finished_at": utc_now(),
                            },
                        )
                else:
                    print(f"RESUME: evaluation already {status} for {key}", flush=True)
                state["runs"][key].update({"status": status or "DRY_RUN", "finished_at": utc_now()})
                write_json(run_state_path, state)
        state.update({"status": "COMPLETE", "finished_at": utc_now()})
        write_json(run_state_path, state)
    except Exception as error:
        state.update({"status": "FAILED", "finished_at": utc_now(), "error": repr(error)})
        write_json(run_state_path, state)
        raise


if __name__ == "__main__":
    main()
