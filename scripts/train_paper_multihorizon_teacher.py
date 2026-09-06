"""Train the paper-only causal multi-horizon pose Transformer teacher."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.model import (PaperMultiHorizonTeacher, PaperMultiHorizonRecoveryTeacher,
                           PaperMultiHorizonActionTeacher, PureTCNBaseline, TCNTEBaseline)
from utils.features import (PAPER_PHYSICS_NO_QUALITY_PROFILE, PAPER_PHYSICS_PROFILE,
                            PAPER_RAW_PROFILE, prepare_features)
from utils.paper_cv import fold_ids_from_manifest, nested_masks


def build_sampler_weights(groups, sample_weight, group_balanced, weight_application):
    """Build sampling probabilities without silently squaring v2 loss weights."""
    if not group_balanced:
        return None
    groups = np.asarray(groups).astype(str)
    group_counts = {group: int((groups == group).sum()) for group in np.unique(groups)}
    weights = np.asarray([1.0 / group_counts[group] for group in groups], dtype=np.float64)
    if weight_application == "sampler_and_loss":
        weights *= np.asarray(sample_weight, dtype=np.float64)
    elif weight_application != "loss_only":
        raise ValueError(f"unsupported weight application: {weight_application}")
    return weights


def split_protocol_from_manifest(path):
    if path is None:
        return "grouped_fall_normal_v1"
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        fields = set(csv.DictReader(handle).fieldnames or ())
    if "physical_source_family" in fields:
        return "paper_family_disjoint_v2"
    return "nested_grouped_5fold_v1"


def grouped_split(groups, is_fall, seed):
    """Split complete normal and fall groups independently, then join masks."""
    rng = np.random.default_rng(seed)
    assignments = {}
    for category in (0, 1):
        candidates = np.unique(groups[is_fall == category]).copy()
        rng.shuffle(candidates)
        development_end = max(1, round(len(candidates) * 0.15))
        test_end = development_end + max(1, round(len(candidates) * 0.15))
        for group in candidates[:development_end]:
            assignments[group] = "development"
        for group in candidates[development_end:test_end]:
            assignments[group] = "test"
        for group in candidates[test_end:]:
            assignments[group] = "train"
    split = np.asarray([assignments[group] for group in groups])
    return split == "train", split == "development", split == "test"


def average_precision(target, score):
    order = np.argsort(-score)
    target = target[order].astype(np.float32)
    positives = target.sum()
    if positives == 0:
        return float("nan")
    precision = np.cumsum(target) / np.arange(1, len(target) + 1)
    return float((precision * target).sum() / positives)


def evaluate(model, loader, device):
    model.eval()
    targets, scores, time_targets, time_predictions = [], [], [], []
    with torch.no_grad():
        for x, risk, time_target, time_valid in loader:
            outputs = model(x.to(device))
            risk_logits, time_prediction = outputs[:2]
            targets.append(risk.numpy())
            scores.append(torch.sigmoid(risk_logits).cpu().numpy())
            mask = time_valid.numpy().astype(bool)
            if mask.any():
                time_targets.append(time_target.numpy()[mask])
                time_predictions.append(time_prediction.cpu().numpy()[mask])
    target = np.concatenate(targets)
    score = np.concatenate(scores)
    aps = [average_precision(target[:, index], score[:, index]) for index in range(3)]
    time_mae = float("nan")
    if time_targets:
        time_mae = float(np.abs(np.concatenate(time_targets) - np.concatenate(time_predictions)).mean())
    return aps, time_mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prediction_sequences/paper_multihorizon_pose_64frame.npz"))
    parser.add_argument("--precomputed-features", type=Path, default=None,
                        help="预计算好的特征 npz（X 已为 profile 目标维度），跳过 prepare_features 循环")
    parser.add_argument("--model", type=Path, default=Path("models/paper_multihorizon_teacher_v1.pth"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-manifest", type=Path,
                        help="Frozen group-disjoint outer-fold manifest for nested cross-validation.")
    parser.add_argument("--outer-fold", type=int,
                        help="Outer test fold to use with --fold-manifest.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--feature-profile", choices=(PAPER_RAW_PROFILE, PAPER_PHYSICS_PROFILE,
                                                       PAPER_PHYSICS_NO_QUALITY_PROFILE),
                        default=PAPER_RAW_PROFILE,
                        help="Paper-only input profile; physics adds 12 causal descriptors to the raw 68 features.")
    parser.add_argument("--initial-model", type=Path,
                        help="Optional teacher checkpoint to fine-tune from.")
    parser.add_argument("--hard-negative-threshold", type=float, default=0.7,
                        help="Mine training-only normal windows above this 1-second risk score.")
    parser.add_argument("--hard-negative-weight", type=float, default=1.0,
                        help="Extra loss weight for mined training-only normal windows.")
    parser.add_argument("--hard-negative-aggregation", choices=("1s_head", "max_all"), default="1s_head",
                        help="Score used to mine hard normals: the 1-second head only (legacy) or the "
                        "maximum over all three risk heads (a normal window firing on any head is a "
                        "false-alarm source for every horizon).")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not inspect the held-out test split during iterative development.")
    parser.add_argument("--group-balanced", action="store_true",
                        help="Sample training windows with equal total weight per group.")
    parser.add_argument("--target-normal-pattern", action="append", default=[],
                        help="Substring of source_path for train-only normal actions to emphasize; repeatable.")
    parser.add_argument("--target-normal-weight", type=float, default=1.0,
                        help="Additional multiplier for targeted train-only normal windows.")
    parser.add_argument("--risk-head-weights", type=float, nargs=3, default=(1.0, 2.0, 1.0),
                        metavar=("W025", "W050", "W100"),
                        help="Loss weights for the three risk heads.")
    parser.add_argument("--horizon-positive-sampling", type=float, default=1.0,
                        help="Historical option name: extra window weight for 1-second positives; "
                             "under v2 loss_only it does not alter sampler probabilities.")
    parser.add_argument("--early-lead-sampling", type=float, default=1.0,
                        help="Historical option name: extra window weight for 0.5-1.0s fall windows; "
                             "under v2 loss_only it does not alter sampler probabilities.")
    parser.add_argument(
        "--window-weight-application",
        choices=("sampler_and_loss", "loss_only"),
        default="sampler_and_loss",
        help="Apply hard/early window weights in both sampling and risk loss (v1 legacy) "
             "or only in risk loss (required by family-disjoint v2).",
    )
    parser.add_argument("--nested-risk-weight", type=float, default=0.0,
                        help="Weight for enforcing p(0.25s) <= p(0.5s) <= p(1.0s).")
    parser.add_argument("--soft-risk-labels", action="store_true",
                        help="Use time-to-impact soft targets for training while keeping binary labels for evaluation.")
    parser.add_argument("--recovery-aux-weight", type=float, default=0.0,
                        help="Auxiliary loss weight for normal/recoverable versus pre-impact fall windows.")
    parser.add_argument("--action-aux-weight", type=float, default=0.0,
                        help="Auxiliary coarse action classification loss weight.")
    parser.add_argument("--architecture", choices=("teacher", "tcnte", "tcn"), default="teacher",
                        help="Backbone: teacher, tcnte (TCN+Transformer), or tcn (pure causal TCN).")
    parser.add_argument("--temporal-consistency-weight", type=float, default=0.0,
                        help="Penalty for abrupt risk changes between adjacent causal windows of one event.")
    parser.add_argument("--normal-spike-weight", type=float, default=0.0,
                        help="Penalty for upward risk spikes in adjacent normal windows.")
    parser.add_argument("--fall-monotonic-weight", type=float, default=0.0,
                        help="Penalty when risk decreases between adjacent pre-impact fall windows.")
    args = parser.parse_args()
    split_protocol = split_protocol_from_manifest(args.fold_manifest)
    if split_protocol == "paper_family_disjoint_v2" and args.window_weight_application != "loss_only":
        parser.error(
            "family-disjoint v2 requires --window-weight-application loss_only; "
            "sampler_and_loss is retained only for v1 regression"
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    if args.precomputed_features:
        data = np.load(args.precomputed_features)
        x = np.asarray(data["X"], dtype=np.float32)
    else:
        data = np.load(args.data)
        x = prepare_features(data["X"].astype(np.float32, copy=False), args.feature_profile)
    input_dim = int(x.shape[-1])
    groups = data["group"].astype(str)
    is_fall = data["is_fall_window"].astype(np.uint8)
    risk = np.column_stack((data["risk_025ms"], data["risk_050ms"], data["risk_100ms"])).astype(np.float32)
    remaining = data["time_to_impact_seconds"].astype(np.float32)
    time_valid = np.isfinite(remaining) & (remaining > 0) & (remaining <= 1.0)
    time_target = np.nan_to_num(np.clip(remaining, 0.0, 1.0), nan=0.0)
    train_risk = risk.copy()
    if args.soft_risk_labels:
        valid_remaining = np.isfinite(remaining) & (remaining > 0)
        for index, horizon in enumerate((0.25, 0.5, 1.0)):
            # Preserve zero targets for normal windows; within a fall event,
            # provide a smooth target that rises toward impact.
            values = np.clip((horizon - remaining) / horizon, 0.0, 1.0)
            train_risk[:, index] = np.where(valid_remaining, values, 0.0).astype(np.float32)
    if args.fold_manifest:
        if args.outer_fold is None:
            parser.error("--outer-fold is required with --fold-manifest")
        fold_ids = fold_ids_from_manifest(groups, args.fold_manifest)
        train, development, test = nested_masks(fold_ids, args.outer_fold)
    else:
        train, development, test = grouped_split(groups, is_fall, args.seed)
    previous_index = np.full(len(x), -1, dtype=np.int64)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        indices = indices[np.argsort(data["end_frame"][indices])]
        if len(indices) > 1:
            previous_index[indices[1:]] = indices[:-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = int(data["window_size"])
    if args.architecture == "tcnte":
        model = TCNTEBaseline(input_dim=input_dim, window_size=window_size).to(device)
    elif args.architecture == "tcn":
        model = PureTCNBaseline(input_dim=input_dim, window_size=window_size).to(device)
    else:
        model = (PaperMultiHorizonActionTeacher(input_dim=input_dim, window_size=window_size) if args.action_aux_weight > 0
                 else PaperMultiHorizonRecoveryTeacher(input_dim=input_dim, window_size=window_size) if args.recovery_aux_weight > 0
                 else PaperMultiHorizonTeacher(input_dim=input_dim, window_size=window_size)).to(device)
    if args.initial_model:
        checkpoint = torch.load(args.initial_model, map_location=device)
        if int(checkpoint.get("input_dim", 68)) != input_dim:
            raise RuntimeError("Initial checkpoint input_dim does not match --feature-profile")
        if checkpoint.get("feature_profile", PAPER_RAW_PROFILE) != args.feature_profile:
            raise RuntimeError("Initial checkpoint feature_profile does not match --feature-profile")
        model.load_state_dict(checkpoint["state_dict"])
    positives = risk[train].sum(axis=0)
    pos_weight = torch.tensor((train.sum() - positives) / np.maximum(positives, 1.0), dtype=torch.float32, device=device)
    risk_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    time_loss = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    hard_negative = np.zeros(int(train.sum()), dtype=bool)
    if args.initial_model and args.hard_negative_weight > 1.0:
        model.eval()
        scores = []
        with torch.no_grad():
            for (batch,) in DataLoader(TensorDataset(torch.from_numpy(x[train])), batch_size=args.batch_size):
                logits, _ = model(batch.to(device))[:2]
                head_scores = torch.sigmoid(logits).cpu().numpy()
                if args.hard_negative_aggregation == "max_all":
                    scores.append(head_scores.max(axis=1))
                else:
                    scores.append(head_scores[:, 2])
        hard_negative = (is_fall[train] == 0) & (np.concatenate(scores) >= args.hard_negative_threshold)
    sample_weight = np.where(hard_negative, args.hard_negative_weight, 1.0).astype(np.float32)
    if args.horizon_positive_sampling > 1.0:
        sample_weight *= np.where(risk[train, 2] > 0.5, args.horizon_positive_sampling, 1.0)
    if args.early_lead_sampling > 1.0:
        early_lead = np.isfinite(remaining[train]) & (remaining[train] > 0.5) & (remaining[train] <= 1.0)
        sample_weight *= np.where(early_lead, args.early_lead_sampling, 1.0)
    targeted = np.zeros(int(train.sum()), dtype=bool)
    if args.target_normal_pattern and args.target_normal_weight > 1.0:
        train_paths = data["source_path"].astype(str)[train]
        targeted = (is_fall[train] == 0) & np.logical_or.reduce([
            np.char.find(train_paths, pattern) >= 0 for pattern in args.target_normal_pattern])
        sample_weight[targeted] *= args.target_normal_weight
    recovery_target = (1.0 - is_fall).astype(np.float32)
    paths_lower = np.char.lower(data["source_path"].astype(str))
    action_target = np.zeros(len(paths_lower), dtype=np.int64)
    action_target[is_fall == 1] = 1
    action_target[(is_fall == 0) & np.char.find(paths_lower, "sit") >= 0] = 2
    action_target[(is_fall == 0) & ((np.char.find(paths_lower, "kneel") >= 0) | (np.char.find(paths_lower, "squat") >= 0))] = 3
    action_target[(is_fall == 0) & ((np.char.find(paths_lower, "pick") >= 0) | (np.char.find(paths_lower, "bend") >= 0))] = 4
    action_target[(is_fall == 0) & ((np.char.find(paths_lower, "walk") >= 0) | (np.char.find(paths_lower, "run") >= 0))] = 5
    action_target[(is_fall == 0) & ((np.char.find(paths_lower, "hop") >= 0) | (np.char.find(paths_lower, "jump") >= 0))] = 6
    action_target[(is_fall == 0) & (action_target == 0)] = 7
    train_dataset = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(train_risk[train]),
                                  torch.from_numpy(time_target[train]), torch.from_numpy(time_valid[train]),
                                  torch.from_numpy(sample_weight), torch.from_numpy(recovery_target[train]),
                                  torch.from_numpy(action_target[train]),
                                  torch.from_numpy(np.where(previous_index[train] >= 0, previous_index[train], 0)),
                                  torch.from_numpy((previous_index[train] >= 0).astype(np.uint8)),
                                  torch.from_numpy(is_fall[train]))
    train_sampler = None
    if args.group_balanced:
        train_groups = groups[train]
        sampler_weights = build_sampler_weights(
            train_groups, sample_weight, args.group_balanced,
            args.window_weight_application,
        )
        train_sampler = WeightedRandomSampler(torch.from_numpy(sampler_weights), len(sampler_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=train_sampler is None)
    development_loader = DataLoader(TensorDataset(torch.from_numpy(x[development]), torch.from_numpy(risk[development]),
                                                   torch.from_numpy(time_target[development]), torch.from_numpy(time_valid[development])),
                                    batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x[test]), torch.from_numpy(risk[test]),
                                            torch.from_numpy(time_target[test]), torch.from_numpy(time_valid[test])),
                             batch_size=args.batch_size)
    best = -float("inf")
    print(f"Device={device}; train/dev/test groups={len(set(groups[train]))}/{len(set(groups[development]))}/{len(set(groups[test]))}")
    print(f"Train positives 0.25/0.50/1.00={positives.astype(int).tolist()}; time targets={int(time_valid[train].sum())}; "
          f"mined hard normals={int(hard_negative.sum())}; targeted normals={int(targeted.sum())}; "
          f"window_weight_application={args.window_weight_application}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, riskb, timeb, validb, weightb, recoveryb, actionb, previousb, has_previousb, isfallb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb.to(device))
            risk_logits, predicted_time = outputs[:2]
            per_window = (risk_loss(risk_logits, riskb.to(device)) *
                          torch.tensor(args.risk_head_weights, device=device)).mean(dim=1)
            loss = (per_window * weightb.to(device)).sum() / weightb.to(device).sum()
            if args.nested_risk_weight > 0:
                probabilities = torch.sigmoid(risk_logits)
                nested_penalty = (torch.relu(probabilities[:, 0] - probabilities[:, 1]) +
                                  torch.relu(probabilities[:, 1] - probabilities[:, 2])).mean()
                loss = loss + args.nested_risk_weight * nested_penalty
            mask = validb.to(device).bool()
            if mask.any():
                loss = loss + 0.3 * time_loss(predicted_time[mask], timeb.to(device)[mask])
            if args.recovery_aux_weight > 0:
                # Recovery teacher returns the auxiliary logit as its third output.
                recovery_logits = outputs[2]
                loss = loss + args.recovery_aux_weight * nn.functional.binary_cross_entropy_with_logits(
                    recovery_logits, recoveryb.to(device))
            if args.action_aux_weight > 0:
                loss = loss + args.action_aux_weight * nn.functional.cross_entropy(outputs[2], actionb.to(device))
            if args.temporal_consistency_weight > 0:
                valid_previous = has_previousb.bool()
                if valid_previous.any():
                    previous_logits = model(torch.from_numpy(x[previousb[valid_previous].numpy()]).to(device))[0]
                    current_probability = torch.sigmoid(risk_logits[valid_previous.to(device)])
                    previous_probability = torch.sigmoid(previous_logits)
                    loss = loss + args.temporal_consistency_weight * nn.functional.smooth_l1_loss(
                        current_probability, previous_probability)
            if args.normal_spike_weight > 0 or args.fall_monotonic_weight > 0:
                valid_previous = has_previousb.bool()
                if valid_previous.any():
                    previous_logits = model(torch.from_numpy(x[previousb[valid_previous].numpy()]).to(device))[0]
                    current_probability = torch.sigmoid(risk_logits[valid_previous.to(device)])
                    previous_probability = torch.sigmoid(previous_logits)
                    normal_mask = (isfallb[valid_previous] == 0).to(device)
                    fall_mask = (isfallb[valid_previous] == 1).to(device)
                    if args.normal_spike_weight > 0 and normal_mask.any():
                        normal_spike = torch.relu(current_probability[normal_mask] - previous_probability[normal_mask]).mean()
                        loss = loss + args.normal_spike_weight * normal_spike
                    if args.fall_monotonic_weight > 0 and fall_mask.any():
                        fall_drop = torch.relu(previous_probability[fall_mask] - current_probability[fall_mask]).mean()
                        loss = loss + args.fall_monotonic_weight * fall_drop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        aps, time_mae = evaluate(model, development_loader, device)
        score = float(np.nanmean(aps))
        print(f"Epoch {epoch:02d}: train_loss={np.mean(losses):.4f} dev_AP={np.round(aps, 4).tolist()} dev_mean_AP={score:.4f} dev_time_MAE={time_mae:.4f}")
        if score > best:
            best = score
            args.model.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "architecture": ("tcnte_baseline_v1" if args.architecture == "tcnte"
                                  else "pure_tcn_baseline_v1" if args.architecture == "tcn"
                                  else "paper_multihorizon_action_teacher_v1" if args.action_aux_weight > 0
                                  else "paper_multihorizon_recovery_teacher_v1" if args.recovery_aux_weight > 0
                                  else "paper_multihorizon_teacher_v1"),
                "input_dim": input_dim, "feature_profile": args.feature_profile,
                "window_size": window_size, "risk_horizons_seconds": (0.25, 0.5, 1.0),
                "development_average_precision": aps, "development_mean_average_precision": score,
                "development_time_mae_seconds": time_mae,
                "training_config": vars(args), "mined_hard_normal_windows": int(hard_negative.sum()),
                "hard_negative_aggregation": args.hard_negative_aggregation,
                "split_protocol": split_protocol,
                "sampling": "group_balanced" if args.group_balanced else "window_shuffle",
                "window_weight_application": args.window_weight_application,
                "target_normal_patterns": args.target_normal_pattern,
                "target_normal_weight": args.target_normal_weight,
            }, args.model)
        scheduler.step()
    if args.skip_test:
        print(f"Saved {args.model}; held-out test was intentionally not inspected.")
        return
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_aps, test_time_mae = evaluate(model, test_loader, device)
    print(f"Held-out test AP={np.round(test_aps, 4).tolist()} mean_AP={np.nanmean(test_aps):.4f} time_MAE={test_time_mae:.4f}")
    print(f"Saved {args.model}")


if __name__ == "__main__":
    main()
