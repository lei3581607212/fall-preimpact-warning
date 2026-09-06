"""Generate publication figures only from the family-disjoint v2 protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "paper_track" / "v2_figure_data"
OUT_DIR = ROOT / "submission" / "figures"
PROTOCOL = "paper_family_disjoint_v2"
FAR_BUDGET = 1.0 / 12.0
SEEDS = [0, 7, 13, 21, 42, 123]

COLORS = {
    "ink": "#1F2937",
    "blue": "#2F6B8A",
    "green": "#2E7D5B",
    "amber": "#B7791F",
    "red": "#B23A48",
    "gray": "#D6DCE2",
    "light_blue": "#E8F1F5",
    "light_green": "#E7F2EC",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def read_rows(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing v2 figure input: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty v2 figure input: {path}")
    return rows


def require_fields(rows: list[dict[str, str]], fields: set[str], name: str) -> None:
    missing = fields - set(rows[0])
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")


def load_v2_contract() -> dict:
    path = DATA_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing v2 metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "protocol_version": PROTOCOL,
        "known_cross_fold_families": 0,
        "outer_folds": 5,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"v2 metadata {key}={metadata.get(key)!r}; expected {expected!r}")
    if sorted(metadata.get("seeds", [])) != SEEDS:
        raise ValueError(f"v2 metadata seeds must equal {SEEDS}")
    if abs(float(metadata.get("far_budget", -1)) - FAR_BUDGET) > 1e-4:
        raise ValueError(f"v2 metadata far_budget must equal {FAR_BUDGET:.6f}")
    for key in ("manifest_path", "manifest_sha256", "data_path", "data_sha256"):
        if not metadata.get(key):
            raise ValueError(f"v2 metadata missing {key}")
    return metadata


def plot_protocol() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    steps = [
        ("1  Family audit", "full video + aliases\n+ clips share one key", COLORS["light_blue"]),
        ("2  Outer split", "5 family-disjoint folds\n+ outer fold stays sealed", COLORS["light_green"]),
        ("3  Inner development", "fit on training families\n+ select policy on dev", "#FFF4DD"),
        ("4  Outer report", "all events + failures\n+ FAR, recall, lead time", "#F4E8EA"),
    ]
    xs = [1.5, 4.5, 7.5, 10.5]
    for idx, ((title, detail, fill), x) in enumerate(zip(steps, xs)):
        box = FancyBboxPatch(
            (x - 1.15, 1.2),
            2.3,
            1.55,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=COLORS["ink"],
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x, 2.28, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(x, 1.70, detail, ha="center", va="center", fontsize=9, linespacing=1.4)
        if idx < len(xs) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 1.18, 1.98),
                    (xs[idx + 1] - 1.18, 1.98),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.2,
                    color=COLORS["ink"],
                )
            )
    ax.add_patch(
        FancyArrowPatch(
            (10.5, 1.05),
            (7.5, 1.05),
            connectionstyle="arc3,rad=-0.28",
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color=COLORS["red"],
        )
    )
    ax.text(9.0, 0.42, "X  No model, threshold, or loss selection from outer results", ha="center", fontsize=9.2, color=COLORS["red"])
    ax.text(
        6.0,
        3.42,
        "Family-disjoint nested evaluation protocol",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        6.0,
        3.05,
        "Analysis unit: physical source family; policy selection: development events only",
        ha="center",
        fontsize=9.2,
        color=COLORS["blue"],
    )
    save_figure(fig, "V1c_experiment_protocol")


def plot_horizons() -> None:
    rows = read_rows("main_results.csv")
    require_fields(
        rows,
        {"method", "horizon_seconds", "recall", "ci_low", "ci_high", "fall_events", "eligible"},
        "main_results.csv",
    )
    rows = [row for row in rows if row["eligible"].lower() == "true"]
    methods = sorted({row["method"] for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers = ["o", "s", "^"]
    for idx, method in enumerate(methods):
        subset = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["horizon_seconds"]))
        x = np.array([float(row["horizon_seconds"]) for row in subset])
        y = np.array([float(row["recall"]) for row in subset])
        lo = np.array([float(row["ci_low"]) for row in subset])
        hi = np.array([float(row["ci_high"]) for row in subset])
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker=markers[idx % len(markers)], capsize=4, lw=1.8, label=method)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1%}", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="Required lead time (s)", ylabel="Event-level recall", xticks=[0.25, 0.5, 1.0], ylim=(0, 1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "V10_multihorizon_recall_v2")


def plot_ablation() -> None:
    rows = read_rows("ablation_results.csv")
    require_fields(rows, {"variant", "delta_recall_050", "ci_low", "ci_high", "eligible"}, "ablation_results.csv")
    rows = [row for row in rows if row["eligible"].lower() == "true"]
    rows.sort(key=lambda row: float(row["delta_recall_050"]))
    y = np.arange(len(rows))
    values = np.array([float(row["delta_recall_050"]) for row in rows])
    lo = np.array([float(row["ci_low"]) for row in rows])
    hi = np.array([float(row["ci_high"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, max(3.8, 0.5 * len(rows) + 1.5)))
    colors = [COLORS["red"] if value < 0 else COLORS["green"] for value in values]
    ax.barh(y, values, color=colors, alpha=0.86)
    ax.errorbar(values, y, xerr=np.vstack([values - lo, hi - values]), fmt="none", ecolor=COLORS["ink"], capsize=3)
    ax.axvline(0, color=COLORS["ink"], lw=1)
    ax.set(yticks=y, yticklabels=[row["variant"] for row in rows], xlabel="Change in event recall@0.5 s")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "V11_physics_ablation_v2")


def plot_frontier() -> None:
    rows = read_rows("policy_frontier.csv")
    require_fields(rows, {"method", "far", "recall_050", "split"}, "policy_frontier.csv")
    if any(row["split"].lower() != "development" for row in rows):
        raise ValueError("policy_frontier.csv must contain development rows only")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in sorted({row["method"] for row in rows}):
        subset = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["far"]))
        far = np.array([float(row["far"]) for row in subset])
        recall = np.maximum.accumulate([float(row["recall_050"]) for row in subset])
        ax.plot(far, recall, marker="o", ms=3.5, lw=1.7, label=method)
    ax.axvline(FAR_BUDGET, color=COLORS["red"], ls="--", lw=1.3, label=f"FAR budget {FAR_BUDGET:.2%}")
    ax.set(xlabel="Development event FAR", ylabel="Best attainable recall@0.5 s", xlim=(0, max(0.12, ax.get_xlim()[1])), ylim=(0, 1))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    save_figure(fig, "V12_far_recall_frontier_v2")


def plot_feasibility() -> None:
    rows = read_rows("seed_fold_results.csv")
    require_fields(rows, {"seed", "fold", "status", "far"}, "seed_fold_results.csv")
    expected = {(seed, fold) for seed in SEEDS for fold in range(5)}
    keyed = {(int(row["seed"]), int(row["fold"])): row for row in rows}
    if set(keyed) != expected:
        raise ValueError(f"seed_fold_results.csv must contain exactly {len(expected)} prespecified seed-fold rows")
    status_value = {"pass": 0, "far_exceeded": 1, "infeasible": 2}
    matrix = np.zeros((len(SEEDS), 5), dtype=int)
    labels = np.empty((len(SEEDS), 5), dtype=object)
    for i, seed in enumerate(SEEDS):
        for fold in range(5):
            row = keyed[(seed, fold)]
            status = row["status"].lower()
            if status not in status_value:
                raise ValueError(f"unsupported seed-fold status: {status}")
            matrix[i, fold] = status_value[status]
            labels[i, fold] = "INFEASIBLE" if status == "infeasible" else f"FAR {float(row['far']):.1%}"
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.imshow(matrix, cmap=ListedColormap([COLORS["light_green"], "#F8E4C5", "#E6E8EB"]), vmin=0, vmax=2, aspect="auto")
    ax.set(xticks=range(5), xticklabels=[f"fold {fold}" for fold in range(5)], yticks=range(len(SEEDS)), yticklabels=[f"seed {seed}" for seed in SEEDS])
    for i in range(len(SEEDS)):
        for fold in range(5):
            ax.text(fold, i, labels[i, fold], ha="center", va="center", fontsize=8, color=COLORS["ink"])
    ax.set_xlabel("Outer fold; green=pass, amber=FAR exceeded, gray=infeasible")
    save_figure(fig, "V13_seed_fold_feasibility_v2")


def plot_failures() -> None:
    rows = read_rows("failure_modes.csv")
    require_fields(rows, {"category", "count"}, "failure_modes.csv")
    rows.sort(key=lambda row: int(row["count"]))
    counts = np.array([int(row["count"]) for row in rows])
    total = counts.sum()
    if total <= 0:
        raise ValueError("failure_modes.csv must contain at least one event")
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.4, max(3.8, 0.48 * len(rows) + 1.5)))
    ax.barh(y, counts, color=COLORS["amber"], alpha=0.9)
    display = {
        "clear_adl": "Clear ADL",
        "fall_like": "Fall-like action",
        "unannotated_or_other": "Unannotated / other",
    }
    ax.set(
        yticks=y,
        yticklabels=[display.get(row["category"], row["category"]) for row in rows],
        xlabel="False-alert event-runs",
    )
    for yi, count in zip(y, counts):
        ax.text(count, yi, f" {count} ({count / total:.1%})", va="center", fontsize=8.5)
    ax.set_xlim(0, max(counts) * 1.28)
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "V14_failure_modes_v2")


PLOTS = {
    "protocol": plot_protocol,
    "horizons": plot_horizons,
    "ablation": plot_ablation,
    "frontier": plot_frontier,
    "feasibility": plot_feasibility,
    "failures": plot_failures,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", *PLOTS], default="all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    configure_style()

    selected = list(PLOTS) if args.figure == "all" else [args.figure]
    quantitative = [name for name in selected if name != "protocol"]
    if quantitative:
        load_v2_contract()
    if args.validate_only:
        for name in quantitative:
            input_name = {
                "horizons": "main_results.csv",
                "ablation": "ablation_results.csv",
                "frontier": "policy_frontier.csv",
                "feasibility": "seed_fold_results.csv",
                "failures": "failure_modes.csv",
            }[name]
            read_rows(input_name)
        print(f"validated: {', '.join(selected)}")
        return 0
    for name in selected:
        PLOTS[name]()
    print(f"generated: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
