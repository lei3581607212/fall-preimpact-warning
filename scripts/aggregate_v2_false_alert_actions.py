"""Aggregate audited v2 false-alert events into the T7/V14 table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/paper_track/v2_summary/false_alert_events_reconstructed.csv"
DEFAULT_AUDIT = ROOT / "outputs/paper_track/v2_summary/false_alert_reconstruction_audit.json"
DEFAULT_OUTPUT = ROOT / "outputs/paper_track/v2_figure_data/failure_modes.csv"
DEFAULT_DETAIL = ROOT / "outputs/paper_track/v2_summary/false_alert_action_table.csv"
DEFAULT_SUMMARY = ROOT / "outputs/paper_track/v2_summary/false_alert_action_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    if not args.input.exists() or not args.audit.exists():
        raise FileNotFoundError("F7 reconstruction input or audit is missing")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("mismatch_run_count", 0) != 0:
        raise RuntimeError("refusing to aggregate F7 rows while mismatched runs remain")
    rows = list(csv.DictReader(args.input.open(newline="", encoding="utf-8-sig")))
    if len(rows) != int(audit.get("false_alert_rows_released", -1)):
        raise RuntimeError("released-row count does not match the F7 audit")
    counts = Counter(row.get("category", "unclassified") or "unclassified" for row in rows)
    action_counts = Counter(
        (row.get("action", "unannotated") or "unannotated",
         row.get("category", "unclassified") or "unclassified")
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "count"])
        writer.writeheader()
        writer.writerows({"category": key, "count": counts[key]} for key in sorted(counts))
    args.detail.parent.mkdir(parents=True, exist_ok=True)
    with args.detail.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action", "category", "count", "percentage"])
        writer.writeheader()
        writer.writerows(
            {
                "action": action,
                "category": category,
                "count": count,
                "percentage": count / len(rows),
            }
            for (action, category), count in sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))
        )
    summary = {
        "protocol": audit.get("protocol"),
        "matched_run_count": audit.get("matched_run_count"),
        "mismatch_run_count": audit.get("mismatch_run_count"),
        "false_alert_rows_released": len(rows),
        "action_counts": dict(sorted(counts.items())),
        "detailed_action_counts": {
            action: count for (action, _), count in sorted(action_counts.items())
        },
        "input": str(args.input),
        "audit": str(args.audit),
        "output": str(args.output),
        "detail": str(args.detail),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
