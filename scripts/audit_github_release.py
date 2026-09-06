"""Audit the local GitHub release candidate for privacy and protocol leaks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    if (ROOT / "paper_family_disjoint_v2_manifest_public.csv").exists()
    else ROOT / "submission" / "github_release"
)
AUDIT = RELEASE / "privacy_audit.json"
CHECKSUMS = RELEASE / "release_checksums.sha256"

BANNED_SUFFIXES = {
    ".avi", ".mp4", ".mov", ".mkv", ".wav", ".mp3",
    ".npz", ".npy", ".pt", ".pth", ".onnx", ".kmodel", ".bin",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".csv", ".json", ".cff", ".yaml", ".yml",
    ".toml", ".ini", ".cfg",
}
WINDOWS_PATH = re.compile(r"(?i)(?<![a-z])(?:[a-z]:[\\/]|\\\\[\w.-]+\\)")
EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
PUBLIC_ID = re.compile(r"^(?:grp|fam)_[0-9a-f]{20}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = sorted(
        path for path in RELEASE.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(RELEASE).parts
        and "__pycache__" not in path.relative_to(RELEASE).parts
        and path.suffix.lower() != ".pyc"
    )
    issues: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(RELEASE).as_posix()
        if path.suffix.lower() in BANNED_SUFFIXES:
            issues.append({"file": relative, "kind": "banned_binary_or_media"})
        if path.suffix.lower() not in TEXT_SUFFIXES or path in {AUDIT, CHECKSUMS}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for kind, pattern in (
            ("local_absolute_path", WINDOWS_PATH),
            ("email_address", EMAIL),
            ("possible_phone_number", PHONE),
        ):
            if pattern.search(text):
                issues.append({"file": relative, "kind": kind})

    manifest = RELEASE / "paper_family_disjoint_v2_manifest_public.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "public_group_id", "public_family_id", "source_dataset",
        "outer_fold", "event_type", "windows",
    }
    if len(rows) != 284:
        issues.append({"file": manifest.name, "kind": f"expected_284_rows_found_{len(rows)}"})
    if set(rows[0]) != required:
        issues.append({"file": manifest.name, "kind": "unexpected_manifest_columns"})
    if any(not PUBLIC_ID.fullmatch(row["public_group_id"]) for row in rows):
        issues.append({"file": manifest.name, "kind": "invalid_public_group_id"})
    if any(not PUBLIC_ID.fullmatch(row["public_family_id"]) for row in rows):
        issues.append({"file": manifest.name, "kind": "invalid_public_family_id"})
    family_folds: dict[str, set[str]] = {}
    for row in rows:
        family_folds.setdefault(row["public_family_id"], set()).add(row["outer_fold"])
    cross_fold = sum(len(folds) > 1 for folds in family_folds.values())
    if cross_fold:
        issues.append({"file": manifest.name, "kind": f"cross_fold_families_{cross_fold}"})

    blockers = []
    if not (RELEASE / "LICENSE").exists():
        blockers.append("repository_license_not_selected")
    citation = (RELEASE / "CITATION.cff").read_text(encoding="utf-8")
    if "OWNER/REPOSITORY" in citation:
        blockers.append("github_repository_url_not_configured")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not issues else "FAIL",
        "privacy_or_protocol_issues": issues,
        "publication_blockers": blockers,
        "scanned_file_count": len(files),
        "v2_manifest_rows": len(rows),
        "v2_cross_fold_public_families": cross_fold,
        "note": "Author names in CITATION.cff are intentional publication metadata.",
    }
    AUDIT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checksum_files = [path for path in files if path != CHECKSUMS]
    lines = [f"{sha256(path)}  {path.relative_to(RELEASE).as_posix()}" for path in checksum_files]
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="ascii")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {CHECKSUMS.relative_to(ROOT)} ({len(lines)} files)")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
