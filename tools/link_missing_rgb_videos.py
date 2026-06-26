#!/usr/bin/env python3
"""Hardlink a listed set of RGB videos into a staging folder.

The input is usually missing_rgb_videos.csv or missing_expected_rgb.csv. The
script reads the filename column, finds each video under a source root, then
places links under <output-root>/rgb.

Hardlinks require source and output to be on the same filesystem/drive. If that
is not true, pass --copy-if-link-fails.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("missing_rgb_videos.csv")
DEFAULT_SOURCE_ROOT = Path(r"C:\path\to\ACCV\pilot_pack")
DEFAULT_OUTPUT_ROOT = Path(r"C:\path\to\ACCV\missing_rgb_videos_pack")


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading Excel files requires pandas/openpyxl. Use CSV or install pandas openpyxl.") from exc
        return pd.read_excel(path).fillna("").astype(str).to_dict("records")

    raise ValueError(f"Unsupported manifest type: {path}")


def filename_from_row(row: dict[str, Any]) -> str:
    for key in ["filename", "rgb_filename", "video", "video_name"]:
        value = str(row.get(key, "")).strip()
        if value:
            return Path(value).name

    for key in ["expected_rgb_path", "rgb_path", "path"]:
        value = str(row.get(key, "")).strip()
        if value:
            return Path(value.replace("\\", "/")).name

    raise ValueError(f"Could not find a filename column in row: {row}")


def setup_from_row(row: dict[str, Any], filename: str) -> int | None:
    value = str(row.get("setup", "")).strip()
    if value.isdigit():
        return int(value)
    if filename.startswith("S") and len(filename) >= 4 and filename[1:4].isdigit():
        return int(filename[1:4])
    return None


def source_candidates(source_root: Path, row: dict[str, Any], filename: str) -> list[Path]:
    setup = setup_from_row(row, filename)
    candidates = []

    for key in ["expected_rgb_path", "rgb_path", "path"]:
        value = str(row.get(key, "")).strip()
        if value:
            raw = Path(value)
            candidates.append(raw)
            candidates.append(source_root / value.replace("\\", "/"))

    candidates.extend(
        [
            source_root / "rgb" / filename,
            source_root / filename,
        ]
    )

    if setup is not None:
        setup_dir = f"nturgbd_rgb_s{setup:03d}"
        candidates.extend(
            [
                source_root / setup_dir / filename,
                source_root / "datasets" / setup_dir / filename,
                source_root.parent / setup_dir / filename,
            ]
        )

    seen = set()
    unique = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def find_source(source_root: Path, row: dict[str, Any], filename: str) -> Path | None:
    for candidate in source_candidates(source_root, row, filename):
        if candidate.exists():
            return candidate.resolve()
    return None


def place_file(src: Path, dst: Path, copy_if_link_fails: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"

    try:
        os.link(src, dst)
        return "hardlinked"
    except OSError:
        if not copy_if_link_fails:
            raise

    shutil.copy2(src, dst)
    return "copied"


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "status", "source", "target", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardlink listed RGB videos into an output rgb folder.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV/XLSX with a filename column.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing pilot_pack/rgb, rgb, or nturgbd_rgb_sXXX folders.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root to create.")
    parser.add_argument(
        "--copy-if-link-fails",
        action="store_true",
        help="Copy files when hardlinking fails, for example across drives/filesystems.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be linked.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    out_rgb = output_root / "rgb"

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    rows = read_rows(manifest)
    report_rows: list[dict[str, Any]] = []
    counts = {"hardlinked": 0, "copied": 0, "exists": 0, "missing": 0, "error": 0}

    for row in rows:
        filename = filename_from_row(row)
        target = out_rgb / filename
        source = find_source(source_root, row, filename)

        if source is None:
            counts["missing"] += 1
            report_rows.append(
                {"filename": filename, "status": "missing", "source": "", "target": str(target), "error": ""}
            )
            continue

        if args.dry_run:
            status = "exists" if target.exists() else "would_link"
            counts["exists" if target.exists() else "hardlinked"] += 1
        else:
            try:
                status = place_file(source, target, args.copy_if_link_fails)
                counts[status] += 1
            except OSError as exc:
                status = "error"
                counts["error"] += 1
                report_rows.append(
                    {
                        "filename": filename,
                        "status": status,
                        "source": str(source),
                        "target": str(target),
                        "error": str(exc),
                    }
                )
                continue

        report_rows.append(
            {"filename": filename, "status": status, "source": str(source), "target": str(target), "error": ""}
        )

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, output_root / manifest.name)
    write_report(output_root / "link_report.csv", report_rows)

    print(f"Manifest rows : {len(rows)}")
    print(f"Source root   : {source_root}")
    print(f"Output root   : {output_root}")
    print(f"Output RGB dir: {out_rgb}")
    for key in ["hardlinked", "copied", "exists", "missing", "error"]:
        print(f"{key:12s}: {counts[key]}")
    print(f"Report        : {output_root / 'link_report.csv'}")
    return 1 if counts["missing"] or counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
