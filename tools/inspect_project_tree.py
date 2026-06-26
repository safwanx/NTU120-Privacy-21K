#!/usr/bin/env python3
"""Inspect the ACCV project layout from the project root.

Run from the project root:
    python3 tools/inspect_project_tree.py

Outputs:
    project_tree_report.txt   compact readable directory tree
    project_tree_report.json  structured report for debugging path mapping
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".ipynb_checkpoints",
    ".cache",
    "node_modules",
    "venv",
    ".venv",
}

RUNTIME_HEAVY_DIR_NAMES = {
    "frames",
    "detections",
    "anonymized",
    "features",
}

KEY_FILENAMES = {
    "metadata.csv",
    "pilot_design.md",
    "pilot_pipeline.py",
    "asset_manifest.md",
    "dataset_recommendation.md",
    "findings.md",
    "requirements.txt",
    "paths.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ACCV project folder structure.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root. Defaults to cwd.")
    parser.add_argument("--max-depth", type=int, default=8, help="Text tree max depth.")
    parser.add_argument("--json-depth", type=int, default=20, help="Structured JSON max depth.")
    parser.add_argument("--include-runtime", action="store_true", help="Expand frames/detections/anonymized/features dirs.")
    parser.add_argument("--sample-files", type=int, default=1, help="Sample file names per directory.")
    parser.add_argument("--one-file-only", action="store_true", default=True, help="Keep only one sample file per directory.")
    parser.add_argument("--full-scan", action="store_true", help="Also compute recursive totals and key-file paths.")
    parser.add_argument("--txt-out", default="project_tree_report.txt")
    parser.add_argument("--json-out", default="project_tree_report.json")
    return parser.parse_args()


def should_skip_dir(path: Path, include_runtime: bool) -> bool:
    if path.name in DEFAULT_SKIP_DIR_NAMES:
        return True
    if not include_runtime and path.name in RUNTIME_HEAVY_DIR_NAMES:
        return True
    return False


def safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def list_children(path: Path) -> tuple[list[Path], list[Path], list[str]]:
    errors: list[str] = []
    dirs: list[Path] = []
    files: list[Path] = []
    try:
        children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError as exc:
        return [], [], [str(exc)]

    for child in children:
        try:
            if child.is_dir():
                dirs.append(child)
            elif child.is_file():
                files.append(child)
        except OSError as exc:
            errors.append(f"{child}: {exc}")
    return dirs, files, errors


def summarize_dir(path: Path, root: Path, include_runtime: bool, sample_files: int) -> dict[str, Any]:
    dirs, files, errors = list_children(path)
    visible_dirs = [item for item in dirs if not should_skip_dir(item, include_runtime)]
    skipped_dirs = [item.name for item in dirs if should_skip_dir(item, include_runtime)]
    total_file_bytes = 0
    for file_path in files:
        stat = safe_stat(file_path)
        if stat is not None:
            total_file_bytes += stat.st_size

    try:
        rel = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel = str(path)

    return {
        "path": "." if rel == "." else rel,
        "absolute_path": str(path.resolve()),
        "direct_dir_count": len(dirs),
        "visible_dir_count": len(visible_dirs),
        "skipped_dirs": skipped_dirs,
        "direct_file_count": len(files),
        "direct_file_bytes": total_file_bytes,
        "sample_files": [item.name for item in files[:sample_files]],
        "key_files": [item.name for item in files if item.name in KEY_FILENAMES],
        "errors": errors,
    }


def build_json_tree(
    path: Path,
    root: Path,
    depth: int,
    max_depth: int,
    include_runtime: bool,
    sample_files: int,
) -> dict[str, Any]:
    node = summarize_dir(path, root, include_runtime, sample_files)
    if depth >= max_depth:
        node["children_truncated"] = True
        node["children"] = []
        return node

    dirs, _, _ = list_children(path)
    children = []
    for child in dirs:
        if should_skip_dir(child, include_runtime):
            continue
        children.append(build_json_tree(child, root, depth + 1, max_depth, include_runtime, sample_files))
    node["children"] = children
    return node


def directory_totals(root: Path, include_runtime: bool) -> dict[str, Any]:
    total_dirs = 0
    total_files = 0
    total_bytes = 0
    skipped = 0
    errors: list[str] = []

    stack = [root]
    while stack:
        current = stack.pop()
        total_dirs += 1
        dirs, files, child_errors = list_children(current)
        errors.extend(child_errors[:10])
        for file_path in files:
            total_files += 1
            stat = safe_stat(file_path)
            if stat is not None:
                total_bytes += stat.st_size
        for directory in dirs:
            if should_skip_dir(directory, include_runtime):
                skipped += 1
                continue
            stack.append(directory)

    return {
        "dir_count_walked": total_dirs,
        "file_count_walked": total_files,
        "total_bytes_walked": total_bytes,
        "skipped_dir_count": skipped,
        "errors_sample": errors[:20],
    }


def make_text_tree(
    path: Path,
    root: Path,
    lines: list[str],
    prefix: str,
    connector: str,
    depth: int,
    max_depth: int,
    include_runtime: bool,
    sample_files: int,
) -> None:
    dirs, files, errors = list_children(path)
    visible_dirs = [item for item in dirs if not should_skip_dir(item, include_runtime)]
    skipped_count = len(dirs) - len(visible_dirs)
    stat = safe_stat(path)
    size_note = "" if stat is None else ""
    label = path.name if path != root else root.name
    note = f" ({len(visible_dirs)} dirs, {len(files)} files"
    if skipped_count:
        note += f", {skipped_count} skipped dirs"
    if errors:
        note += f", {len(errors)} errors"
    sample = [item.name for item in files[:sample_files]]
    if sample:
        note += f", sample_file={sample[0]}"
    note += ")"
    lines.append(f"{prefix}{connector}{label}/{note}{size_note}")

    if depth >= max_depth:
        if visible_dirs:
            child_prefix = prefix + ("    " if connector == "`-- " else "|   " if connector else "")
            lines.append(f"{child_prefix}... depth limit reached ({len(visible_dirs)} dirs below)")
        return

    child_prefix = prefix + ("    " if connector == "`-- " else "|   " if connector else "")
    for index, directory in enumerate(visible_dirs):
        is_last = index == len(visible_dirs) - 1
        branch = "`-- " if is_last else "|-- "
        make_text_tree(
            directory,
            root,
            lines,
            child_prefix,
            branch,
            depth + 1,
            max_depth,
            include_runtime,
            sample_files,
        )


def collect_key_paths(root: Path, include_runtime: bool) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {name: [] for name in KEY_FILENAMES}
    stack = [root]
    while stack:
        current = stack.pop()
        dirs, files, _ = list_children(current)
        for directory in dirs:
            if not should_skip_dir(directory, include_runtime):
                stack.append(directory)
        for path in files:
            if path.name not in found:
                continue
            try:
                found[path.name].append(str(path.relative_to(root)).replace("\\", "/"))
            except ValueError:
                found[path.name].append(str(path))
    return {key: value for key, value in found.items() if value}


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 1

    include_runtime = bool(args.include_runtime)
    sample_files = 1 if args.one_file_only else args.sample_files
    report = {
        "root": str(root),
        "cwd": str(Path.cwd().resolve()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "executable": sys.executable,
        },
        "env": {
            "ACCV_ROOT": os.environ.get("ACCV_ROOT"),
            "PILOT_ROOT": os.environ.get("PILOT_ROOT"),
            "PILOT_OUTPUT_ROOT": os.environ.get("PILOT_OUTPUT_ROOT"),
            "REPOS_DIR": os.environ.get("REPOS_DIR"),
            "MODELS_DIR": os.environ.get("MODELS_DIR"),
            "NTU_ROOT": os.environ.get("NTU_ROOT"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
        },
        "options": {
            "include_runtime": include_runtime,
            "text_max_depth": args.max_depth,
            "json_max_depth": args.json_depth,
            "full_scan": bool(args.full_scan),
        },
        "totals": directory_totals(root, include_runtime) if args.full_scan else {"skipped": "use --full-scan for recursive totals"},
        "key_paths": collect_key_paths(root, include_runtime) if args.full_scan else {},
        "tree": build_json_tree(root, root, 0, args.json_depth, include_runtime, sample_files),
    }

    json_out = root / args.json_out
    txt_out = root / args.txt_out

    json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"ACCV project tree report",
        f"root: {root}",
        f"include_runtime: {include_runtime}",
        f"totals: {report['totals']}",
        "",
    ]
    if report["key_paths"]:
        lines.append("Key paths:")
        for name, paths in sorted(report["key_paths"].items()):
            lines.append(f"  {name}:")
            for path in paths[:20]:
                lines.append(f"    - {path}")
            if len(paths) > 20:
                lines.append(f"    ... {len(paths) - 20} more")
    else:
        lines.append("Key paths: skipped (use --full-scan to collect)")
    lines.extend(["", "Directory tree:"])
    make_text_tree(root, root, lines, "", "", 0, args.max_depth, include_runtime, sample_files)
    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {txt_out}")
    print(f"Wrote: {json_out}")
    print("Paste project_tree_report.txt first. If I need more detail, paste/upload project_tree_report.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
