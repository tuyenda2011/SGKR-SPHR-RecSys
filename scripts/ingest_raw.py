"""Ingest raw datasets into ``data/raw/<run_id>/`` and record a SHA-256 manifest.

Usage:
    python scripts/ingest_raw.py --source data/source [--move] [--run-id 20260823]

Each snapshot owns ``manifest.json`` and ``data/raw/current.json`` points to
the selected snapshot with a manifest checksum, so downstream stages can pin
the exact raw inputs.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "source"
RAW_ROOT = REPO_ROOT / "data" / "raw"
RAW_POINTER_PATH = REPO_ROOT / "data" / "raw" / "current.json"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, streaming in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest(source: Path, move: bool = False, run_id: str | None = None) -> dict:
    """Copy (or move) raw files into a timestamped folder and write the manifest.

    Returns the manifest dictionary that was persisted inside the raw snapshot.
    """
    files = sorted(
        p for p in source.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No raw data files found in {source}")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = RAW_ROOT / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for src in files:
        dst = target_dir / src.name
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)
        entries.append(
            {
                "path": f"raw/{run_id}/{dst.name}",
                "sha256": sha256_file(dst),
                "size_bytes": dst.stat().st_size,
                "original_name": src.name,
            }
        )
        print(f"  ingested {src.name} -> data/raw/{run_id}/{dst.name}")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "mode": "move" if move else "copy",
        "num_files": len(entries),
        "files": entries,
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pointer = {
        "schema_version": 1,
        "run_id": run_id,
        "snapshot_path": target_dir.relative_to(RAW_ROOT).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
    }
    RAW_POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_POINTER_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
    temporary.replace(RAW_POINTER_PATH)
    print(f"Manifest written: {manifest_path.relative_to(REPO_ROOT)} ({len(entries)} files)")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Directory containing raw input files (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run identifier")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        sys.exit(f"Source directory does not exist: {args.source}")

    ingest(args.source, move=args.move, run_id=args.run_id)


if __name__ == "__main__":
    main()
