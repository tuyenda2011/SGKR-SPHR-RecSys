"""Rebuild metadata/text from streamed raw metadata; keep interaction files fixed."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.bundle import (
    DatasetBundle,
    publish_bundle,
    refresh_manifest_inventory,
    resolve_bundle,
    rollback_bundle,
    sha256_file,
)
from src.data.export import metadata_frame_for_export
from src.data.pipeline import run_metadata_only
from src.data.preprocessing import parse_categories, parse_category_paths


def stream_metadata(path: Path, item_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Retain all brands for the unchanged vocabulary; retain text only for cohort."""
    rows = []
    counts = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            counts["raw_lines"] += 1
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
            except (SyntaxError, ValueError, TypeError):
                counts["invalid_lines"] += 1
                continue
            if not isinstance(record, dict) or not isinstance(record.get("asin"), str):
                counts["invalid_lines"] += 1
                continue
            row = {"asin": record["asin"], "brand": record.get("brand")}
            if record["asin"] in item_ids:
                row.update(title=record.get("title"), categories=record.get("categories", []))
                paths = parse_category_paths(row["categories"])
                counts["cohort_rows"] += 1
                counts["cohort_multi_path"] += len(paths) > 1
                counts["cohort_no_category"] += not paths
            rows.append(row)
    if counts["invalid_lines"]:
        raise ValueError(f"Metadata contains {counts['invalid_lines']} invalid records")
    frame = pd.DataFrame(rows)
    if frame["asin"].duplicated().any():
        raise ValueError("Duplicate raw metadata ASIN; resolve explicitly before rebuild")
    return frame, dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle")
    parser.add_argument("--staged_bundle", help="Resume a verified metadata-only stage before publication")
    parser.add_argument("--raw_metadata", default="data/raw/amazon-electronics/meta_Electronics.json.gz")
    parser.add_argument("--output", default="artifacts/data_quality_fix_20261008/rebuild.json")
    parser.add_argument("--no-activate", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output already exists; use a fresh output path")
    base = resolve_bundle(args.bundle)
    raw_path = Path(args.raw_metadata).resolve()
    source = next(entry for entry in base.manifest["source_files"] if Path(entry["path"]).name == raw_path.name)
    if sha256_file(raw_path) != source["sha256"]:
        raise ValueError("Raw metadata differs from the pinned source")
    with base.artifact("mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    if args.staged_bundle:
        stage_root = Path(args.staged_bundle).resolve()
        if stage_root.parent != (ROOT / ".tmp" / "data-pipeline" / "staging").resolve():
            parser.error("Resume must refer to a local, unpublished staging directory")
        counts = {}
        existing_bundle = False
    else:
        raw_items, counts = stream_metadata(raw_path, set(mappings["item2id"]))
        counts["changed_category_display"] = sum(
            parse_categories(row.categories) != mappings["item_metadata"][mappings["item2id"][row.asin]]["categories"]
            for row in raw_items[raw_items["asin"].isin(mappings["item2id"])].itertuples()
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".stream.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        stage = run_metadata_only(bundle=base.root, raw_items=raw_items, publish=False)
        existing_bundle = isinstance(stage, DatasetBundle)
        stage_root = stage.root if existing_bundle else stage
    candidate = resolve_bundle(stage_root, require_build_dir=False)
    unchanged = {}
    for name in ("train.parquet", "val.parquet", "test.parquet", "disliked_interactions.parquet"):
        unchanged[name] = sha256_file(base.artifact(name)) == sha256_file(candidate.artifact(name))
    if not all(unchanged.values()):
        raise ValueError("Metadata rebuild changed interaction files")
    with candidate.artifact("mappings.pkl").open("rb") as stream:
        updated = pickle.load(stream)
    if args.staged_bundle:
        resolution = candidate.manifest["statistics"]["metadata_resolution"]
        counts = {
            "raw_valid_records": resolution["raw_meta_records"],
            "cohort_rows": resolution["filtered_asin_matched"],
            "cohort_multi_path": sum(len(meta.get("category_paths", [])) > 1 for meta in updated["item_metadata"].values()),
            "cohort_no_category": sum(not meta.get("category_paths") for meta in updated["item_metadata"].values()),
            "changed_category_display": sum(meta["categories"] != mappings["item_metadata"][idx]["categories"] for idx, meta in updated["item_metadata"].items()),
            "resumed_from_verified_stage": True,
        }
        metadata_path = candidate.csv_dir / "metadata.csv"
        metadata_frame_for_export(updated).to_csv(metadata_path, index=False, encoding="utf-8-sig", lineterminator="\n")
        export_path = candidate.csv_dir / "export_manifest.json"
        export = json.loads(export_path.read_text(encoding="utf-8"))
        for entry in export["files"]:
            if entry.get("kind") == "metadata":
                entry.update(sha256=sha256_file(metadata_path), size_bytes=metadata_path.stat().st_size)
        export_path.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    for key in ("user2id", "item2id", "user_disliked_items"):
        if mappings.get(key) != updated.get(key):
            raise ValueError(f"Metadata rebuild changed {key}")
    report = {
        "base_build_id": base.build_id,
        "build_id": candidate.build_id,
        "raw_metadata_sha256": source["sha256"],
        "stream_audit": counts,
        "interaction_hashes_unchanged": unchanged,
        "brand_sources": dict(Counter(meta.get("brand_source") for meta in updated["item_metadata"].values())),
        "review_status": "title fallback remains unverified; no automatic promotion to verified overrides",
        "feature_policy_selection": "pending validation-only comparison",
    }
    if existing_bundle:
        published = candidate if args.no_activate else rollback_bundle(candidate.root)
        report["reused_existing_bundle"] = True
    else:
        (candidate.root / "reports" / "metadata_rebuild_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        refresh_manifest_inventory(candidate.root)
        published = publish_bundle(candidate.root, activate=not args.no_activate)
    report.update(bundle=str(published.root), activated=not args.no_activate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
