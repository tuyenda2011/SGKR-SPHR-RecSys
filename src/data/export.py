"""Deterministic, human-readable exports for an immutable dataset bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocessing import metadata_flags
from src.data.provenance import sha256_file

DEFAULT_MAX_ROWS = 500_000
DEFAULT_INSPECTION_ROWS = 1_000
EXPORT_SCHEMA_VERSION = 1


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _protect_formula(value: Any) -> Any:
    """Keep spreadsheet imports from evaluating user-controlled text."""
    if not isinstance(value, str):
        return value
    return "'" + value if value[:1] in {"=", "+", "-", "@"} else value


def _safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=("object", "string")).columns:
        result[column] = result[column].map(_protect_formula)
    return result


def _write_csv_parts(
    frame: pd.DataFrame,
    stem: str,
    output_dir: Path,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[dict[str, Any]]:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(output_dir.glob(f"{stem}.csv")) + list(output_dir.glob(f"{stem}.part-*.csv")):
        stale.unlink()
    frame = _safe_frame(frame)
    chunks = [frame.iloc[start : start + max_rows] for start in range(0, len(frame), max_rows)] or [frame]
    records = []
    for index, chunk in enumerate(chunks, start=1):
        filename = f"{stem}.csv" if len(chunks) == 1 else f"{stem}.part-{index:05d}.csv"
        path = output_dir / filename
        chunk.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
            na_rep="",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        records.append(
            {
                "path": path.name,
                "rows": len(chunk),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _timestamp_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="s", utc=True, errors="coerce").astype("string")


def enrich_split_for_export(
    frame: pd.DataFrame,
    mappings: dict[str, Any],
) -> pd.DataFrame:
    """Add stable external IDs while leaving train artifacts untouched."""
    result = frame.copy()
    id2user = {value: key for key, value in mappings.get("user2id", {}).items()}
    item_metadata = mappings.get("item_metadata", {})
    result.insert(
        0,
        "original_user_id",
        result["u_idx"].map(id2user).map(_text),
    )
    result.insert(
        1,
        "asin",
        result["i_idx"].map(
            lambda value: _text(item_metadata.get(int(value), {}).get("original_id"))
        ),
    )
    if "timestamp" in result.columns:
        result["timestamp_utc"] = _timestamp_utc(result["timestamp"])
    return result


def metadata_frame_for_export(mappings: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for i_idx, raw in sorted(mappings.get("item_metadata", {}).items()):
        meta = dict(raw)
        flags = metadata_flags(meta)
        rows.append(
            {
                "i_idx": int(i_idx),
                "asin": _text(meta.get("original_id")),
                "title": _text(meta.get("title")),
                "brand": _text(meta.get("brand")),
                "category": _text(meta.get("categories")),
                "category_paths": json.dumps(meta.get("category_paths", []), ensure_ascii=False),
                "category_policy": _text(meta.get("category_policy")),
                **{f"{key}": bool(value) for key, value in flags.items()},
                "title_source": _text(meta.get("title_source")),
                "brand_source": _text(meta.get("brand_source")),
                "brand_resolution_rule": _text(meta.get("brand_resolution_rule")),
                "brand_resolution_evidence": _text(meta.get("brand_resolution_evidence")),
                "review_status": _text(meta.get("review_status")),
            }
        )
    return pd.DataFrame(rows)


def metadata_issues_frame(mappings: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i_idx, meta in sorted(mappings.get("item_metadata", {}).items()):
        flags = metadata_flags(meta)
        common = {"i_idx": int(i_idx), "asin": _text(meta.get("original_id"))}
        if not flags["has_title"]:
            rows.append({**common, "issue": "missing_title", "value": _text(meta.get("title")), "source": _text(meta.get("title_source")), "suggestion": "Review title or keep explicit fallback"})
        if not flags["has_brand"]:
            rows.append({**common, "issue": "missing_brand", "value": _text(meta.get("brand")), "source": _text(meta.get("brand_source")), "suggestion": "Verify brand in source before adding override"})
        if not flags["has_category"]:
            rows.append({**common, "issue": "missing_category", "value": _text(meta.get("categories")), "source": "", "suggestion": "Keep item and mark category missing"})
        elif not flags["has_specific_category"]:
            rows.append({**common, "issue": "generic_category", "value": _text(meta.get("categories")), "source": "metadata", "suggestion": "Use only as generic context"})
        if meta.get("brand_source") == "title_fallback":
            rows.append({**common, "issue": "unverified_brand_fallback", "value": _text(meta.get("brand")), "source": "title_fallback", "suggestion": "Verify before promoting to override"})
    columns = ["i_idx", "asin", "issue", "value", "source", "suggestion"]
    return pd.DataFrame(rows, columns=columns)


def quality_summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section, value in report.items():
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                if isinstance(metric_value, (dict, list)):
                    continue
                rows.append({"section": section, "metric": metric, "value": metric_value, "severity": "info"})
        elif isinstance(value, (str, int, float, bool)):
            rows.append({"section": "root", "metric": section, "value": value, "severity": "info"})
    return pd.DataFrame(rows, columns=["section", "metric", "value", "severity"])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def export_bundle_csvs(
    *,
    bundle_root: str | Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    disliked: pd.DataFrame | None = None,
    report: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
    build_id: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    inspection_rows: int = DEFAULT_INSPECTION_ROWS,
    seed: int = 42,
) -> dict[str, Any]:
    """Write the standard CSV view into ``bundle_root/csv``.

    The returned manifest deliberately excludes itself to avoid a checksum
    cycle with the bundle manifest.
    """
    root = Path(bundle_root)
    target = Path(output_root) if output_root is not None else root
    csv_dir = target / "csv"
    reports_dir = target / "reports"
    csv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    split_frames = {"train": train, "val": val, "test": test}
    for name, frame in split_frames.items():
        files.extend({"kind": name, **entry} for entry in _write_csv_parts(enrich_split_for_export(frame, mappings), name, csv_dir, max_rows=max_rows))

    metadata = metadata_frame_for_export(mappings)
    files.extend({"kind": "metadata", **entry} for entry in _write_csv_parts(metadata, "metadata", csv_dir, max_rows=max_rows))
    issues = metadata_issues_frame(mappings)
    files.extend({"kind": "metadata_issues", **entry} for entry in _write_csv_parts(issues, "metadata_issues", reports_dir, max_rows=max_rows))

    if disliked is not None:
        files.extend({"kind": "disliked_interactions", **entry} for entry in _write_csv_parts(enrich_split_for_export(disliked, mappings), "disliked_interactions", csv_dir, max_rows=max_rows))
    elif not (csv_dir / "disliked_interactions.csv").exists():
        empty_dislikes = pd.DataFrame(columns=["u_idx", "i_idx", "rating", "timestamp"])
        files.extend(
            {
                "kind": "disliked_interactions",
                **entry,
                "status": "disabled",
            }
            for entry in _write_csv_parts(
                enrich_split_for_export(empty_dislikes, mappings),
                "disliked_interactions",
                csv_dir,
                max_rows=max_rows,
            )
        )

    queue_source = root / "brand_review_queue.csv"
    if not queue_source.exists():
        queue_source = root / "reports" / "brand_review_queue.csv"
    if queue_source.exists():
        queue = pd.read_csv(queue_source)
        files.extend({"kind": "brand_review_queue", **entry} for entry in _write_csv_parts(queue, "brand_review_queue", reports_dir, max_rows=max_rows))
    sample_source = root / "brand_review_sample.csv"
    if not sample_source.exists():
        sample_source = root / "reports" / "brand_review_sample.csv"
    if sample_source.exists():
        sample = pd.read_csv(sample_source)
        files.extend({"kind": "brand_review_sample", **entry} for entry in _write_csv_parts(sample, "brand_review_sample", reports_dir, max_rows=max_rows))

    quality = report or _read_json(root / "data_quality_report.json")
    if not quality:
        quality = _read_json(root / "reports" / "data_quality_report.json")
    files.extend({"kind": "data_quality_summary", **entry} for entry in _write_csv_parts(quality_summary_frame(quality), "data_quality_summary", reports_dir, max_rows=max_rows))

    samples = []
    for split, frame in split_frames.items():
        sample = enrich_split_for_export(frame, mappings).sample(
            n=min(inspection_rows, len(frame)), random_state=seed
        ).assign(split=split)
        samples.append(sample)
    inspection = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    files.extend({"kind": "inspection_sample", **entry} for entry in _write_csv_parts(inspection, "inspection_sample", csv_dir, max_rows=max_rows))

    # Manifest paths are relative to the export root, so a consumer can join
    # them without knowing whether the export was written into a bundle or to
    # a separate inspection directory.
    csv_kinds = {"train", "val", "test", "metadata", "disliked_interactions", "inspection_sample"}
    for entry in files:
        folder = "csv" if entry["kind"] in csv_kinds else "reports"
        entry["path"] = f"{folder}/{entry['path']}"

    export_manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "build_id": build_id or _read_json(root / "manifest.json").get("build_id"),
        "encoding": "utf-8-sig",
        "null_policy": "empty cell",
        "formula_protection": "prefix apostrophe for text beginning =,+,-,@",
        "csv_max_rows_per_file": max_rows,
        "inspection_rows_per_split": inspection_rows,
        "seed": seed,
        "files": files,
    }
    (csv_dir / "export_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_manifest
