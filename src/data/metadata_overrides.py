"""Apply explicit, attributable brand corrections without changing item IDs."""

import csv
from pathlib import Path

from src.data.preprocessing import metadata_flags
from src.data.provenance import sha256_file

DEFAULT_OVERRIDES = Path(__file__).resolve().parents[2] / "data/metadata_overrides.csv"


def apply_brand_overrides(item_metadata, path=DEFAULT_OVERRIDES):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        if not all(row.get(key, "").strip() for key in ("asin", "brand", "source")):
            raise ValueError("Overrides require asin, brand and source")
        if row["asin"] in seen:
            raise ValueError(f"Duplicate override: {row['asin']}")
        seen.add(row["asin"])
    by_asin = {meta["original_id"]: meta for meta in item_metadata.values()}
    applied = 0
    for row in rows:
        meta = by_asin.get(row["asin"])
        if meta is None or (metadata_flags(meta)["has_brand"] and meta.get("brand_source") != "title_fallback"):
            continue
        meta["brand_original"] = meta.get("brand")
        meta["brand"] = row["brand"].strip()
        meta["brand_source"] = "override"
        meta["brand_override_source"] = row["source"].strip()
        meta.update(metadata_flags(meta))
        applied += 1
    return {"path": "data/metadata_overrides.csv", "sha256": sha256_file(path),
            "applied": applied, "entries": len(rows)}
