"""Metadata policies and diagnostics specific to AdaptiveGCL experiments.

The canonical metadata remains unchanged.  This module creates a derived text
view for AdaptiveGCL without turning inferred values into ground truth.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from src.data.preprocessing import clean_text, metadata_flags

ADAPTIVE_TEXT_POLICY = "adaptivegcl_quality_v1"
VERIFIED_BRAND_SOURCES = frozenset({"metadata", "override"})


def is_verified_brand(meta: dict[str, Any]) -> bool:
    """Return whether a present brand came from an allowed source."""
    flags = metadata_flags(meta)
    return bool(flags["has_brand"] and meta.get("brand_source") in VERIFIED_BRAND_SOURCES)


def metadata_group(meta: dict[str, Any]) -> str:
    """Return one mutually exclusive, interpretable item-quality group."""
    flags = metadata_flags(meta)
    title = flags["has_title"]
    brand = is_verified_brand(meta)
    category = flags["has_specific_category"]
    if not flags["has_usable_text"]:
        return "unusable"
    if title and brand and category:
        return "title_brand_category"
    if title and brand:
        return "title_brand"
    if title and category:
        return "title_category"
    if brand and category:
        return "brand_category"
    if title:
        return "title_only"
    if brand:
        return "brand_only"
    if category:
        return "category_only"
    return "partial"


def adaptive_text_metadata(item_metadata: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Create a derived metadata view that excludes unverified title brands.

    Source metadata and its provenance stay untouched.  The returned copy is
    only used to generate an AdaptiveGCL text tensor.
    """
    view = deepcopy(item_metadata)
    for meta in view.values():
        if meta.get("brand_source") not in VERIFIED_BRAND_SOURCES:
            meta["brand"] = ""
            meta["brand_source"] = "excluded_unverified"
            meta["brand_resolution_rule"] = "excluded_from_adaptive_text"
        meta.update(metadata_flags(meta))
    return view


def semantic_ssl_mask(item_metadata: dict[int, dict[str, Any]], num_items: int) -> list[bool]:
    """Eligibility for item-specific semantic SSL.

    Category-only rows remain available to item fusion and user pooling, but
    they do not form item-specific positive pairs for semantic SSL.
    """
    if set(item_metadata) != set(range(num_items)):
        raise ValueError("Item metadata must contain contiguous indices")
    values = []
    for index in range(num_items):
        flags = metadata_flags(item_metadata[index])
        values.append(bool(flags["has_usable_text"] and (
            flags["has_title"] or is_verified_brand(item_metadata[index])
        )))
    return values


def metadata_view_frame(item_metadata: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """Return a row-per-item audit view for spreadsheet inspection."""
    rows = []
    for index, raw in sorted(item_metadata.items()):
        meta = dict(raw)
        flags = metadata_flags(meta)
        rows.append(
            {
                "i_idx": int(index),
                "asin": clean_text(meta.get("original_id")),
                "title": clean_text(meta.get("title")),
                "brand": clean_text(meta.get("brand")),
                "brand_source": clean_text(meta.get("brand_source")),
                "category": clean_text(meta.get("categories")),
                "metadata_group": metadata_group(meta),
                "is_verified_brand": is_verified_brand(meta),
                "category_only": metadata_group(meta) == "category_only",
                "ssl_eligible": bool(
                    flags["has_usable_text"]
                    and (flags["has_title"] or is_verified_brand(meta))
                ),
                **flags,
            }
        )
    return pd.DataFrame(rows)
