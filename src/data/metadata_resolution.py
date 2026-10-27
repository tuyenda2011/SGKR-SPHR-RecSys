"""Resolve metadata with explicit sources and conservative title matching."""

import csv
import random
import re
from collections import Counter
from pathlib import Path

from src.data.preprocessing import (
    CATEGORY_POLICY,
    clean_text,
    metadata_flags,
    parse_categories,
    parse_category_paths,
)

# A small, explicit allowlist is safer than accepting every short token in a
# title. These brands are too short for the general frequency rule.
SHORT_BRAND_ALLOWLIST = frozenset(
    {"GE", "RCA", "JVC", "KLH", "LG", "HP", "IBM", "WD"}
)
SHORT_BRAND_KEYS = frozenset(brand.casefold() for brand in SHORT_BRAND_ALLOWLIST)
EXCLUDED_BRAND_TOKENS = frozenset(
    {
        "generic", "unknown", "none", "case", "new", "for", "with", "the",
        "cable", "dual", "ultra", "green", "image", "exact", "cctv", "bass",
        "ipod", "ipad", "iphone", "nexus", "kindle", "galaxy", "replacement",
        "digital", "professional", "wireless", "premium", "universal",
    }
)


def usable(value):
    return metadata_flags({"brand": value})["has_brand"]


def build_brand_vocabulary(raw_items):
    """Build canonical brand spellings from repeated official metadata."""
    counts = Counter(
        clean_text(value)
        for value in (raw_items["brand"] if "brand" in raw_items.columns else [])
        if usable(value)
    )
    vocabulary = [
        brand
        for brand, count in counts.items()
        if count >= 3
        and (len(brand) >= 4 or brand.casefold() in SHORT_BRAND_KEYS)
        and brand.casefold() not in EXCLUDED_BRAND_TOKENS
    ]
    vocabulary.extend(
        brand
        for brand in SHORT_BRAND_ALLOWLIST
        if brand.casefold() not in {item.casefold() for item in vocabulary}
    )
    vocabulary = sorted(
        set(vocabulary), key=lambda value: (-len(value), value.casefold(), value)
    )
    canonical = {brand.casefold(): brand for brand in vocabulary}
    return vocabulary, canonical, counts


def _compile_title_pattern(vocabulary, prefix):
    if not vocabulary:
        return None
    alternatives = "|".join(re.escape(brand) for brand in vocabulary)
    if prefix:
        # Unicode escapes avoid source-file encoding differences for ® and ™.
        return re.compile(
            rf"^({alternatives})(?=$|\s|[\u00ae\u2122:])", re.IGNORECASE
        )
    return re.compile(rf"(?<!\w)({alternatives})(?!\w)", re.IGNORECASE)


def title_brand_matches(title, vocabulary, canonical, prefix=False, pattern=None):
    """Return canonical brand names mentioned in a title, longest first."""
    if not title or not vocabulary:
        return []
    pattern = pattern or _compile_title_pattern(vocabulary, prefix=prefix)
    if pattern is None:
        return []
    matches = []
    for match in pattern.finditer(title):
        value = canonical.get(match.group(1).casefold())
        if value and value not in matches:
            matches.append(value)
        if prefix:
            break
    return matches


def build_brand_review_queue(item_metadata, raw_items, train_counts=None):
    """Create a deterministic manual-review queue for inferred/unknown brands."""
    vocabulary, canonical, _ = build_brand_vocabulary(raw_items)
    mention_pattern = _compile_title_pattern(vocabulary, prefix=False)
    train_counts = train_counts or {}
    rows = []
    for index, meta in item_metadata.items():
        source = meta.get("brand_source", "missing")
        if source == "title_fallback":
            candidate = meta.get("brand", "")
            rule = "title_prefix_applied"
        elif not meta.get("has_brand", False):
            candidates = title_brand_matches(
                meta.get("title", ""),
                vocabulary,
                canonical,
                prefix=False,
                pattern=mention_pattern,
            )
            candidate = " | ".join(candidates)
            rule = "title_mention_review" if candidates else "none"
        else:
            continue
        rows.append(
            {
                "asin": meta["original_id"],
                "title": meta.get("title", ""),
                "current_brand": meta.get("brand", "Unknown"),
                "brand_candidate": candidate,
                "brand_rule": rule,
                "brand_source": source,
                "train_interactions": int(train_counts.get(index, 0)),
            }
        )
    return sorted(rows, key=lambda row: (-row["train_interactions"], row["asin"]))


def write_brand_review_queue(item_metadata, raw_items, train_counts, path, sample_path=None):
    rows = build_brand_review_queue(item_metadata, raw_items, train_counts)
    fieldnames = [
        "asin", "title", "current_brand", "brand_candidate", "brand_rule",
        "brand_source", "train_interactions",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    sample_rows = []
    if sample_path is not None:
        rng = random.Random(42)
        fallback = [row for row in rows if row["brand_source"] == "title_fallback"]
        candidates = [
            row for row in rows
            if row["brand_source"] == "missing" and row["brand_candidate"]
        ]
        rng.shuffle(fallback)
        rng.shuffle(candidates)
        for group, values, limit in (
            ("fallback", fallback, 200), ("candidate", candidates, 100)
        ):
            for row in values[:limit]:
                sample_rows.append({**row, "sample_group": group, "verified_brand": "", "review_status": ""})
        sample_fields = fieldnames + ["sample_group", "verified_brand", "review_status"]
        with Path(sample_path).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sample_fields)
            writer.writeheader()
            writer.writerows(sample_rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "candidate_rows": sum(bool(row["brand_candidate"]) for row in rows),
        "auto_fallback_rows": sum(
            row["brand_source"] == "title_fallback" for row in rows
        ),
        "sample_path": str(sample_path) if sample_path is not None else None,
        "sample_rows": len(sample_rows),
    }


def resolve_metadata(item_metadata, raw_items, overrides_path):
    if raw_items["asin"].duplicated().any():
        raise ValueError("Duplicate ASIN in raw metadata")
    raw = raw_items.set_index("asin").to_dict("index")
    with open(overrides_path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    overrides = {}
    for row in rows:
        asin = row["asin"].strip()
        brand = row.get("brand_override", row.get("brand", ""))
        if not usable(brand):
            continue
        if asin in overrides:
            raise ValueError(f"Duplicate override: {asin}")
        overrides[asin] = clean_text(brand)

    brands, canonical, counts = build_brand_vocabulary(raw_items)
    prefix_pattern = _compile_title_pattern(brands, prefix=True)
    before = sum(metadata_flags(m)["has_brand"] for m in item_metadata.values())
    matched = official = 0
    for meta in item_metadata.values():
        asin = meta["original_id"]
        source = raw.get(asin, {})
        matched += asin in raw
        if "categories" in source:
            meta["categories"] = parse_categories(source["categories"])
            meta["category_paths"] = parse_category_paths(source["categories"])
            meta["category_policy"] = CATEGORY_POLICY
        old_title = meta.get("title")
        if metadata_flags({"title": source.get("title"), "original_id": asin})["has_title"]:
            meta["title"], meta["title_source"] = clean_text(source["title"]), "metadata"
        elif metadata_flags(meta)["has_title"]:
            meta["title"], meta["title_source"] = old_title, "existing"
        else:
            meta["title"], meta["title_source"] = "Unknown Item", "missing"

        old_brand = meta.get("brand")
        meta.pop("brand_candidate", None)
        meta.pop("brand_candidate_rule", None)
        brand = source.get("brand")
        if usable(brand):
            meta["brand"], meta["brand_source"] = clean_text(brand), "metadata"
            meta["brand_resolution_rule"] = "official_metadata"
            meta["brand_resolution_evidence"] = clean_text(brand)
            official += 1
        elif asin in overrides:
            meta["brand"], meta["brand_source"] = overrides[asin], "override"
            meta["brand_resolution_rule"] = "manual_override"
            meta["brand_resolution_evidence"] = asin
        else:
            match = (
                prefix_pattern.match(meta["title"])
                if prefix_pattern and meta["title_source"] != "missing"
                else None
            )
            if match:
                meta["brand"] = canonical[match.group(1).casefold()]
                meta["brand_source"] = "title_fallback"
                meta["brand_resolution_rule"] = "title_prefix_v2"
                meta["brand_resolution_evidence"] = match.group(0)
            else:
                meta["brand"] = "Unknown"
                meta["brand_source"] = "missing"
                meta["brand_resolution_rule"] = "no_verified_source"
                meta["brand_resolution_evidence"] = ""
                mentions = title_brand_matches(
                    meta["title"], brands, canonical, prefix=False
                )
                if mentions:
                    meta["brand_candidate"] = " | ".join(mentions)
                    meta["brand_candidate_rule"] = "title_mention_review"
        if meta["brand"] != old_brand:
            meta.setdefault("brand_original", old_brand)
        meta.update(metadata_flags(meta))
    return {
        "raw_meta_records": len(raw_items),
        "raw_meta_records_with_brand": sum(counts.values()),
        "filtered_asin_count": len(item_metadata),
        "filtered_asin_matched": matched,
        "matched_items_with_brand": official,
        "processed_brands_before": before,
        "processed_brands_after": sum(m["has_brand"] for m in item_metadata.values()),
        "brand_sources": dict(Counter(m["brand_source"] for m in item_metadata.values())),
        "title_sources": dict(Counter(m["title_source"] for m in item_metadata.values())),
        "brand_candidate_count": sum("brand_candidate" in m for m in item_metadata.values()),
        "fallback_policy": "official_brand_frequency_ge3_or_curated_short_title_prefix_v2",
        "short_brand_allowlist": sorted(SHORT_BRAND_ALLOWLIST),
    }


def write_brand_template(item_metadata, path):
    entered = {}
    if Path(path).exists():
        with open(path, encoding="utf-8-sig", newline="") as stream:
            entered = {
                row["asin"]: row.get("brand_override", row.get("brand", ""))
                for row in csv.DictReader(stream)
            }
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "asin", "title", "current_brand", "brand_override", "brand_source", "category"
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for meta in item_metadata.values():
            writer.writerow(
                {
                    "asin": meta["original_id"],
                    "title": meta["title"],
                    "current_brand": meta["brand"],
                    "brand_override": entered.get(meta["original_id"], ""),
                    "brand_source": meta["brand_source"],
                    "category": meta["categories"],
                }
            )
