import html
import logging
import re
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def clean_text(text: Any) -> str:
    """Clean and unescape HTML entities and remove raw HTML tags from text."""
    if text is None or not isinstance(text, (str, bytes)):
        return UNKNOWN_PLACEHOLDER
    text_str = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
    # Decode HTML entities like &amp;, &#39;, &quot;
    text_str = html.unescape(text_str)
    # Remove HTML tags
    text_str = re.sub(r"<[^>]+>", " ", text_str)
    # Remove redundant whitespace
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str if text_str else UNKNOWN_PLACEHOLDER


UNKNOWN_PLACEHOLDER = "unknown item"

# Values that indicate an absent metadata field after normalisation.  Keep this
# list shared by all metadata diagnostics so a placeholder cannot be counted as
# present in one report and missing in another.
MISSING_TEXT_VALUES = frozenset(
    {
        "",
        "unknown",
        UNKNOWN_PLACEHOLDER,
        "electronics product",
        "unknown electronics product",
        "nan",
        "none",
        "null",
        "n/a",
    }
)


def is_missing_text(value: Any) -> bool:
    """Return whether a raw or cleaned value is an absent placeholder."""
    if value is None:
        return True
    cleaned = clean_text(value).casefold()
    return cleaned in MISSING_TEXT_VALUES

METADATA_FLAGS = (
    "has_title",
    "has_brand",
    "has_category",
    "has_specific_category",
    "has_usable_text",
)
METADATA_POLICY = "title_or_brand_or_specific_category_v2"


def metadata_flags(meta: dict) -> dict:
    """Classify cleaned source fields, never imputed labels."""
    title = clean_text(meta.get("title"))
    brand = clean_text(meta.get("brand"))
    category = clean_text(meta.get("categories"))
    has_title = not is_missing_text(title) and title.casefold() != f"item {meta.get('original_id', '')}".casefold()
    has_brand = not is_missing_text(brand)
    has_category = not is_missing_text(category)
    specific = has_category and any(
        not is_missing_text(c.strip()) and c.strip().casefold() != "electronics"
        for c in category.split(">")
    )
    return dict(
        zip(
            METADATA_FLAGS,
            (
                has_title,
                has_brand,
                has_category,
                specific,
                has_title or has_brand or specific,
            ),
        )
    )


def summarize_metadata_quality(item_metadata: dict) -> dict:
    """Count missing/fallback fields without treating placeholders as real text."""
    titles = brands = generic_categories = 0
    for meta in item_metadata.values():
        flags = metadata_flags(meta)
        titles += not flags["has_title"]
        brands += not flags["has_brand"]
        generic_categories += not flags["has_category"] or not flags["has_specific_category"]
    count = len(item_metadata)
    flags = [metadata_flags(meta) for meta in item_metadata.values()]
    complete = sum(
        f["has_title"] and f["has_brand"] and f["has_category"] for f in flags
    )
    unusable = sum(not f["has_usable_text"] for f in flags)
    return {
        "num_items": count,
        "missing_or_fallback_titles": titles,
        "missing_or_unknown_brands": brands,
        "generic_or_missing_categories": generic_categories,
        "missing_title_fraction": titles / max(1, count),
        "missing_brand_fraction": brands / max(1, count),
        "complete_metadata": complete,
        "partial_metadata": count - complete - unusable,
        "no_usable_text": unusable,
        "category_only_text": sum(f["has_specific_category"] and not f["has_title"] and not f["has_brand"] for f in flags),
    }


CATEGORY_POLICY = "separate_category_paths_v1"


def parse_category_paths(categories_raw: Any) -> list[list[str]]:
    """Keep source taxonomy branches separate, normalized and deterministic."""
    if not isinstance(categories_raw, list):
        return []
    branches = [item for item in categories_raw if isinstance(item, list)]
    flat = [item for item in categories_raw if isinstance(item, str)]
    if flat:
        branches.append(flat)
    paths = set()
    for branch in branches:
        path = []
        for token in branch:
            cleaned = clean_text(token)
            if cleaned != UNKNOWN_PLACEHOLDER and (not path or path[-1] != cleaned):
                path.append(cleaned)
        if path:
            paths.add(tuple(path))
    return [list(path) for path in sorted(paths)]


def parse_categories(categories_raw: Any) -> str:
    """Display the longest source path; break ties lexically, never merge branches."""
    paths = parse_category_paths(categories_raw)
    primary = min(paths, key=lambda path: (-len(path), tuple(path))) if paths else []
    return " > ".join(primary[-3:]) if primary else UNKNOWN_PLACEHOLDER


def preprocess_amazon_electronics(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    positive_threshold: float = 4.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[str, int],
    dict[int, dict[str, str]],
    dict[str, Any],
]:
    """Convert ratings to implicit feedback, deduplicate interactions, filter bipartite K-core,
    re-index contiguous IDs, clean metadata, and calculate graph statistics.
    """
    logger.info(
        f"Preprocessing ratings: positive threshold >= {positive_threshold}, "
        f"min_user_interactions >= {min_user_interactions}, min_item_interactions >= {min_item_interactions}"
    )

    # Normalize column names from Amazon to standard
    ratings_df = ratings_df.rename(
        columns={
            "reviewerID": "user_id",
            "asin": "item_id",
            "overall": "rating",
            "unixReviewTime": "timestamp",
        }
    )

    # 1. Filter implicit positive feedback
    interaction_columns = ["user_id", "item_id", "rating", "timestamp"]
    if "raw_row_id" in ratings_df:
        interaction_columns.append("raw_row_id")
    df = ratings_df.loc[ratings_df["rating"] >= positive_threshold, interaction_columns].copy()
    logger.info(
        f"Retained {len(df)} positive interactions out of {len(ratings_df)} total ratings."
    )

    # 2. De-duplication: For identical (user_id, item_id), keep the most recent
    # interaction; tie-break by highest rating (spec: [timestamp DESC, rating DESC]).
    initial_len = len(df)
    df = df.sort_values(by=["timestamp", "rating"], ascending=[False, False], kind="stable")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first").copy()
    num_dups = initial_len - len(df)
    ledger: dict[str, Any] = {
        "input_ratings": len(ratings_df),
        "positive_ratings": initial_len,
        "removed_by_rating": len(ratings_df) - initial_len,
        "removed_duplicates": num_dups,
        "after_dedup": len(df),
        "kcore_rounds": [],
    }
    if num_dups > 0:
        logger.info(f"Removed {num_dups} duplicate (user_id, item_id) interactions.")

    # 3. Iterative Bipartite K-core - Vectorized numpy version
    # Much faster and memory efficient than pandas operations
    from collections import Counter

    filter_pbar = tqdm(total=None, desc="Filtering Bipartite K-core", unit=" passes")
    pass_count = 0

    while True:
        pass_count += 1
        filter_pbar.update(1)

        # Count frequencies with Counter
        user_counter = Counter(df["user_id"].values)
        item_counter = Counter(df["item_id"].values)

        # Build Python sets for O(1) hash lookup
        valid_user_ids = {
            u for u, c in user_counter.items() if c >= min_user_interactions
        }
        valid_item_ids = {
            it for it, c in item_counter.items() if c >= min_item_interactions
        }

        prev_len = len(df)
        if prev_len == 0:
            break

        # Fast vectorized filtering using pandas .isin with sets
        mask = df["user_id"].isin(valid_user_ids) & df["item_id"].isin(valid_item_ids)
        df = df[mask].reset_index(drop=True)
        ledger["kcore_rounds"].append(
            {
                "round": pass_count,
                "before": prev_len,
                "removed": prev_len - len(df),
                "after": len(df),
            }
        )

        # Each nonterminal pass removes edges, so finite input guarantees
        # termination without an arbitrary cap that could return a non-core.
        if len(df) == prev_len:
            break

    filter_pbar.close()
    if df.empty:
        raise ValueError(
            "No interactions remain after positive filtering and K-core pruning"
        )
    if (
        df.groupby("user_id").size().min() < min_user_interactions
        or df.groupby("item_id").size().min() < min_item_interactions
    ):
        raise AssertionError("K-core postcondition failed")
    logger.info(
        f"Bipartite K-core converged in {pass_count} passes. Final: {len(df)} interactions."
    )

    # 4. Create contiguous 0-indexed mappings
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user2id = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item2id = {item_id: idx for idx, item_id in enumerate(unique_items)}

    df["u_idx"] = df["user_id"].map(user2id)
    df["i_idx"] = df["item_id"].map(item2id)

    # 5. Map and clean item metadata for contiguous item IDs
    items_df = items_df.drop_duplicates(subset=["asin"]).set_index("asin")
    items_dict = items_df.to_dict(orient="index")
    item_metadata = {}

    for item_asin, idx in tqdm(
        item2id.items(), desc="Cleaning & Mapping Metadata", unit=" items"
    ):
        meta = items_dict.get(
            item_asin,
            {
                "title": None,
                "brand": None,
                "categories": [],
            },
        )

        title_cleaned = clean_text(meta.get("title", f"Item {item_asin}"))
        brand_cleaned = clean_text(meta.get("brand", "Unknown"))
        categories_raw = meta.get("categories", [])
        categories_cleaned = parse_categories(categories_raw)
        # Preserve absence before parse_categories supplies a display fallback.
        if not isinstance(categories_raw, list) or not categories_raw:
            categories_cleaned = UNKNOWN_PLACEHOLDER

        item_metadata[idx] = {
            "title": title_cleaned,
            "brand": brand_cleaned,
            "categories": categories_cleaned,
            "category_paths": parse_category_paths(categories_raw),
            "category_policy": CATEGORY_POLICY,
            "original_id": str(item_asin),
        }
        item_metadata[idx].update(metadata_flags(item_metadata[idx]))

    # 6. Compute graph statistics
    num_users = len(unique_users)
    num_items = len(unique_items)
    num_interactions = len(df)
    density = num_interactions / (num_users * num_items)

    user_interaction_counts = df.groupby("u_idx").size()
    item_interaction_counts = df.groupby("i_idx").size()

    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "density": float(density),
        "mean_user_interactions": float(user_interaction_counts.mean()),
        "median_user_interactions": float(user_interaction_counts.median()),
        "min_user_interactions": int(user_interaction_counts.min()),
        "max_user_interactions": int(user_interaction_counts.max()),
        "mean_item_interactions": float(item_interaction_counts.mean()),
        "min_item_interactions": int(item_interaction_counts.min()),
        "kcore_iterations": pass_count,
        "metadata_quality": summarize_metadata_quality(item_metadata),
        "filter_ledger": {**ledger, "output_interactions": len(df)},
    }

    logger.info(
        f"Preprocessing complete: Users={num_users}, Items={num_items}, "
        f"Interactions={num_interactions}, Density={density:.6f}, "
        f"MinUserInteractions={stats['min_user_interactions']}, MinItemInteractions={stats['min_item_interactions']}"
    )

    result = df[["u_idx", "i_idx", "timestamp"]].copy()
    result.attrs.clear()  # Keep ingestion provenance in manifest, not parquet schema metadata.
    return result, user2id, item2id, item_metadata, stats
