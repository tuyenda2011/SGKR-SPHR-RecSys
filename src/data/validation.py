"""Fail-fast data validation for interaction and metadata tables.

Primary engine: Great Expectations (if installed). Graceful fallback: pandas
assertions with identical fail-fast semantics, so CI never breaks on a missing
heavy dependency.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.preprocessing import MISSING_TEXT_VALUES

logger = logging.getLogger(__name__)

RAW_COLUMN_ALIASES = {
    "reviewerID": "user_id",
    "asin": "item_id",
    "overall": "rating",
    "unixReviewTime": "timestamp",
}

try:  # pragma: no cover - exercised only when GE is installed
    import great_expectations as gx

    GE_AVAILABLE = True
except ImportError:
    gx = None
    GE_AVAILABLE = False


class ValidationError(Exception):
    """Raised when any schema or value expectation fails."""


def _format_violations(violations: list[str]) -> str:
    header = f"{len(violations)} validation violation(s):"
    body = "\n".join(f"  - {v}" for v in violations)
    return f"{header}\n{body}"


def _fallback_validate_interactions(df: pd.DataFrame) -> list[str]:
    """Pandas-assertion mirror of the interaction schema expectations."""
    df = df.rename(columns=RAW_COLUMN_ALIASES)
    violations = []

    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        violations.append(f"missing required columns: {sorted(missing)}")
        return violations

    for col in ("user_id", "item_id"):
        n_null = int(df[col].isna().sum())
        if n_null:
            violations.append(f"{col}: {n_null} null values (expected 0)")
        empty = df[col].notna() & df[col].astype(str).str.strip().eq("")
        if empty.any():
            violations.append(f"{col}: {int(empty.sum())} empty values")
        numeric_ids = pd.to_numeric(df[col], errors="coerce")
        numeric_values = numeric_ids[df[col].notna() & numeric_ids.notna()]
        if (numeric_values <= 0).any():
            violations.append(f"{col}: {int((numeric_values <= 0).sum())} values <= 0")

    rating = pd.to_numeric(df["rating"], errors="coerce")
    bad_rating = rating.isna() | ~rating.between(1, 5, inclusive="both")
    if len(bad_rating):
        bad_count = int(bad_rating.sum())
        if bad_count:
            violations.append(f"rating: {bad_count} invalid values or outside [1, 5]")

    timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
    bad_ts = timestamp.isna() | (timestamp < 0)
    if bad_ts.any():
        violations.append(f"timestamp: {int(bad_ts.sum())} invalid or negative values")

    return violations


def _fallback_validate_metadata(df: pd.DataFrame) -> list[str]:
    """Pandas-assertion mirror of the metadata expectations."""
    violations = []

    for col in ("title", "brand"):
        if col not in df.columns:
            violations.append(f"metadata missing required column: {col}")
            continue
        empty = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if len(empty):
            violations.append(f"{col}: {len(empty)} null/empty strings after cleaning")

    if "original_id" in df.columns:
        dup = int(df["original_id"].duplicated().sum())
        if dup:
            violations.append(f"original_id: {dup} duplicated entries")

    return violations


def summarize_raw_metadata_fields(df: pd.DataFrame) -> dict:
    """Report missing raw metadata without treating it as an interaction error."""
    fields = {}
    for column in ("title", "brand", "categories"):
        if column not in df.columns:
            fields[column] = {
                "column_present": False,
                "missing_or_placeholder": len(df),
                "null_or_blank": len(df),
                "placeholder": 0,
            }
            continue

        values = df[column]

        def is_null_or_blank(value):
            if value is None:
                return True
            if isinstance(value, (list, dict)):
                return len(value) == 0
            try:
                missing = pd.isna(value)
                if isinstance(missing, bool) and missing:
                    return True
                if hasattr(missing, "item") and bool(missing.item()):
                    return True
            except (TypeError, ValueError):
                pass
            return isinstance(value, (str, bytes)) and not str(value).strip()

        def is_placeholder(value):
            if not isinstance(value, (str, bytes)):
                return False
            text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            return text.strip().casefold() in MISSING_TEXT_VALUES

        null_or_blank = values.map(is_null_or_blank)
        placeholder = values.map(is_placeholder)
        fields[column] = {
            "column_present": True,
            "missing_or_placeholder": int((null_or_blank | placeholder).sum()),
            "null_or_blank": int(null_or_blank.sum()),
            "placeholder": int(placeholder.sum()),
        }
    return {"rows": len(df), "fields": fields}


def validate_raw_metadata(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate identity integrity before metadata is joined to filtered items.

    Title, brand and categories may be absent in the source and are reported by
    :func:`summarize_raw_metadata_fields`; only the ASIN identity is required to
    be present and unique.
    """
    violations = []
    if "asin" not in df.columns:
        violations.append("metadata missing required column: asin")
    else:
        null_asin = df["asin"].isna() | df["asin"].astype(str).str.strip().eq("")
        if null_asin.any():
            violations.append(f"asin: {int(null_asin.sum())} null/empty values")
        duplicates = int(df["asin"].duplicated().sum())
        if duplicates:
            violations.append(f"asin: {duplicates} duplicated entries")

    if violations and raise_on_error:
        raise ValidationError(_format_violations(violations))
    if violations:
        logger.error("Raw metadata validation issues:\n%s", _format_violations(violations))
    return violations


def _ge_validate(df: pd.DataFrame, expectations: list[tuple], suite_name: str, html_report_dir: Path | None):
    """Run expectations through Great Expectations; return violation strings."""
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(f"{suite_name}_ds")
    asset = datasource.add_dataframe_asset(f"{suite_name}_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe(f"{suite_name}_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name=suite_name)
    for method, kwargs in expectations:
        suite.add_expectation(getattr(gx.expectations, method)(**kwargs))
    suite = context.suites.add(suite)

    results = batch.validate(expectation_suite=suite, result_format="SUMMARY")
    violations = [
        f"{res.expectation_config.type}: {res.result}"
        for res in results.results
        if not res.success
    ]

    if html_report_dir is not None and violations:
        html_report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("GE HTML report generation skipped (ephemeral context).")

    return violations


INTERACTION_EXPECTATIONS = [
    ("ExpectColumnValuesToNotBeNull", {"column": "user_id"}),
    ("ExpectColumnValuesToBeOfType", {"column": "user_id", "type_": "int64"}),
    ("ExpectColumnValuesToBeGreaterThan", {"column": "user_id", "value": 0}),
    ("ExpectColumnValuesToNotBeNull", {"column": "item_id"}),
    ("ExpectColumnValuesToBeOfType", {"column": "item_id", "type_": "int64"}),
    ("ExpectColumnValuesToBeGreaterThan", {"column": "item_id", "value": 0}),
    ("ExpectColumnValuesToBeBetween", {"column": "rating", "min_value": 1, "max_value": 5}),
    ("ExpectColumnValuesToNotBeNull", {"column": "timestamp"}),
    ("ExpectColumnValuesToBeBetween", {"column": "timestamp", "min_value": 0}),
]

METADATA_EXPECTATIONS = [
    ("ExpectColumnValuesToNotBeNull", {"column": "title"}),
    ("ExpectColumnValuesToNotMatchRegex", {"column": "title", "regex": r"&amp;|<br\s*/?>|<[^>]+>"}),
    ("ExpectColumnValuesToNotBeNull", {"column": "brand"}),
]


def validate_interactions(
    df: pd.DataFrame,
    html_report_dir: Path | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate the interactions table schema; return violation list.

    Raises ``ValidationError`` when ``raise_on_error`` is True and any check fails.
    """
    logger.info("Validating interactions table (%d rows)...", len(df))

    violations = _fallback_validate_interactions(df)

    if violations:
        detail = _format_violations(violations)
        if raise_on_error:
            raise ValidationError(detail)
        logger.error("Interaction validation issues:\n%s", detail)
    else:
        logger.info("Interactions validation PASSED.")

    return violations


def validate_processed_interactions(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate mapped interaction data used by training and evaluation."""
    violations = []
    required = {"u_idx", "i_idx", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        violations.append(f"missing required columns: {sorted(missing)}")
    else:
        for column in ("u_idx", "i_idx", "timestamp"):
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = values.isna() | (values < 0)
            if invalid.any():
                violations.append(
                    f"{column}: {int(invalid.sum())} invalid or negative values"
                )
        duplicates = int(df.duplicated(["u_idx", "i_idx"]).sum())
        if duplicates:
            violations.append(f"interactions: {duplicates} duplicate user-item pairs")

    if violations:
        detail = _format_violations(violations)
        if raise_on_error:
            raise ValidationError(detail)
        logger.error("Processed interaction validation issues:\n%s", detail)
    else:
        logger.info("Processed interactions validation PASSED.")
    return violations


def validate_metadata(
    df: pd.DataFrame,
    html_report_dir: Path | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate the item metadata table; semantics match ``validate_interactions``."""
    logger.info("Validating metadata table (%d rows)...", len(df))

    if GE_AVAILABLE:
        try:
            violations = _ge_validate(df, METADATA_EXPECTATIONS, "metadata", html_report_dir)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
            logger.warning("Great Expectations failed (%s); falling back to pandas validator.", exc)
            violations = _fallback_validate_metadata(df)
    else:
        violations = _fallback_validate_metadata(df)

    if violations:
        detail = _format_violations(violations)
        if raise_on_error:
            raise ValidationError(detail)
        logger.error("Metadata validation issues:\n%s", detail)
    else:
        logger.info("Metadata validation PASSED.")

    return violations
