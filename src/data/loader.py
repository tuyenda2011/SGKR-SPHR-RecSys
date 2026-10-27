import ast
import gzip
import hashlib
import json
import logging
import os
import urllib.request
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_REVIEWS_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
DEFAULT_META_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_raw_snapshot(raw_root: str | os.PathLike[str]) -> str | None:
    """Return the verified immutable raw snapshot selected by current.json.

    A present but invalid pointer is an error.  The legacy
    ``raw/amazon-electronics`` directory is used only when no pointer exists,
    which keeps old local checkouts usable during the migration.
    """
    root = os.path.abspath(os.fspath(raw_root))
    pointer_path = os.path.join(root, "current.json")
    if not os.path.exists(pointer_path):
        return None
    try:
        with open(pointer_path, encoding="utf-8") as stream:
            pointer = json.load(stream)
        if pointer.get("schema_version") != 1:
            raise ValueError("unsupported raw pointer schema")
        relative = pointer["snapshot_path"]
        if os.path.isabs(relative):
            raise ValueError("snapshot_path must be relative")
        snapshot = os.path.realpath(os.path.join(root, relative))
        if os.path.commonpath([root, snapshot]) != root:
            raise ValueError("snapshot_path escapes raw directory")
        manifest_path = os.path.join(snapshot, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise ValueError("raw snapshot manifest is missing")
        if pointer.get("manifest_sha256") != _sha256_file(manifest_path):
            raise ValueError("raw snapshot manifest checksum does not match current.json")
        required = {
            "reviews_Electronics_5.json.gz",
            "meta_Electronics.json.gz",
        }
        if not required.issubset(
            name for name in os.listdir(snapshot)
            if os.path.isfile(os.path.join(snapshot, name))
        ):
            raise ValueError("raw snapshot is missing one or more required files")
        return snapshot
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid raw snapshot pointer: {pointer_path}") from exc


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_amazon_electronics(
    raw_dir: str = "data/raw",
    reviews_url: str = DEFAULT_REVIEWS_URL,
    meta_url: str = DEFAULT_META_URL,
) -> str:
    """Download Amazon Electronics reviews and metadata with visual download progress bars."""
    snapshot = _active_raw_snapshot(raw_dir)
    if snapshot is not None:
        logger.info("Using immutable raw snapshot at %s", snapshot)
        return snapshot
    os.makedirs(raw_dir, exist_ok=True)
    dataset_dir = os.path.join(raw_dir, "amazon-electronics")
    os.makedirs(dataset_dir, exist_ok=True)

    reviews_file = os.path.join(dataset_dir, "reviews_Electronics_5.json.gz")
    meta_file = os.path.join(dataset_dir, "meta_Electronics.json.gz")

    if os.path.exists(reviews_file) and os.path.exists(meta_file):
        logger.info(f"Amazon Electronics dataset already exists at {dataset_dir}")
        return dataset_dir

    if not os.path.exists(reviews_file):
        logger.info(f"Downloading Reviews from {reviews_url}...")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc="Downloading Reviews"
        ) as t:
            urllib.request.urlretrieve(
                reviews_url, filename=reviews_file, reporthook=t.update_to
            )
        logger.info("Reviews download completed.")

    if not os.path.exists(meta_file):
        logger.info(f"Downloading Meta from {meta_url}...")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc="Downloading Metadata"
        ) as t:
            urllib.request.urlretrieve(
                meta_url, filename=meta_file, reporthook=t.update_to
            )
        logger.info("Meta download completed.")

    return dataset_dir


def get_df_from_json_gz(path: str, desc: str | None = None) -> pd.DataFrame:
    """Load json.gz into Pandas DataFrame with live tqdm progress bar."""
    file_name = os.path.basename(path)
    pbar_desc = desc if desc is not None else f"Parsing {file_name}"
    logger.info(f"Parsing JSON GZ file: {path}")

    data = []
    ledger: dict[str, Any] = {
        "total_lines": 0,
        "blank_lines": 0,
        "parsed_rows": 0,
        "invalid_lines": 0,
        "error_examples": [],
    }
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=pbar_desc, unit=" lines", dynamic_ncols=True):
            ledger["total_lines"] += 1
            line_str = line.strip()
            if not line_str:
                ledger["blank_lines"] += 1
                continue
            try:
                record = json.loads(line_str)
            except (ValueError, TypeError):
                try:
                    record = ast.literal_eval(line_str)
                except (ValueError, SyntaxError, TypeError):
                    record = None
            if not isinstance(record, dict):
                ledger["invalid_lines"] += 1
                if len(ledger["error_examples"]) < 20:
                    ledger["error_examples"].append(
                        {
                            "line": ledger["total_lines"],
                            "reason": "not a parseable object",
                        }
                    )
                continue
            record["raw_row_id"] = ledger["total_lines"] - 1
            data.append(record)
            ledger["parsed_rows"] += 1
    if ledger["invalid_lines"]:
        logger.warning(
            "%s: %s invalid lines; examples (no review text): %s",
            file_name,
            ledger["invalid_lines"],
            ledger["error_examples"],
        )
    frame = pd.DataFrame.from_dict(data)
    frame.attrs["ingestion_ledger"] = ledger
    return frame


def load_raw_data(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Amazon Electronics reviews and metadata into DataFrames with progress bars."""
    reviews_path = os.path.join(dataset_dir, "reviews_Electronics_5.json.gz")
    meta_path = os.path.join(dataset_dir, "meta_Electronics.json.gz")

    logger.info("Loading reviews into DataFrame...")
    ratings_df = get_df_from_json_gz(reviews_path, desc="Loading Reviews (1.68M)")

    logger.info("Loading metadata into DataFrame...")
    items_df = get_df_from_json_gz(meta_path, desc="Loading Metadata (498K)")

    return ratings_df, items_df


def resolve_dataset_bundle(bundle: str | os.PathLike[str] | None = None):
    """Resolve the active processed view, pinned to one verified bundle."""
    from src.data.bundle import resolve_bundle

    return resolve_bundle(bundle)


def load_processed_bundle(bundle: str | os.PathLike[str] | None = None):
    """Load all processed artifacts from one pinned bundle."""
    resolved = resolve_dataset_bundle(bundle)
    return resolved, resolved.load()
