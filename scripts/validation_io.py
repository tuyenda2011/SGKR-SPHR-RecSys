"""Load the pinned train/validation protocol without opening held-out test data."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bundle import BundleError, _inventory_path, resolve_bundle, sha256_file


def load_validation_inputs(bundle_arg, root: Path):
    # Default resolve_bundle hashes every artifact, including test and metadata.
    bundle = resolve_bundle(bundle_arg, data_root=root / "data", check_hashes=False)
    paths = [bundle.split_path("train"), bundle.split_path("val"), bundle.artifact("mappings.pkl")]
    inventory = bundle.manifest.get("artifacts", [])
    for path in paths:
        matches = [entry for entry in inventory if _inventory_path(bundle, entry["path"]) == path.resolve()]
        if len(matches) != 1 or sha256_file(path) != matches[0].get("sha256"):
            raise BundleError(f"Missing or mismatched validation-input checksum: {path.name}")
    train, validation = (pd.read_parquet(path) for path in paths[:2])
    with paths[2].open("rb") as stream:
        # Trusted local bundle; checksum verified above.
        stats = pickle.load(stream)["stats"]  # nosec B301
    return bundle, train, validation, stats


def edge_hash(frame):
    edges = frame[["u_idx", "i_idx"]].sort_values(["u_idx", "i_idx"]).to_numpy(dtype=np.int64)
    return hashlib.sha256(edges.tobytes()).hexdigest()
