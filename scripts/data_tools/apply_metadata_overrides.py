"""Create a new immutable bundle after verified metadata overrides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.bundle import BundleError
from src.data.loader import download_amazon_electronics, load_raw_data
from src.data.pipeline import run_metadata_only
from src.data.validation import validate_raw_metadata

ROOT = Path(__file__).resolve().parents[2]


def main(
    bundle: str | None = None,
    overrides_path: str | Path | None = None,
    no_publish: bool = False,
):
    """Rebuild metadata/text/reports without editing the active bundle."""
    raw_dataset = Path(download_amazon_electronics(ROOT / "data" / "raw"))
    _, items = load_raw_data(str(raw_dataset))
    validate_raw_metadata(items, raise_on_error=True)
    result = run_metadata_only(
        bundle=bundle,
        raw_items=items,
        overrides_path=overrides_path or ROOT / "data" / "metadata_overrides.csv",
        publish=not no_publish,
    )
    payload = {"bundle": str(result), "published": not no_publish}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="Base bundle path or data/current.json")
    parser.add_argument("--overrides_path", help="Verified override CSV")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    try:
        main(args.bundle, args.overrides_path, args.no_publish)
    except BundleError as exc:
        parser.error(str(exc))
