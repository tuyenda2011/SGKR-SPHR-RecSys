"""Backward-compatible wrapper for the unified metadata-only pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_tools.apply_metadata_overrides import main as apply_metadata_overrides


def main(
    bundle: str | None = None,
    overrides_path: str | Path | None = None,
    no_publish: bool = False,
):
    """Preserve the old command name while publishing a new bundle."""
    return apply_metadata_overrides(bundle, overrides_path, no_publish)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="Base bundle path or data/current.json")
    parser.add_argument("--overrides_path", help="Verified override CSV")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    main(args.bundle, args.overrides_path, args.no_publish)
