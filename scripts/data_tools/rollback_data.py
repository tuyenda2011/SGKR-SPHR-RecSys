"""Activate a previously published, verified dataset bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.bundle import BundleError, rollback_bundle


def main(target: str, data_root: str | Path = "data") -> int:
    try:
        bundle = rollback_bundle(target, data_root=data_root)
    except BundleError as exc:
        print(f"ROLLBACK FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "PASS",
        "build_id": bundle.build_id,
        "bundle": str(bundle.root),
        "pointer": str(Path(data_root) / "current.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="Verified build directory or build id below data/versions",
    )
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    raise SystemExit(main(args.target, args.data_root))
