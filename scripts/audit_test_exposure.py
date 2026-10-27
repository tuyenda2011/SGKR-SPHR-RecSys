"""Inventory saved test scores without loading datasets or reporting test values."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"source_snapshot", "workflow_snapshot", "workflow_amendments", "report_snapshot", "configs"}
MAX_BYTES = 16 * 2**20


def inventory(directory: Path, active_bundle: str):
    records, issues = [], []
    manifests = {}
    for manifest in (ROOT / "data/versions").glob("*/manifest.json"):
        content = json.loads(manifest.read_text(encoding="utf-8"))
        manifests[manifest.parent.name] = {a["path"]: a["sha256"] for a in content["artifacts"]}
    active_test_hash = manifests.get(active_bundle, {}).get("train/test.parquet")
    paths = sorted(directory.rglob("*.json"))
    if len(paths) > 20_000:
        raise ValueError("Inventory exceeds the declared 20,000-file bound")
    for path in paths:
        if EXCLUDED.intersection(path.relative_to(directory).parts):
            continue
        if path.stat().st_size > MAX_BYTES:
            issues.append({"path": str(path), "reason": "above 16 MiB bound"})
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            issues.append({"path": str(path), "reason": type(error).__name__})
            continue
        if not isinstance(result, dict) or not isinstance(result.get("val_metrics"), dict) or not result.get("model_name"):
            continue
        config = result.get("effective_config", result.get("config", {}))
        dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
        bundle = dataset.get("bundle_id") or result.get("bundle_id")
        saved_test = result.get("test_metrics")
        test_present = isinstance(saved_test, dict) and bool(saved_test)
        test_hash = manifests.get(bundle, {}).get("train/test.parquet")
        records.append({
            "path": path.relative_to(ROOT).as_posix(), "model": result["model_name"],
            "seed": result.get("seed"), "density": result.get("sparsity_level"),
            "bundle_id": bundle, "matches_active_bundle": bundle == active_bundle if bundle else None,
            "test_artifact_sha256_from_manifest": test_hash,
            "matches_active_test_artifact": test_hash == active_test_hash if test_hash and active_test_hash else None,
            "validation_only_flag": result.get("validation_only"), "saved_test_scores_present": test_present,
            "contradictory_mode": test_present and result.get("validation_only") is True,
            "test_metric_names": ";".join(sorted(saved_test)) if test_present else "",
            "fingerprint": result.get("experiment_fingerprint"),
        })
    return records, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace this inventory's reports only")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.relative_to(ROOT.resolve())
    names = ("test_exposure.csv", "test_exposure.json")
    if not args.force and any((output / name).exists() for name in names):
        raise FileExistsError("Inventory exists; choose another directory or use --force")
    pointer = json.loads((ROOT / "data/current.json").read_text(encoding="utf-8"))
    active_bundle = pointer.get("build_id") or pointer.get("bundle_id")
    if not active_bundle:
        raise ValueError("Active bundle identifier is absent")
    records, issues = inventory(ROOT / "results", active_bundle)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else ["path", "model", "saved_test_scores_present"]
    with (output / names[0]).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    scored = [r for r in records if r["saved_test_scores_present"]]
    same_bundle = [r for r in scored if r["matches_active_bundle"] is True]
    same_test = [r for r in scored if r["matches_active_test_artifact"] is True]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(), "active_bundle": active_bundle,
        "result_files": len(records), "saved_test_score_files": len(scored),
        "same_active_bundle_test_score_files": len(same_bundle),
        "same_active_test_artifact_score_files": len(same_test),
        "unknown_bundle_test_score_files": sum(r["bundle_id"] is None for r in scored),
        "unknown_test_artifact_score_files": sum(r["test_artifact_sha256_from_manifest"] is None for r in scored),
        "mode_contradictions": sum(r["contradictory_mode"] for r in records),
        "scored_models": dict(Counter(r["model"] for r in scored)), "unreadable_or_oversize_files": issues,
        "test_labels_loaded": False, "test_metric_values_reported": False,
        "untouched_holdout_established": False, "test_used_for_tuning_established": False,
        "interpretation": "Saved scores establish exposure, not whether they informed tuning. Absence of saved scores cannot prove absence of prior access.",
        "scope": "Local saved JSON results and declared manifest hashes (bundle IDs alone cannot establish split independence); excludes source/config snapshots; does not inspect deleted files, terminal history, or human decisions.",
    }
    (output / names[1]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "unreadable_or_oversize_files"}, indent=2))
    print(f"Inventory issues: {len(issues)}")


if __name__ == "__main__":
    main()
