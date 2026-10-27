"""Shared dataset build orchestration.

The existing preprocessing functions remain the source of truth for cleaning,
splitting and feature construction.  This module owns only the lifecycle
around them: stable input identity, bundle layout, metadata-only rebuilds and
publication.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.adaptive_metadata import ADAPTIVE_TEXT_POLICY
from src.data.bundle import (
    DATA_ROOT,
    BundleError,
    DatasetBundle,
    build_manifest_payload,
    find_existing_bundle,
    make_build_id,
    publish_bundle,
    refresh_manifest_inventory,
    resolve_bundle,
    rollback_bundle,
    sha256_file,
    stable_fingerprint,
    write_manifest,
)
from src.data.export import export_bundle_csvs
from src.data.metadata_resolution import resolve_metadata, write_brand_review_queue
from src.data.preprocessing import CATEGORY_POLICY, summarize_metadata_quality
from src.data.provenance import sha256_file as source_sha256_file
from src.data.text_encoder import (
    PINNED_REVISION,
    build_adaptivegcl_text_view,
    encode_item_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_TMP = REPO_ROOT / ".tmp" / "data-pipeline"


def input_fingerprint(
    *,
    source_files: list[Path],
    config: dict[str, Any],
    overrides_path: Path,
    encoder: str,
    revision: str | None,
    policy_version: str,
    build_options: dict[str, Any] | None = None,
) -> str:
    """Return an identity independent of staging paths and timestamps."""
    # ``prepare_data`` temporarily points ``dataset.processed_dir`` at a
    # staging directory.  Output locations and pointer settings describe the
    # build environment, not its data identity, so remove them before hashing.
    dataset_config = deepcopy(config.get("dataset", {}))
    for key in ("processed_dir", "bundle_pointer", "metadata_overrides_path"):
        dataset_config.pop(key, None)
    config_identity = deepcopy(config)
    if isinstance(config_identity.get("dataset"), dict):
        config_identity["dataset"] = dataset_config
    return stable_fingerprint(
        {
            "source_files": [
                {"name": path.name, "sha256": source_sha256_file(path)}
                for path in sorted(source_files)
            ],
            "overrides_sha256": source_sha256_file(overrides_path)
            if overrides_path.is_file()
            else None,
            "dataset_config": config_identity,
            "encoder": encoder,
            "revision": revision,
            "category_policy": CATEGORY_POLICY,
            "adaptive_text_policy": ADAPTIVE_TEXT_POLICY,
            "metadata_policy": policy_version,
            "build_options": build_options or {},
        }
    )


def _copy_or_move(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def _legacy_manifest_metadata(legacy_root: Path, *, build_id: str, fingerprint: str) -> dict[str, Any]:
    old = {}
    manifest_path = legacy_root / "manifest.json"
    if manifest_path.is_file():
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                old = value
        except json.JSONDecodeError:
            pass
    metadata = {
        "created_at": old.get("created_at"),
        "dataset": old.get("dataset", "amazon-electronics"),
        "source_files": old.get("source_files", []),
        "preprocessing": old.get("preprocessing", {}),
        "text_features": old.get("text_features", {}),
        "statistics": old.get("statistics", {}),
        "input_fingerprint": fingerprint,
        "build_id": build_id,
    }
    return metadata


def _source_entries(source_files: list[Path]) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(Path(item).resolve() for item in source_files):
        try:
            display_path = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            display_path = path.name
        entries.append(
            {
                "path": display_path,
                "sha256": source_sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def _ordered_id_fingerprint(mappings: dict[str, Any]) -> str:
    users = sorted(mappings.get("user2id", {}).items(), key=lambda item: item[1])
    items = sorted(mappings.get("item2id", {}).items(), key=lambda item: item[1])
    return stable_fingerprint(
        {
            "users": [key for key, _ in users],
            "items": [key for key, _ in items],
        }
    )


def _manifest_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep manifest config portable and free of temporary staging paths."""
    result = deepcopy(config)
    dataset = result.get("dataset")
    if isinstance(dataset, dict):
        dataset["processed_dir"] = "train"
        dataset["bundle_pointer"] = "data/current.json"
        dataset["metadata_overrides_path"] = "data/metadata_overrides.csv"
    return result


def _metadata_fingerprint(
    base: DatasetBundle,
    *,
    raw_items: pd.DataFrame,
    overrides_path: Path,
    encoder: str,
    revision: str | None,
) -> str:
    """Resolve metadata in memory to identify a reusable metadata-only build."""
    with base.artifact("mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    item_metadata = deepcopy(mappings["item_metadata"])
    resolve_metadata(item_metadata, raw_items, overrides_path)
    return stable_fingerprint(
        {
            "base": base.input_fingerprint or base.build_id,
            "metadata": item_metadata,
            "overrides": sha256_file(overrides_path) if overrides_path.exists() else None,
            "encoder": encoder,
            "revision": revision,
            "category_policy": CATEGORY_POLICY,
            "adaptive_text_policy": ADAPTIVE_TEXT_POLICY,
        }
    )


def package_legacy_staging(
    legacy_root: str | Path,
    *,
    source_files: list[Path],
    config: dict[str, Any],
    overrides_path: Path,
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
    revision: str | None = PINNED_REVISION,
    policy_version: str = "metadata_policy_v2",
    max_csv_rows: int = 500_000,
    inspection_rows: int = 1_000,
    seed: int = 42,
    export_csv: bool = True,
    build_options: dict[str, Any] | None = None,
) -> Path:
    """Convert a validated legacy staging directory into a bundle layout."""
    legacy_root = Path(legacy_root).resolve()
    fingerprint = input_fingerprint(
        source_files=source_files,
        config=config,
        overrides_path=overrides_path,
        encoder=encoder,
        revision=revision,
        policy_version=policy_version,
        build_options=build_options,
    )
    build_id = make_build_id(fingerprint)
    train_dir = legacy_root / "train"
    csv_dir = legacy_root / "csv"
    reports_dir = legacy_root / "reports"
    train_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    for name in (
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "mappings.pkl",
        "disliked_interactions.parquet",
        "item_text_embeddings.pt",
        "item_text_embeddings.pt.json",
    ):
        source = legacy_root / name
        if source.exists():
            _copy_or_move(source, train_dir / name)

    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    mappings_path = train_dir / "mappings.pkl"
    with mappings_path.open("rb") as stream:
        mappings = pickle.load(stream)
    train = pd.read_parquet(train_dir / "train.parquet")
    val = pd.read_parquet(train_dir / "val.parquet")
    test = pd.read_parquet(train_dir / "test.parquet")
    disliked = None
    disliked_path = train_dir / "disliked_interactions.parquet"
    if disliked_path.exists():
        disliked = pd.read_parquet(disliked_path)

    report = {}
    quality_path = legacy_root / "data_quality_report.json"
    if quality_path.exists():
        try:
            report = json.loads(quality_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    # Move legacy reports before generating the standard export so an old
    # queue cannot overwrite the newly generated, bundle-relative copy.
    for name in (
        "audit_report.json",
        "data_quality_report.json",
        "data_quality_summary.csv",
        "brand_review_queue.csv",
        "brand_review_sample.csv",
        "metadata_issues.csv",
    ):
        source = legacy_root / name
        if source.exists():
            _copy_or_move(source, reports_dir / name)
    if export_csv:
        export_bundle_csvs(
            bundle_root=legacy_root,
            train=train,
            val=val,
            test=test,
            mappings=mappings,
            disliked=disliked,
            report=report,
            build_id=build_id,
            max_rows=max_csv_rows,
            inspection_rows=inspection_rows,
            seed=seed,
        )
    else:
        (legacy_root / "csv").mkdir(exist_ok=True)

    old_manifest = _legacy_manifest_metadata(
        legacy_root, build_id=build_id, fingerprint=fingerprint
    )
    old_manifest["source_files"] = _source_entries(source_files)
    old_manifest["config"] = _manifest_config(config)
    old_manifest["encoder"] = {"model_name": encoder, "revision": revision}
    old_manifest["metadata_policy"] = policy_version
    old_manifest["build_options"] = build_options or {}
    old_manifest["ordered_id_fingerprint"] = _ordered_id_fingerprint(mappings)
    old_manifest["optional_artifacts"] = {
        "disliked_interactions.parquet": "present" if disliked_path.exists() else "disabled",
        "csv": "present" if export_csv else "disabled",
    }
    statistics = old_manifest.get("statistics")
    if isinstance(statistics, dict):
        queue_stats = statistics.get("brand_review_queue")
        if isinstance(queue_stats, dict):
            if queue_stats.get("path"):
                queue_stats["path"] = "reports/brand_review_queue.csv"
            if queue_stats.get("sample_path"):
                queue_stats["sample_path"] = "reports/brand_review_sample.csv"
    old_manifest["artifacts"] = []
    payload = build_manifest_payload(
        build_id=build_id,
        input_fingerprint=fingerprint,
        metadata=old_manifest,
        bundle_root=legacy_root,
    )
    write_manifest(legacy_root / "manifest.json", payload)
    from scripts.audit_data import main as audit_dataset

    if audit_dataset(bundle=legacy_root):
        raise BundleError(f"Bundle audit failed before publish: {legacy_root}")
    refresh_manifest_inventory(legacy_root)
    return legacy_root


def publish_legacy_staging(
    staging: str | Path,
    *,
    source_files: list[Path],
    config: dict[str, Any],
    overrides_path: Path,
    publish: bool = True,
    data_root: str | Path = DATA_ROOT,
    build_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DatasetBundle | Path:
    """Package a prepare staging directory and optionally activate it."""
    bundle_root = package_legacy_staging(
        staging,
        source_files=source_files,
        config=config,
        overrides_path=overrides_path,
        build_options=build_options,
        **kwargs,
    )
    if not publish:
        return bundle_root
    return publish_bundle(bundle_root, data_root=data_root, activate=True)


def find_existing_full_bundle(
    *,
    source_files: list[Path],
    config: dict[str, Any],
    overrides_path: Path,
    encoder: str,
    revision: str | None,
    policy_version: str,
    build_options: dict[str, Any] | None = None,
    data_root: str | Path = DATA_ROOT,
) -> DatasetBundle | None:
    fingerprint = input_fingerprint(
        source_files=source_files,
        config=config,
        overrides_path=overrides_path,
        encoder=encoder,
        revision=revision,
        policy_version=policy_version,
        build_options=build_options,
    )
    return find_existing_bundle(fingerprint, data_root=data_root)


def _metadata_only_stage(
    base: DatasetBundle,
    *,
    raw_items: pd.DataFrame,
    overrides_path: Path,
    encoder: str,
    revision: str | None,
) -> tuple[Path, dict[str, Any]]:
    """Rebuild only metadata-dependent artifacts from a pinned base bundle."""
    staging_root = PIPELINE_TMP / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="metadata-only-", dir=staging_root))
    (stage / "train").mkdir(parents=True)
    (stage / "csv").mkdir()
    (stage / "reports").mkdir()
    for name in ("train.parquet", "val.parquet", "test.parquet", "disliked_interactions.parquet"):
        source = base.artifact(name, required=name != "disliked_interactions.parquet")
        if source.exists():
            shutil.copy2(source, stage / "train" / name)
    with base.artifact("mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    before = summarize_metadata_quality(mappings["item_metadata"])
    resolution = resolve_metadata(mappings["item_metadata"], raw_items, overrides_path)
    mappings["stats"]["metadata_resolution"] = resolution
    mappings["stats"]["metadata_quality"] = summarize_metadata_quality(mappings["item_metadata"])
    with (stage / "train" / "mappings.pkl").open("wb") as stream:
        pickle.dump(mappings, stream)
    encode_item_metadata(
        mappings["item_metadata"],
        num_items=len(mappings["item2id"]),
        model_name=encoder,
        revision=revision,
        save_path=str(stage / "train" / "item_text_embeddings.pt"),
        batch_size=64,
    )
    text_provenance = json.loads(
        (stage / "train" / "item_text_embeddings.pt.json").read_text(encoding="utf-8")
    )
    build_adaptivegcl_text_view(
        mappings["item_metadata"],
        len(mappings["item2id"]),
        model_name=encoder,
        revision=revision,
        save_path=str(stage / "train" / "adaptivegcl_text_embeddings.pt"),
        batch_size=64,
    )
    quality_provenance = json.loads(
        (stage / "train" / "adaptivegcl_text_embeddings.pt.json").read_text(encoding="utf-8")
    )
    fingerprint = stable_fingerprint(
        {
            "base": base.input_fingerprint or base.build_id,
            "metadata": mappings["item_metadata"],
            "overrides": sha256_file(overrides_path) if overrides_path.exists() else None,
            "encoder": encoder,
            "revision": revision,
            "category_policy": CATEGORY_POLICY,
            "adaptive_text_policy": ADAPTIVE_TEXT_POLICY,
        }
    )
    train = pd.read_parquet(stage / "train" / "train.parquet")
    val = pd.read_parquet(stage / "train" / "val.parquet")
    test = pd.read_parquet(stage / "train" / "test.parquet")
    disliked = None
    if (stage / "train" / "disliked_interactions.parquet").exists():
        disliked = pd.read_parquet(stage / "train" / "disliked_interactions.parquet")
    quality = {
        "metadata_quality_before": before,
        "metadata_quality_after": mappings["stats"]["metadata_quality"],
        "metadata_resolution": resolution,
        "base_build_id": base.build_id,
        "ordered_id_fingerprint": base.manifest.get("ordered_id_fingerprint"),
    }
    build_id = make_build_id(fingerprint)
    queue_stats = write_brand_review_queue(
        mappings["item_metadata"], raw_items,
        train["i_idx"].value_counts().to_dict(),
        stage / "reports" / "brand_review_queue.csv",
        sample_path=stage / "reports" / "brand_review_sample.csv",
    )
    queue_stats["path"] = "reports/brand_review_queue.csv"
    queue_stats["sample_path"] = "reports/brand_review_sample.csv"
    quality["brand_review_queue"] = queue_stats
    export_bundle_csvs(
        bundle_root=stage,
        train=train,
        val=val,
        test=test,
        mappings=mappings,
        disliked=disliked,
        report=quality,
        build_id=build_id,
        seed=42,
    )
    (stage / "reports" / "data_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    statistics = deepcopy(base.manifest.get("statistics", {}))
    if isinstance(statistics, dict):
        queue_stats = statistics.get("brand_review_queue")
        if isinstance(queue_stats, dict):
            if queue_stats.get("path"):
                queue_stats["path"] = "reports/brand_review_queue.csv"
            if queue_stats.get("sample_path"):
                queue_stats["sample_path"] = "reports/brand_review_sample.csv"
        statistics.update(quality)
    metadata = {
        "dataset": base.manifest.get("dataset", "amazon-electronics"),
        "source_files": base.manifest.get("source_files", []),
        "preprocessing": base.manifest.get("preprocessing", {}),
        "statistics": statistics,
        "text_features": {
            **base.manifest.get("text_features", {}),
            **text_provenance,
        },
        "config": _manifest_config(base.manifest.get("config", {})),
        "encoder": {"model_name": encoder, "revision": revision},
        "metadata_only": True,
        "category_policy": CATEGORY_POLICY,
        "feature_views": {
            "adaptivegcl_quality": {
                key: quality_provenance[key]
                for key in ("metadata_policy", "input_fingerprint", "sha256", "shape")
            }
        },
        "base_build_id": base.build_id,
        "ordered_id_fingerprint": base.manifest.get("ordered_id_fingerprint"),
        "optional_artifacts": {
            "disliked_interactions.parquet": "present" if disliked is not None else "disabled",
            "csv": "present",
        },
    }
    write_manifest(
        stage / "manifest.json",
        build_manifest_payload(
            build_id=build_id,
            input_fingerprint=fingerprint,
            metadata=metadata,
            bundle_root=stage,
        ),
    )
    from scripts.audit_data import main as audit_dataset

    if audit_dataset(bundle=stage):
        raise BundleError(f"Metadata-only bundle audit failed: {stage}")
    refresh_manifest_inventory(stage)
    return stage, mappings


def run_metadata_only(
    *,
    bundle: str | Path | None = None,
    raw_items: pd.DataFrame,
    overrides_path: str | Path = REPO_ROOT / "data" / "metadata_overrides.csv",
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
    revision: str | None = PINNED_REVISION,
    publish: bool = True,
    data_root: str | Path = DATA_ROOT,
) -> DatasetBundle | Path:
    base = resolve_bundle(bundle, data_root=data_root)
    identity = _metadata_fingerprint(
        base,
        raw_items=raw_items,
        overrides_path=Path(overrides_path),
        encoder=encoder,
        revision=revision,
    )
    existing = find_existing_bundle(identity, data_root=data_root)
    if existing is not None:
        if publish:
            return rollback_bundle(existing.root, data_root=data_root)
        return existing
    stage, _ = _metadata_only_stage(
        base,
        raw_items=raw_items,
        overrides_path=Path(overrides_path),
        encoder=encoder,
        revision=revision,
    )
    if not publish:
        return stage
    return publish_bundle(stage, data_root=data_root, activate=True)
