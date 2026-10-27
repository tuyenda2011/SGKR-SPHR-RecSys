"""Immutable dataset bundles and the active bundle pointer.

The training artifacts remain Parquet/pickle/tensor files, but their location
is pinned by a small manifest.  Readers resolve one bundle once and keep that
root for the lifetime of a run, so a later publication cannot mix versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CURRENT_POINTER = DATA_ROOT / "current.json"
BUNDLE_SCHEMA_VERSION = 1
REQUIRED_TRAIN_ARTIFACTS = (
    "train.parquet",
    "val.parquet",
    "test.parquet",
    "mappings.pkl",
)


class BundleError(RuntimeError):
    """Raised when a dataset bundle or pointer is not safe to consume."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(payload: Any) -> str:
    """Hash JSON-compatible input without paths or timestamps."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def make_build_id(input_fingerprint: str, *, now: datetime | None = None) -> str:
    """Create a filesystem-safe publication id.

    The identity is derived only from the input fingerprint.  Re-running an
    unchanged build therefore resolves to the same bundle instead of creating
    another timestamp-only copy.  ``now`` remains accepted for API stability
    with early callers but is intentionally not part of the ID.
    """
    del now
    return f"build_{input_fingerprint[:32]}"


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_details(path: Path, root: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise BundleError(f"Missing bundle artifact: {path}")
    detail: dict[str, Any] = {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "required": required,
    }
    if path.suffix == ".parquet":
        try:
            from pyarrow import parquet

            detail["rows"] = int(parquet.ParquetFile(path).metadata.num_rows)
        except (ImportError, OSError, ValueError):
            detail["rows"] = len(pd.read_parquet(path))
    elif path.suffix == ".csv":
        detail["rows"] = max(0, sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1)
    elif path.suffix == ".pkl":
        try:
            with path.open("rb") as stream:
                value = pickle.load(stream)
            if isinstance(value, dict):
                detail["keys"] = sorted(value.keys())
            del value
        except (OSError, pickle.PickleError, EOFError, ValueError, TypeError):
            pass
    elif path.suffix in {".pt", ".pth"}:
        try:
            import torch

            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, torch.Tensor):
                detail["shape"] = list(value.shape)
            del value
        except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.PickleError):
            pass
    return detail


def inventory_artifacts(root: str | Path) -> list[dict[str, Any]]:
    """Inventory files below the three bundle directories."""
    root = Path(root).resolve()
    entries: list[dict[str, Any]] = []
    for directory in ("train", "csv", "reports"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            entries.append(
                _artifact_details(
                    path,
                    root,
                    required=directory == "train" and path.name in REQUIRED_TRAIN_ARTIFACTS,
                )
            )
    return entries


@dataclass(frozen=True)
class DatasetBundle:
    """A resolved, validated dataset root."""

    root: Path
    manifest: dict[str, Any]
    legacy: bool = False

    @property
    def build_id(self) -> str:
        return str(self.manifest.get("build_id", "legacy"))

    @property
    def input_fingerprint(self) -> str | None:
        value = self.manifest.get("input_fingerprint")
        return str(value) if value else None

    @property
    def manifest_path(self) -> Path:
        if not self.legacy:
            return self.root / "manifest.json"
        local_manifest = self.root / "manifest.json"
        return local_manifest if local_manifest.is_file() else self.root.parent / "manifest.json"

    @property
    def train_dir(self) -> Path:
        return self.root / "train" if not self.legacy else self.root

    @property
    def csv_dir(self) -> Path:
        return self.root / "csv"

    @property
    def reports_dir(self) -> Path:
        nested = self.root / "reports"
        return nested if not self.legacy or nested.is_dir() else self.root

    def artifact(self, name: str, *, required: bool = True) -> Path:
        """Resolve an artifact by its stable filename."""
        candidate = self.train_dir / name
        if candidate.exists():
            return candidate
        candidate = self.csv_dir / name
        if candidate.exists():
            return candidate
        candidate = self.reports_dir / name
        if candidate.exists():
            return candidate
        if self.legacy:
            legacy = self.root / name
            if legacy.exists():
                return legacy
        if required:
            raise BundleError(f"Artifact {name!r} is missing from bundle {self.root}")
        return candidate

    def split_path(self, split: str) -> Path:
        filename = {"train": "train.parquet", "validation": "val.parquet", "val": "val.parquet", "test": "test.parquet"}.get(split, split)
        return self.train_dir / filename

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        train = pd.read_parquet(self.split_path("train"))
        val = pd.read_parquet(self.split_path("val"))
        test = pd.read_parquet(self.split_path("test"))
        with self.artifact("mappings.pkl").open("rb") as stream:
            mappings = pickle.load(stream)
        return train, val, test, mappings


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Cannot read JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"Expected a JSON object: {path}")
    return value


def _inventory_path(bundle: DatasetBundle, relative: str) -> Path:
    """Map archived and flattened processed inventories into the pinned root."""
    relative = relative.replace("\\", "/")
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise BundleError(f"Unsafe manifest artifact path: {relative}")
    if bundle.legacy:
        if relative.startswith("data/processed/"):
            relative = relative[len("data/processed/"):]
        elif relative.startswith("train/"):
            relative = relative[len("train/"):]
    root = bundle.root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise BundleError(f"Manifest artifact path is missing or escapes bundle: {relative}")
    return path


def verify_bundle(
    bundle: DatasetBundle,
    *,
    check_hashes: bool = True,
    require_build_dir: bool = True,
) -> None:
    """Verify layout, schema and manifest artifact checksums."""
    if "schema_version" in bundle.manifest and bundle.manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"Unsupported bundle schema: {bundle.manifest['schema_version']}")
    if bundle.legacy:
        required = [bundle.root / name for name in REQUIRED_TRAIN_ARTIFACTS]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise BundleError(f"Legacy processed data is incomplete: {missing}")
    elif bundle.manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(
            f"Unsupported bundle schema: {bundle.manifest.get('schema_version')}"
        )
    if not bundle.legacy and require_build_dir and bundle.manifest.get("build_id") != bundle.root.name:
        raise BundleError("Bundle build_id does not match its directory name")
    if not bundle.legacy:
        for directory in ("train", "csv", "reports"):
            if not (bundle.root / directory).is_dir():
                raise BundleError(f"Bundle directory is missing: {directory}")
    for name in REQUIRED_TRAIN_ARTIFACTS:
        if not (bundle.train_dir / name).is_file():
            raise BundleError(f"Required train artifact is missing: {name}")
    if not check_hashes:
        return
    inventory = bundle.manifest.get("artifacts")
    if inventory is None and bundle.legacy and "schema_version" not in bundle.manifest:
        warnings.warn("Legacy manifest has no inventory; artifact integrity is unverified", RuntimeWarning, stacklevel=2)
        return
    if not isinstance(inventory, list) or not inventory:
        raise BundleError("Manifest has no artifact inventory")
    entries = {}
    for entry in inventory:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise BundleError("Invalid manifest inventory entry")
        path = _inventory_path(bundle, entry["path"])
        if path in entries:
            raise BundleError(f"Duplicate manifest artifact: {entry['path']}")
        entries[path] = entry
    for name in REQUIRED_TRAIN_ARTIFACTS:
        if (bundle.train_dir / name).resolve() not in entries:
            raise BundleError(f"Manifest has no inventory entry for {name}")
    for path, entry in entries.items():
        if sha256_file(path) != entry.get("sha256"):
            raise BundleError(f"Artifact checksum mismatch: {entry['path']}")


def _bundle_from_root(
    root: Path,
    *,
    legacy: bool = False,
    check_hashes: bool = True,
    require_build_dir: bool = True,
) -> DatasetBundle:
    if legacy:
        local_manifest = root / "manifest.json"
        manifest_path = local_manifest if local_manifest.is_file() else root.parent / "manifest.json"
    else:
        manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"Bundle manifest is missing: {manifest_path}")
    bundle = DatasetBundle(root=root.resolve(), manifest=_read_json(manifest_path), legacy=legacy)
    verify_bundle(
        bundle,
        check_hashes=check_hashes,
        require_build_dir=require_build_dir,
    )
    return bundle


def materialize_processed_view(
    bundle: DatasetBundle,
    *,
    data_root: str | Path = DATA_ROOT,
) -> Path:
    """Expose the active bundle through the simple legacy ``data/processed`` view.

    The immutable bundle remains the source of truth. This view is refreshed
    only after a verified publication/rollback so course scripts can keep the
    familiar ``data/processed/{train,val,test}.parquet`` layout.
    """
    if bundle.legacy:
        return bundle.root
    data = Path(data_root).resolve()
    view = data / "processed"
    temporary = data / f".processed-view-{bundle.build_id}"
    if temporary.exists():
        shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    for source in bundle.train_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, temporary / source.name)
    for directory in ("csv", "reports"):
        source = bundle.root / directory
        if source.is_dir():
            shutil.copytree(source, temporary / directory)

    manifest = deepcopy(bundle.manifest)
    for artifact in manifest.get("artifacts", []):
        relative = str(artifact.get("path", ""))
        if relative.startswith("train/"):
            artifact["path"] = f"data/processed/{relative[6:]}"
        elif relative.startswith(("csv/", "reports/")):
            artifact["path"] = f"data/processed/{relative}"
    queue_stats = manifest.get("statistics", {}).get("brand_review_queue")
    if isinstance(queue_stats, dict):
        for key in ("path", "sample_path"):
            if queue_stats.get(key) and not str(queue_stats[key]).startswith("data/"):
                queue_stats[key] = f"data/processed/{queue_stats[key]}"
    write_manifest(data / "manifest.json", manifest)
    write_manifest(temporary / "manifest.json", manifest)
    if view.exists():
        shutil.rmtree(view, ignore_errors=True)
    shutil.move(str(temporary), str(view))
    return view


def resolve_bundle(
    bundle: str | Path | None = None,
    *,
    data_root: str | Path = DATA_ROOT,
    check_hashes: bool = True,
    require_build_dir: bool = True,
) -> DatasetBundle:
    """Resolve an explicit bundle or the active pointer.

    If ``current.json`` exists but is invalid, an error is raised instead of
    silently falling back to legacy data. Legacy mode is only used when no
    pointer has ever been published.
    """
    root = Path(data_root).resolve()
    if bundle is not None:
        candidate = Path(bundle)
        if candidate.name == "current.json":
            pointer_path = candidate.resolve()
            pointer_root = pointer_path.parent
            pointer = _read_json(pointer_path)
            if pointer.get("schema_version") != BUNDLE_SCHEMA_VERSION:
                raise BundleError("Unsupported current.json schema")
            relative = pointer.get("bundle_path")
            if not relative or Path(str(relative)).is_absolute():
                raise BundleError("current.json has no bundle_path")
            candidate = (pointer_root / str(relative)).resolve()
            if pointer_root not in candidate.parents:
                raise BundleError("current.json points outside its data directory")
            manifest_path = candidate / "manifest.json"
            expected_hash = pointer.get("manifest_sha256")
            if not manifest_path.is_file() or not expected_hash:
                raise BundleError("Active bundle pointer is incomplete")
            if sha256_file(manifest_path) != expected_hash:
                raise BundleError("Active bundle manifest checksum does not match current.json")
            legacy = candidate.name == "processed" and (candidate / "train.parquet").is_file()
            resolved = _bundle_from_root(
                candidate,
                legacy=legacy,
                check_hashes=check_hashes,
                require_build_dir=require_build_dir,
            )
            if pointer.get("build_id") != resolved.build_id:
                raise BundleError("Active bundle pointer build_id does not match manifest")
            return resolved
        elif candidate.is_file() and candidate.suffix == ".json":
            pointer_path = candidate.resolve()
            pointer_root = pointer_path.parent
            pointer = _read_json(pointer_path)
            if pointer.get("schema_version") != BUNDLE_SCHEMA_VERSION:
                raise BundleError(f"Unsupported pointer schema: {candidate}")
            relative = pointer.get("bundle_path")
            if not relative or Path(str(relative)).is_absolute():
                raise BundleError(f"Pointer has no bundle_path: {candidate}")
            candidate = (pointer_root / str(relative)).resolve()
            if pointer_root not in candidate.parents:
                raise BundleError("Pointer points outside its data directory")
            manifest_path = candidate / "manifest.json"
            expected_hash = pointer.get("manifest_sha256")
            if not manifest_path.is_file() or not expected_hash:
                raise BundleError("Bundle pointer is incomplete")
            if sha256_file(manifest_path) != expected_hash:
                raise BundleError("Bundle manifest checksum does not match pointer")
            legacy = candidate.name == "processed" and (candidate / "train.parquet").is_file()
            resolved = _bundle_from_root(
                candidate,
                legacy=legacy,
                check_hashes=check_hashes,
                require_build_dir=require_build_dir,
            )
            if pointer.get("build_id") != resolved.build_id:
                raise BundleError("Bundle pointer build_id does not match manifest")
            return resolved
        if not candidate.is_absolute():
            candidate = candidate.resolve() if candidate.exists() else root / candidate
        legacy = candidate.name == "processed" and (candidate / "train.parquet").is_file()
        return _bundle_from_root(
            candidate,
            legacy=legacy,
            check_hashes=check_hashes,
            require_build_dir=require_build_dir,
        )

    pointer_path = root / "current.json"
    if pointer_path.exists():
        pointer = _read_json(pointer_path)
        if pointer.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise BundleError("Unsupported current.json schema")
        relative = pointer.get("bundle_path")
        if not relative or Path(str(relative)).is_absolute():
            raise BundleError("current.json bundle_path must be a relative path")
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents:
            raise BundleError("current.json points outside the data directory")
        manifest_path = candidate / "manifest.json"
        expected_hash = pointer.get("manifest_sha256")
        if not manifest_path.is_file() or not expected_hash:
            raise BundleError("Active bundle pointer is incomplete")
        if sha256_file(manifest_path) != expected_hash:
            raise BundleError("Active bundle manifest checksum does not match current.json")
        legacy = candidate.name == "processed" and (candidate / "train.parquet").is_file()
        bundle_obj = _bundle_from_root(candidate, legacy=legacy, check_hashes=check_hashes)
        if pointer.get("build_id") != bundle_obj.build_id:
            raise BundleError("Active bundle pointer build_id does not match manifest")
        if legacy:
            return bundle_obj
        legacy = root / "processed"
        if legacy.is_dir():
            legacy_bundle = _bundle_from_root(legacy, legacy=True, check_hashes=check_hashes)
            if legacy_bundle.build_id != bundle_obj.build_id:
                raise BundleError("data/processed is stale relative to data/current.json")
            return legacy_bundle
        return bundle_obj

    legacy = root / "processed"
    if legacy.is_dir():
        return _bundle_from_root(legacy, legacy=True, check_hashes=check_hashes)
    raise BundleError(f"No active dataset bundle found below {root}")


def load_bundle(bundle: str | Path | None = None, **kwargs: Any):
    """Resolve and load all train artifacts in one pinned read."""
    resolved = resolve_bundle(bundle, **kwargs)
    return resolved, resolved.load()


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


@contextmanager
def writer_lock(data_root: str | Path = DATA_ROOT) -> Iterator[None]:
    """Acquire a process-level writer lock using atomic file creation."""
    lock = Path(data_root) / ".bundle.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BundleError(f"Another data build is publishing: {lock}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _pointer_payload(
    bundle_root: Path,
    data_root: Path,
    *,
    active_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = bundle_root / "manifest.json"
    manifest = _read_json(manifest_path)
    active = (active_root or bundle_root).resolve()
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "build_id": manifest.get("build_id", bundle_root.name),
        "bundle_path": _relative(active, data_root),
        "manifest_sha256": sha256_file(active / "manifest.json"),
    }
    if active != bundle_root.resolve():
        payload["archive_path"] = _relative(bundle_root, data_root)
    return payload


def publish_bundle(
    staging_root: str | Path,
    *,
    data_root: str | Path = DATA_ROOT,
    activate: bool = True,
) -> DatasetBundle:
    """Publish an audited staging bundle and atomically update current.json."""
    staging = Path(staging_root).resolve()
    data = Path(data_root).resolve()
    manifest = _read_json(staging / "manifest.json")
    build_id = str(manifest.get("build_id", ""))
    if not build_id or Path(build_id).name != build_id:
        raise BundleError("Manifest has no filesystem-safe build_id")
    candidate = DatasetBundle(staging, manifest)
    verify_bundle(candidate, check_hashes=True, require_build_dir=False)
    versions = data / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    destination = versions / build_id
    if staging.parent == versions:
        raise BundleError("Published bundles are immutable; use rollback_bundle to reactivate")
    with writer_lock(data):
        if destination.exists():
            existing = _bundle_from_root(destination, check_hashes=True)
            if existing.input_fingerprint != candidate.input_fingerprint:
                raise BundleError(f"Build id already belongs to another input: {destination}")
            shutil.rmtree(staging, ignore_errors=True)
            if activate:
                materialize_processed_view(existing, data_root=data)
                write_manifest(
                    data / "current.json",
                    _pointer_payload(
                        destination,
                        data,
                        active_root=data / "processed",
                    ),
                )
            return existing
        os.replace(staging, destination)
        published = _bundle_from_root(destination, check_hashes=True)
        if activate:
            materialize_processed_view(published, data_root=data)
            pointer = data / "current.json"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="current-", suffix=".json", dir=data
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                temporary.write_text(
                    json.dumps(
                        _pointer_payload(
                            destination,
                            data,
                            active_root=data / "processed",
                        ),
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, pointer)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return published


def rollback_bundle(
    target: str | Path,
    *,
    data_root: str | Path = DATA_ROOT,
) -> DatasetBundle:
    """Atomically point ``current.json`` at an already verified bundle.

    Rollback never edits or deletes a published version.  ``target`` may be a
    build directory, ``versions/<build_id>``, or a pointer-like JSON path.
    """
    data = Path(data_root).resolve()
    candidate = Path(target)
    if not candidate.is_absolute():
        if candidate.exists():
            candidate = candidate.resolve()
        else:
            candidate = data / candidate
            if not candidate.exists() and Path(target).parent == Path('.'):
                candidate = data / "versions" / Path(target)
    if candidate.is_file() and candidate.suffix == ".json":
        pointer = _read_json(candidate)
        relative = pointer.get("bundle_path")
        if not relative:
            raise BundleError(f"Pointer has no bundle_path: {candidate}")
        candidate = (data / str(relative)).resolve()
    candidate = candidate.resolve()
    if data not in candidate.parents:
        raise BundleError("Rollback target must be inside the data directory")
    legacy = candidate.name == "processed" and (candidate / "train.parquet").is_file()
    verified = _bundle_from_root(candidate, legacy=legacy, check_hashes=True)
    with writer_lock(data):
        materialize_processed_view(verified, data_root=data)
        pointer = data / "current.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="current-", suffix=".json", dir=data
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                json.dumps(
                    _pointer_payload(
                        candidate,
                        data,
                        active_root=data / "processed",
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, pointer)
        finally:
            if temporary.exists():
                temporary.unlink()
    return verified


def find_existing_bundle(
    input_fingerprint: str,
    *,
    data_root: str | Path = DATA_ROOT,
) -> DatasetBundle | None:
    """Return a verified bundle for an unchanged input identity, if present."""
    root = Path(data_root).resolve()
    candidate = root / "versions" / make_build_id(input_fingerprint)
    if not candidate.is_dir():
        candidate = root / "processed"
        if not candidate.is_dir():
            return None
        try:
            active = _bundle_from_root(candidate, legacy=True, check_hashes=True)
        except BundleError:
            return None
        return active if active.input_fingerprint == input_fingerprint else None
    try:
        return _bundle_from_root(candidate, check_hashes=True)
    except BundleError:
        return None


def build_manifest_payload(
    *,
    build_id: str,
    input_fingerprint: str,
    metadata: dict[str, Any],
    bundle_root: str | Path,
) -> dict[str, Any]:
    """Create the non-cyclic manifest payload after artifacts are complete."""
    root = Path(bundle_root).resolve()
    payload = {
        **metadata,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "build_id": build_id,
        "input_fingerprint": input_fingerprint,
    }
    payload["artifacts"] = inventory_artifacts(root)
    return payload


def refresh_manifest_inventory(bundle_root: str | Path) -> dict[str, Any]:
    """Refresh artifact checksums after audit/export files are finalized."""
    root = Path(bundle_root)
    path = root / "manifest.json"
    payload = _read_json(path)
    payload["artifacts"] = inventory_artifacts(root)
    write_manifest(path, payload)
    return payload
