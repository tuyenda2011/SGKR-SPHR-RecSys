"""Shared output-root and run-discovery helpers.

Every new experiment gets one self-contained directory under ``results/runs``.
The legacy ``results/{raw,history,checkpoints,aggregated}`` layout remains
readable so old artifacts can still be audited without being mixed into new
runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
RUN_LAYOUT_VERSION = "run-root-v1"


def resolve_path(value: str | os.PathLike[str], *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a project-relative path deterministically."""

    path = Path(value)
    return path if path.is_absolute() else base / path


def _unique_run_path(kind: str, label: str | None = None) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    prefix = kind.strip().lower().replace(" ", "_")
    if label:
        prefix = f"{prefix}_{label.strip().lower().replace(' ', '_')}"
    candidate = RUNS_ROOT / f"{prefix}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = RUNS_ROOT / f"{prefix}_{stamp}_{suffix:02d}"
        suffix += 1
    return candidate


def resolve_output_root(
    output_root: str | os.PathLike[str] | None,
    *,
    kind: str,
    label: str | None = None,
) -> Path:
    """Return an explicit root or a fresh canonical root for a new run.

    Explicit roots are never rewritten. This preserves reproducible resume
    commands and lets callers intentionally inspect a legacy root.
    """

    if output_root:
        return resolve_path(output_root)

    # A few tests and external callers create a temporary legacy ``results``
    # tree while changing cwd. Keep that fixture-compatible behavior; normal
    # project runs always use the canonical run root below.
    cwd_results = Path.cwd() / "results"
    if Path.cwd().resolve() != PROJECT_ROOT.resolve() and (cwd_results / "raw").exists():
        return cwd_results
    return _unique_run_path(kind, label)


def write_run_manifest(
    output_root: str | os.PathLike[str],
    *,
    kind: str,
    metadata: Mapping[str, object] | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a small identity file at the root of every new run."""

    root = resolve_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_manifest.json"
    if path.exists() and not overwrite:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        # A runner owns the identity of a shared multi-model root. A direct
        # train root may still refresh its own stale manifest.
        if existing.get("run_kind") in {"smoke", "evaluation", "train_all", "benchmark"}:
            return path
    payload: dict[str, object] = {
        "schema_version": 1,
        "layout": RUN_LAYOUT_VERSION,
        "run_id": root.name,
        "run_kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": root.as_posix(),
    }
    if metadata:
        payload["parameters"] = dict(metadata)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _candidate_run_roots() -> Iterable[Path]:
    if not RUNS_ROOT.is_dir():
        return ()
    roots = []
    for parent in RUNS_ROOT.iterdir():
        if not parent.is_dir():
            continue
        if (parent / "run_manifest.json").is_file():
            roots.append(parent)
        else:
            # A comparison can group independently registered single runs.
            roots.extend(child for child in parent.iterdir()
                         if child.is_dir() and (child / "run_manifest.json").is_file())
    return sorted(
        roots,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def find_latest_run_root(
    *,
    model_name: str | None = None,
    sparsity: float = 1.0,
    seed: int = 42,
    required_relative_path: str | None = None,
    preferred_kinds: tuple[str, ...] = (),
) -> Path | None:
    """Find the newest canonical run matching an artifact or model.

    Legacy ``results`` is returned only when no canonical run matches. This
    keeps the dashboard usable while old outputs are being phased out.
    """

    ordered = list(_candidate_run_roots())
    if preferred_kinds:
        def kind_rank(path: Path) -> int:
            for index, kind in enumerate(preferred_kinds):
                if path.name.startswith(f"{kind}_"):
                    return index
            return len(preferred_kinds)

        ordered.sort(key=lambda path: (kind_rank(path), -path.stat().st_mtime))

    tag = f"s{int(sparsity * 100)}"
    for root in ordered:
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        if required_relative_path and not (root / required_relative_path).exists():
            continue
        if model_name:
            model_file = root / "checkpoints" / model_name / f"{model_name}_{tag}_seed{seed}.pt"
            if not model_file.exists():
                continue
        return root

    legacy_root = RESULTS_ROOT
    if required_relative_path and not (legacy_root / required_relative_path).exists():
        return None
    if model_name:
        legacy_model = legacy_root / "checkpoints" / model_name / f"{model_name}_{tag}_seed{seed}.pt"
        if not legacy_model.exists():
            return None
    return legacy_root if legacy_root.exists() else None
