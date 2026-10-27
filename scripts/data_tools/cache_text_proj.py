"""Cache projected item text embeddings with SHA-256 checksum invalidation.

Usage:
    python scripts/data_tools/cache_text_proj.py [--bundle data/current.json] [--out data/cache/item_proj.pt]

Ensures projection is computed once and auto-invalidated whenever input features change.
"""

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch import nn

from src.data.bundle import BundleError, resolve_bundle

logger = logging.getLogger(__name__)


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute SHA-256 hash over raw contiguous tensor bytes."""
    tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def load_cached_projection(
    cache_path: str | Path,
    current_text_features: torch.Tensor,
) -> torch.Tensor | None:
    """Load cached projection tensor if cache exists and SHA-256 matches current input features.

    Returns:
        torch.Tensor of shape (num_items, emb_dim) if valid cache hit, else None.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        logger.info(f"Cache miss: {cache_path} does not exist.")
        return None

    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "proj_text" not in data or "input_sha256" not in data:
            logger.warning(f"Cache format invalid in {cache_path}. Invalidating cache.")
            return None

        current_hash = compute_tensor_sha256(current_text_features)
        cached_hash = data.get("input_sha256", "")

        if current_hash != cached_hash:
            logger.warning(
                f"Cache invalidated! Input feature hash mismatch: current={current_hash[:8]}... vs cached={cached_hash[:8]}..."
            )
            return None

        proj_text = data["proj_text"]
        if proj_text.shape[0] != current_text_features.shape[0]:
            logger.warning(
                f"Cache invalidated! Item count mismatch: current={current_text_features.shape[0]} vs cached={proj_text.shape[0]}"
            )
            return None

        logger.info(f"Cache hit! Successfully loaded projected embeddings from {cache_path}")
        return proj_text

    except Exception as exc:  # noqa: BLE001 - any corrupt cache is a cache miss.
        logger.warning(f"Failed to load cache from {cache_path} ({exc}). Invalidating cache.")
        return None


def save_cached_projection(
    proj_text: torch.Tensor,
    input_text_features: torch.Tensor,
    cache_path: str | Path,
) -> dict:
    """Persist projected text embeddings and metadata with input SHA-256 checksum."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    input_hash = compute_tensor_sha256(input_text_features)
    payload = {
        "proj_text": proj_text.detach().cpu(),
        "input_sha256": input_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_items": int(proj_text.shape[0]),
        "emb_dim": int(proj_text.shape[1]),
    }

    torch.save(payload, cache_path)
    logger.info(f"Saved projected embedding cache to {cache_path} (SHA-256={input_hash[:12]}...)")
    return payload


def cache_projection_pipeline(
    text_features_path: str | Path,
    output_path: str | Path,
    embedding_dim: int = 64,
    force: bool = False,
) -> torch.Tensor:
    """Run caching pipeline for text feature projection."""
    text_features_path = Path(text_features_path)
    if not text_features_path.exists():
        raise FileNotFoundError(f"Input text features not found: {text_features_path}")

    text_features = torch.load(text_features_path, map_location="cpu", weights_only=False)
    if isinstance(text_features, dict) and "embeddings" in text_features:
        text_features = text_features["embeddings"]

    if not force:
        cached = load_cached_projection(output_path, text_features)
        if cached is not None:
            return cached

    text_dim = text_features.shape[1]
    projection_layer = nn.Sequential(
        nn.Linear(text_dim, embedding_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(embedding_dim, embedding_dim),
    )
    for m in projection_layer.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    with torch.no_grad():
        proj_text = projection_layer(text_features.float())

    save_cached_projection(proj_text, text_features, output_path)
    return proj_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-features",
        type=Path,
        default=None,
        help="Path to item text embeddings .pt file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/cache/item_proj.pt"),
        help="Target cache output path",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Projected embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache is valid",
    )
    parser.add_argument(
        "--bundle",
        help="Bundle path or data/current.json when --text-features is omitted",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    text_features = args.text_features
    if text_features is None:
        try:
            text_features = resolve_bundle(args.bundle).artifact("item_text_embeddings.pt")
        except BundleError as exc:
            parser.error(f"Dataset bundle is invalid: {exc}")
    cache_projection_pipeline(text_features, args.out, embedding_dim=args.dim, force=args.force)


if __name__ == "__main__":
    main()
