import hashlib
import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.data.adaptive_metadata import (
    ADAPTIVE_TEXT_POLICY,
    adaptive_text_metadata,
    semantic_ssl_mask,
)
from src.data.preprocessing import METADATA_FLAGS, METADATA_POLICY, metadata_flags
from src.data.provenance import sha256_file

logger = logging.getLogger(__name__)
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def get_item_text_mask(item_metadata, num_items, require_flags=True):
    if set(item_metadata) != set(range(num_items)):
        raise ValueError("Item metadata must contain exactly contiguous item indices")
    masks = []
    for idx in range(num_items):
        meta = item_metadata[idx]
        expected = metadata_flags(meta)
        if (require_flags or any(key in meta for key in METADATA_FLAGS)) and any(
            type(meta.get(key)) is not bool or meta[key] != expected[key]
            for key in METADATA_FLAGS
        ):
            raise ValueError(
                f"Missing/inconsistent metadata flags for item {idx}; rerun prepare_data.py"
            )
        masks.append(expected["has_usable_text"])
    return torch.tensor(masks, dtype=torch.bool)


def valid_text_tensor(tensor, mask):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 2
        or tensor.shape[0] != len(mask)
        or tensor.shape[1] == 0
    ):
        return False
    if not torch.isfinite(tensor).all():
        return False
    mask = mask.to(tensor.device)
    norms = tensor[mask].norm(dim=-1)
    return bool(
        torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
        and (tensor[~mask] == 0).all()
    )


def load_training_text(processed_dir, mappings):
    """One strict, verified data contract shared by training and the demo."""
    mask = get_item_text_mask(mappings["item_metadata"], len(mappings["item2id"]))
    path = Path(processed_dir) / "item_text_embeddings.pt"
    fingerprint = get_text_input_fingerprint(
        mappings["item_metadata"], len(mask), DEFAULT_ENCODER, PINNED_REVISION
    )
    tensor = load_verified_text_cache(path, fingerprint, item_mask=mask)
    if tensor is None or tensor.shape[1] != 384:
        raise ValueError(
            "Text artifact is stale, unpinned, or incompatible; rerun prepare_data.py"
        )
    return tensor, mask


def load_text_view(
    processed_dir,
    mappings,
    view: str = "shared",
):
    """Load a verified text view and its SSL eligibility mask.

    ``shared`` is the frozen canonical tensor used for evaluation diversity and
    all baselines.  ``adaptivegcl_quality`` is an optional derived tensor for
    AdaptiveGCL only; it excludes unverified title-derived brands and carries a
    separate category-only SSL mask.
    """
    if view in {"shared", "masked_text"}:
        tensor, item_mask = load_training_text(processed_dir, mappings)
        return tensor, item_mask, item_mask.clone(), {
            "feature_view": "shared",
            "metadata_policy": METADATA_POLICY,
        }
    if view != "adaptivegcl_quality":
        raise ValueError(f"Unknown AdaptiveGCL text view: {view}")

    path = Path(processed_dir) / "adaptivegcl_text_embeddings.pt"
    metadata_path = Path(str(path) + ".json")
    derived = adaptive_text_metadata(mappings["item_metadata"])
    item_mask = get_item_text_mask(derived, len(mappings["item2id"]))
    fingerprint = get_text_input_fingerprint(
        derived,
        len(item_mask),
        DEFAULT_ENCODER,
        PINNED_REVISION,
        policy=ADAPTIVE_TEXT_POLICY,
    )
    tensor = load_verified_text_cache(path, fingerprint, item_mask=item_mask)
    if tensor is None or tensor.shape[1] != 384:
        raise ValueError(
            "AdaptiveGCL text view is missing, stale, or incompatible; "
            "build adaptivegcl_text_embeddings.pt first"
        )
    try:
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_mask = sidecar["ssl_item_mask"]
        if not isinstance(saved_mask, list) or any(type(value) is not bool for value in saved_mask):
            raise ValueError("ssl_item_mask must contain booleans")
        ssl_mask = torch.tensor(saved_mask, dtype=torch.bool)
        expected_mask = torch.tensor(semantic_ssl_mask(derived, len(item_mask)), dtype=torch.bool)
        if not torch.equal(ssl_mask, expected_mask) or bool((ssl_mask & ~item_mask).any()):
            raise ValueError("ssl_item_mask does not match metadata policy")
        if sidecar.get("feature_view") != "adaptivegcl_quality":
            raise ValueError("invalid AdaptiveGCL feature view metadata")
        if sidecar.get("metadata_policy") != ADAPTIVE_TEXT_POLICY:
            raise ValueError("invalid AdaptiveGCL metadata policy")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("AdaptiveGCL text view sidecar is invalid") from exc
    return tensor, item_mask, ssl_mask, sidecar


def get_ssl_item_mask(item_metadata, item_mask, *, feature_view="shared", eligibility="view_default"):
    """Choose SSL eligibility independently of the text used for fusion."""
    if eligibility == "usable" or (eligibility == "view_default" and feature_view == "shared"):
        return item_mask.clone()
    if eligibility not in {"view_default", "item_specific"}:
        raise ValueError(f"Unknown SSL eligibility policy: {eligibility}")
    metadata = adaptive_text_metadata(item_metadata) if feature_view == "adaptivegcl_quality" else item_metadata
    mask = item_mask.new_tensor(semantic_ssl_mask(metadata, len(item_mask)))
    if bool((mask & ~item_mask).any()):
        raise ValueError("SSL eligibility cannot exceed usable text coverage")
    return mask


def build_adaptivegcl_text_view(
    item_metadata: dict[int, dict],
    num_items: int,
    *,
    save_path: str,
    model_name: str = DEFAULT_ENCODER,
    revision: str | None = PINNED_REVISION,
    **kwargs,
) -> torch.Tensor:
    """Encode the optional quality-controlled AdaptiveGCL text view."""
    derived = adaptive_text_metadata(item_metadata)
    tensor = encode_item_metadata(
        derived,
        num_items,
        model_name=model_name,
        revision=revision,
        save_path=save_path,
        policy=ADAPTIVE_TEXT_POLICY,
        **kwargs,
    )
    sidecar_path = Path(str(save_path) + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(
        {
            "feature_view": "adaptivegcl_quality",
            "source_metadata_policy": METADATA_POLICY,
            "ssl_item_mask": semantic_ssl_mask(derived, num_items),
        }
    )
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return tensor


def build_user_history_features(
    train_df: pd.DataFrame,
    item_text_features: torch.Tensor,
    num_users: int,
    item_text_mask: torch.Tensor | None = None,
):
    """Mean-pool item text features from each user's training history."""
    text_features = item_text_features.float().cpu()
    user_features = torch.zeros(
        (num_users, text_features.shape[1]), dtype=text_features.dtype
    )
    user_indices = torch.from_numpy(train_df["u_idx"].to_numpy(copy=True)).long()
    item_indices = torch.from_numpy(train_df["i_idx"].to_numpy(copy=True)).long()
    counts = torch.zeros(num_users, dtype=text_features.dtype)
    if item_text_mask is not None:
        if item_text_mask.dtype != torch.bool or item_text_mask.shape != (
            len(text_features),
        ):
            raise ValueError("Invalid item text mask")
        valid = item_text_mask.cpu()[item_indices]
        user_indices, item_indices = user_indices[valid], item_indices[valid]

    chunk_size = 100_000
    for start in range(0, len(user_indices), chunk_size):
        end = start + chunk_size
        user_chunk = user_indices[start:end]
        item_chunk = item_indices[start:end]
        user_features.index_add_(0, user_chunk, text_features[item_chunk])
        counts.index_add_(
            0,
            user_chunk,
            torch.ones(len(user_chunk), dtype=text_features.dtype),
        )

    result = user_features / counts.clamp_min(1.0).unsqueeze(1)
    return (result, counts > 0) if item_text_mask is not None else result


def format_item_text(meta: dict, *, policy: str = METADATA_POLICY) -> str:
    """Format item metadata dictionary into structured semantic text description."""
    from src.data.preprocessing import clean_text

    flags = metadata_flags(meta)
    parts = [clean_text(meta.get("title"))] if flags["has_title"] else []
    include_brand = flags["has_brand"]
    if policy == ADAPTIVE_TEXT_POLICY and meta.get("brand_source") not in {"metadata", "override"}:
        include_brand = False
    if policy not in {METADATA_POLICY, ADAPTIVE_TEXT_POLICY}:
        raise ValueError(f"Unknown metadata text policy: {policy}")
    if include_brand:
        parts.append(f"Brand: {clean_text(meta['brand'])}")
    if flags["has_specific_category"]:
        parts.append(f"Category: {clean_text(meta['categories'])}")

    return " | ".join(parts)


def get_text_input_fingerprint(
    item_metadata, num_items, model_name, revision=None, *, policy: str = METADATA_POLICY
) -> str:
    if set(item_metadata) != set(range(num_items)):
        raise ValueError(
            "Item metadata must contain exactly the contiguous item indices"
        )
    payload = {
        "format_version": 3,
        "metadata_policy": policy,
        "mask": get_item_text_mask(
            item_metadata, num_items, require_flags=False
        ).tolist(),
        "model_name": model_name,
        "revision": revision,
        "items": [
            [
                idx,
                item_metadata[idx].get("original_id"),
                format_item_text(item_metadata[idx], policy=policy),
            ]
            for idx in range(num_items)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_verified_text_cache(
    save_path, fingerprint, allow_fallback=False, item_mask=None
):
    """Return a verified tensor, or None for missing, stale, or corrupt cache."""
    try:
        with open(str(save_path) + ".json", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata["input_fingerprint"] != fingerprint:
            return None
        if not allow_fallback and metadata["backend"] != "sentence_transformers":
            return None
        if metadata["sha256"] != sha256_file(save_path):
            return None
        tensor = torch.load(save_path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            return None
        if list(tensor.shape) != metadata["shape"] or not torch.isfinite(tensor).all():
            return None
        saved_mask = metadata.get("item_text_mask")
        if not isinstance(saved_mask, list) or any(
            type(value) is not bool for value in saved_mask
        ):
            return None
        mask = torch.tensor(saved_mask, dtype=torch.bool)
        if item_mask is not None and not torch.equal(mask, item_mask.cpu()):
            return None
        if not valid_text_tensor(tensor, mask):
            return None
        return tensor
    except (OSError, KeyError, ValueError, TypeError, RuntimeError, EOFError):
        return None


def encode_item_metadata(
    item_metadata: dict[int, dict],
    num_items: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: str | None = None,
    save_path: str | None = None,
    force_recompute: bool = False,
    allow_fallback: bool = False,
    revision: str | None = None,
    policy: str = METADATA_POLICY,
) -> torch.Tensor:
    """Extract dense semantic text embeddings for all mapped items from metadata.

    Args:
        item_metadata: Mapping from contiguous item integer index to metadata dict.
        num_items: Total number of items in mapped dataset.
        model_name: Pretrained SentenceTransformer model identifier.
        batch_size: Mini-batch size for transformer inference.
        device: 'cuda' or 'cpu'. Defaults to auto-detect.
        save_path: Optional file path to cache embeddings (.pt).
        force_recompute: Whether to ignore cached tensor and recompute.

    Returns:
        Tensor (num_items, feature_dim): usable rows have unit norm; masked rows are zero.
    """
    fingerprint = get_text_input_fingerprint(
        item_metadata, num_items, model_name, revision, policy=policy
    )
    item_mask = get_item_text_mask(item_metadata, num_items, require_flags=False)
    if save_path and not force_recompute:
        embeddings = load_verified_text_cache(
            save_path, fingerprint, allow_fallback, item_mask
        )
        if embeddings is not None:
            logger.info("Verified text embedding cache hit: %s", save_path)
            return embeddings
        logger.info("Text embedding cache missing or invalid; recomputing.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Compiling text descriptions for {num_items:,} items...")
    text_corpus = []
    for i_idx in range(num_items):
        if not item_mask[i_idx]:
            continue
        meta = item_metadata.get(i_idx, {})
        text_corpus.append(format_item_text(meta, policy=policy))

    logger.info(f"Encoding {len(text_corpus):,} items with {model_name} on {device}...")
    resolved_revision = None
    try:
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"device": device}
        if revision is not None:
            model_kwargs["revision"] = revision
        model = SentenceTransformer(model_name, **model_kwargs)
        first_module = getattr(model, "_first_module", lambda: None)()
        transformer_config = getattr(
            getattr(first_module, "auto_model", None), "config", None
        )
        resolved_revision = getattr(transformer_config, "_commit_hash", None)
        if text_corpus:
            embeddings_np = model.encode(
                text_corpus,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings_tensor = torch.from_numpy(embeddings_np).float()
        else:
            embeddings_tensor = torch.zeros(
                (0, model.get_sentence_embedding_dimension())
            )
        backend = "sentence_transformers"
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                f"SentenceTransformer encoding failed for {model_name}"
            ) from e
        logger.warning(
            f"SentenceTransformer encoding failed or offline ({e}). Using TF-IDF fallback..."
        )
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(text_corpus)
        svd = TruncatedSVD(
            n_components=min(128, tfidf_mat.shape[1] - 1), random_state=42
        )
        svd_mat = svd.fit_transform(tfidf_mat)
        embeddings_tensor = F.normalize(torch.from_numpy(svd_mat).float(), dim=-1)
        backend = "tfidf_svd"

    if (
        embeddings_tensor.ndim != 2
        or embeddings_tensor.shape[0] != int(item_mask.sum())
        or not torch.isfinite(embeddings_tensor).all()
    ):
        raise ValueError("Encoder returned invalid text embeddings")
    full_tensor = torch.zeros((num_items, embeddings_tensor.shape[1]))
    full_tensor[item_mask] = embeddings_tensor
    embeddings_tensor = full_tensor
    if not valid_text_tensor(embeddings_tensor, item_mask):
        raise ValueError("Encoder returned invalid masked text embeddings")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        tensor_tmp = str(save_path) + ".tmp"
        torch.save(embeddings_tensor, tensor_tmp)
        os.replace(tensor_tmp, save_path)
        metadata = {
            "format_version": 3,
            "metadata_policy": policy,
            "item_text_mask": item_mask.tolist(),
            "input_fingerprint": fingerprint,
            "model_name": model_name,
            "revision": revision,
            "resolved_revision": resolved_revision,
            "backend": backend,
            "shape": list(embeddings_tensor.shape),
            "sha256": sha256_file(save_path),
        }
        metadata["software_versions"] = {}
        for package in ("sentence-transformers", "transformers", "tokenizers"):
            try:
                metadata["software_versions"][package] = version(package)
            except PackageNotFoundError:
                metadata["software_versions"][package] = None
        metadata_tmp = str(save_path) + ".json.tmp"
        with open(metadata_tmp, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
        os.replace(metadata_tmp, str(save_path) + ".json")
        logger.info(
            f"Saved {embeddings_tensor.shape} item text embeddings to {save_path}"
        )

    return embeddings_tensor
