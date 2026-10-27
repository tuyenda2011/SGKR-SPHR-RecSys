import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
REPO_ROOT = Path(__file__).resolve().parents[1]

# Force UTF-8 encoding for Windows Command Prompt/PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pickle

import pandas as pd

from src.data.bundle import BundleError, resolve_bundle
from src.data.provenance import sha256_file
from src.data.sparsity import create_sparse_train_set
from src.data.text_encoder import (
    load_training_text,
)
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.models.directau import DirectAU
from src.models.lightgcn import LightGCN
from src.models.xsimgcl import XSimGCL
from src.training.trainer import Trainer
from src.utils.checkpoints import (
    get_checkpoint_dir,
    get_model_output_dir,
    get_run_fingerprint,
)
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.paths import resolve_output_root, write_run_manifest
from src.utils.seed import set_seed

logger = setup_logger("train_script")


def append_to_model_results_csv(results: dict, model_name: str, sparsity: float, seed: int,
                                output_root: str = "results"):
    """Save or append run results to dedicated per-model CSV file (results/aggregated/{model}_results.csv)."""
    agg_dir = os.path.join(output_root, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    model_csv = os.path.join(agg_dir, f"{model_name}_results.csv")

    test_m = results["test_metrics"]
    val_m = results["val_metrics"]
    rep_m = results["representation_metrics"]
    svd_m = results["svd_metrics"]
    sub_m = results["subgroup_metrics"]
    tail_m = sub_m["Tail (Low-Activity)"]
    head_m = sub_m["Head (Active)"]

    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "experiment_fingerprint": results["experiment_fingerprint"],
        "evaluation_protocol": results["evaluation_protocol"],
        "monitor": results["monitor"],
        "monitor_value": val_m[results["monitor"]],
        "scoring_metric": results["scoring_metric"],
        "profile": results["profile"],
        "text_policy": None,
        "sparsity": sparsity,
        "seed": seed,
        "best_epoch": results["best_epoch"],
        "total_epochs": results["total_epochs"],
        "train_time_sec": round(results["total_train_time"], 2),
        "inference_latency_ms": round(results["inference_latency_ms_per_user"], 3),
        # Accuracy Metrics
        "Recall@10": round(test_m["Recall@10"], 4),
        "NDCG@10": round(test_m["NDCG@10"], 4),
        "MRR@10": round(test_m["MRR@10"], 4),
        "Recall@20": round(test_m["Recall@20"], 4),
        "NDCG@20": round(test_m["NDCG@20"], 4),
        # Beyond-Accuracy Metrics
        "Diversity@10": round(test_m["Diversity@10"], 4),
        "Novelty@10": round(test_m["Novelty@10"], 4),
        "Coverage@10": round(test_m["Coverage@10"], 4),
        "Gini@10": round(test_m["Gini@10"], 4),
        # Representation Geometry
        "Alignment": round(rep_m["alignment"], 4),
        "Mean_Uniformity": round(rep_m["mean_uniformity"], 4),
        "User_Effective_Rank": round(svd_m["user_effective_rank"], 2),
        "Item_Effective_Rank": round(svd_m["item_effective_rank"], 2),
        # Subgroup
        "Tail_Recall@10": round(tail_m["Recall@10"], 4),
        "Tail_NDCG@10": round(tail_m["NDCG@10"], 4),
        "Head_Recall@10": round(head_m["Recall@10"], 4),
        "Head_NDCG@10": round(head_m["NDCG@10"], 4),
        "Val_NDCG@10": round(val_m["NDCG@10"], 4),
    }

    new_df = pd.DataFrame([row])
    if os.path.exists(model_csv):
        existing_df = pd.read_csv(model_csv)
        # Update row if exact same model, sparsity, seed exists, else append
        mask = (existing_df["sparsity"] == sparsity) & (existing_df["seed"] == seed)
        mask &= existing_df.get("experiment_fingerprint", pd.Series("", index=existing_df.index)) == results["experiment_fingerprint"]
        if mask.any():
            existing_df = existing_df[~mask]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(model_csv, index=False)
    logger.info(f"Updated dedicated model results file: {model_csv}")


def main():
    parser = argparse.ArgumentParser(description="Train the baseline graph recommenders (LightGCN, XSimGCL, DirectAU)")
    parser.add_argument("--model", type=str, required=True, choices=["lightgcn", "xsimgcl", "directau"], help="Model name")
    parser.add_argument("--sparsity", type=float, default=1.0, help="Sparsity ratio for training edges (0.25 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--validation_only", action="store_true", help="Tune using validation without reading test labels")
    parser.add_argument("--dry_run", action="store_true", help="Print effective configuration without training")
    parser.add_argument("--model_diagnostics", action="store_true", help="Log deterministic layer/gate/norm diagnostics each epoch")
    parser.add_argument("--diagnostics_sample_size", type=int, default=None, help="Node sample size for model diagnostics")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest saved checkpoint")
    parser.add_argument("--config_dir", type=str, default="configs", help="Config directory")
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional bundle/archive path; default uses active data/processed view",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Root for this run. Defaults to results/runs/train_<model>_<density>_<seed>_<timestamp>/.",
    )
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    if not 0 < args.sparsity <= 1 or args.seed < 0:
        parser.error("sparsity must be in (0, 1] and seed nonnegative")
    if args.resume and not args.output_root:
        parser.error("--resume requires an explicit --output_root")

    args.output_root = str(
        resolve_output_root(
            args.output_root,
            kind="train",
            label=f"{args.model}_s{int(args.sparsity * 100)}_seed{args.seed}",
        )
    )

    # 1. Set seed
    set_seed(args.seed)

    # 2. Get device
    device = get_device()

    # 3. Load config
    config = load_config(args.model, args.config_dir, density=args.sparsity)
    config["training"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.model_diagnostics:
        config.setdefault("evaluation", {})["model_diagnostics"] = True
    if args.diagnostics_sample_size is not None:
        if args.diagnostics_sample_size < 1:
            parser.error("diagnostics_sample_size must be positive")
        config.setdefault("evaluation", {})["diagnostics_sample_size"] = args.diagnostics_sample_size
    configured_processed = Path(config["dataset"]["processed_dir"])
    use_active_bundle = args.bundle is not None or (
        args.bundle is None
        and configured_processed.as_posix().replace("\\", "/") == "data/processed"
        and (REPO_ROOT / "data" / "current.json").exists()
    )
    bundle = None
    if use_active_bundle:
        try:
            bundle = resolve_bundle(args.bundle)
        except BundleError as exc:
            raise FileNotFoundError(f"Dataset bundle is invalid: {exc}") from exc
    config["experiment_fingerprint"] = get_run_fingerprint(
        args.model,
        args.sparsity,
        args.seed,
        config,
        args.config_dir,
        manifest_path=bundle.manifest_path if bundle is not None else None,
    )
    config["history_dir"] = get_model_output_dir("history", args.model, args.output_root)

    config["validation_only"] = args.validation_only
    if args.dry_run:
        print(json.dumps({"model": args.model, "sparsity": args.sparsity, "output_root": args.output_root, "config": config}, indent=2))
        return

    write_run_manifest(
        args.output_root,
        kind="train",
        metadata={
            "model": args.model,
            "density": args.sparsity,
            "seed": args.seed,
            "epochs": config["training"]["epochs"],
            "validation_only": args.validation_only,
        },
        # A multi-model runner creates the run-level manifest. A child train
        # process must not replace it with a single-model manifest.
        overwrite=False,
    )

    # 4. Load one pinned dataset bundle. Explicit config paths remain a
    # compatibility adapter for synthetic tests and isolated legacy fixtures.
    if use_active_bundle:
        processed_dir = str(bundle.train_dir)
        config["dataset"]["bundle_id"] = bundle.build_id
        config["dataset"]["bundle_path"] = str(bundle.root)
        run_manifest_path = Path(args.output_root) / "run_manifest.json"
        if run_manifest_path.is_file():
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            run_manifest.setdefault("parameters", {}).update(
                {
                    "bundle_id": bundle.build_id,
                    "bundle_path": str(bundle.root),
                    "manifest_sha256": hashlib.sha256(
                        bundle.manifest_path.read_bytes()
                    ).hexdigest(),
                }
            )
            run_manifest_path.write_text(
                json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        processed_dir = str(configured_processed)
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    mappings_path = os.path.join(processed_dir, "mappings.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset artifacts not found at {processed_dir}. Run prepare_data.py first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = None if args.validation_only else pd.read_parquet(test_path)

    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    stats = mappings["stats"]
    num_users = stats["num_users"]
    num_items = stats["num_items"]

    # 5. Apply sparsity sampling to training set (Validation & Test stay 100% fixed)
    train_df_sparse = create_sparse_train_set(train_df, sparsity_ratio=args.sparsity, seed=args.seed)

    # 6. Initialize warm-start evaluators. Models learn from sparse train data,
    # while all known positives remain excluded from recommendation candidates.
    top_k_list = config["evaluation"]["top_k"]
    eval_batch_size = config["evaluation"]["eval_batch_size"]
    candidate_items = set(train_df_sparse["i_idx"].unique())
    from src.data.quality_report import warm_cohort
    val_warm, val_cohort = warm_cohort(train_df_sparse, val_df)
    test_warm, test_cohort = (None, None) if test_df is None else warm_cohort(train_df_sparse, test_df)
    logger.info("Evaluation protocol: warm-start; validation cohort=%s; test cohort=%s", val_cohort, test_cohort)
    logger.info(
        f"Warm-start evaluation targets: val={len(val_warm):,}/{len(val_df):,}, "
        f"test={len(test_warm) if test_warm is not None else 'not read'}"
    )

    val_evaluator = Evaluator(
        train_df,
        val_warm,
        num_users,
        num_items,
        k_list=top_k_list,
        batch_size=eval_batch_size,
        candidate_items=candidate_items,
        popularity_df=train_df,
    )
    test_evaluator = None
    if not args.validation_only:
        test_history = pd.concat([train_df, val_df], ignore_index=True)
        # Shared frozen content features for evaluation, including ID-only baselines.
        diversity_features, diversity_mask = load_training_text(processed_dir, mappings)
        test_evaluator = Evaluator(
            test_history,
            test_warm,
            num_users,
            num_items,
            k_list=top_k_list,
            batch_size=eval_batch_size,
            candidate_items=candidate_items,
            popularity_df=train_df,
            diversity_features=diversity_features,
            diversity_mask=diversity_mask,
        )

    # 7. Instantiate model
    emb_dim = config["model"]["embedding_dim"]
    num_layers = config["model"]["num_layers"]

    if args.model == "lightgcn":
        model = LightGCN(num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers)
    elif args.model == "xsimgcl":
        xsim_cfg = config["xsimgcl"]
        model = XSimGCL(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            contrastive_weight=xsim_cfg["contrastive_weight"],
            temperature=xsim_cfg["temperature"],
            epsilon=xsim_cfg["epsilon"],
            contrastive_layer=xsim_cfg.get("contrastive_layer", 1),
        )
    elif args.model == "directau":
        dau_cfg = config["directau"]
        model = DirectAU(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            gamma=dau_cfg["gamma"],
            t=dau_cfg["t"],
            profile=dau_cfg.get("profile", "project_cosine"),
        )

    # 8. Train model
    sparsity_tag = f"s{int(args.sparsity * 100)}"
    checkpoint_dir = get_checkpoint_dir(args.model, args.output_root)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_{sparsity_tag}_seed{args.seed}.pt")

    trainer = Trainer(
        model,
        train_df_sparse,
        val_evaluator,
        test_evaluator,
        config,
        device,
        user_disliked_items=mappings.get("user_disliked_items", {}),
        subgroup_reference_df=train_df,
    )
    if os.path.exists(checkpoint_path) and not args.resume:
        raise FileExistsError("Run already exists; choose a new --output_root or explicit --resume")
    results = trainer.train(checkpoint_path, resume=args.resume)

    # 9. Save run results to JSON
    results["sparsity_level"] = args.sparsity
    results["seed"] = args.seed
    results["max_epochs"] = config["training"]["epochs"]
    results["experiment_fingerprint"] = config["experiment_fingerprint"]
    results["experiment_family"] = get_run_fingerprint(
        args.model,
        args.sparsity,
        None,
        config,
        args.config_dir,
        manifest_path=bundle.manifest_path if bundle is not None else None,
    )
    results["effective_config"] = config
    results["scoring_metric"] = model.scoring_metric
    results["encoder"] = "XSimGCL" if args.model == "xsimgcl" else "LightGCN"
    results["profile"] = getattr(model, "profile", None)
    results["text_policy"] = None
    results["feature_view"] = None
    results["fusion_mode"] = None
    results["residual_alpha"] = None
    results["evaluation_protocol"] = EVALUATION_PROTOCOL
    results["evaluation_metadata"] = {
        "main_metric_cohort": "warm_start_users_and_items",
        "validation_cohort": val_cohort,
        "test_cohort": test_cohort,
        "history_mask_policy": "full_train_for_val_full_train_plus_val_for_test",
        "sparsity_scope": "model_training_graph_only",
        "popularity_reference": "full_train_unique_users",
        "subgroup_degree_reference": "full_train_fixed_across_sparsity",
        "candidate_items": len(candidate_items),
        "validation_targets": len(val_warm),
        "validation_total_targets": len(val_df),
        "test_targets": len(test_warm) if test_warm is not None else None,
        "test_total_targets": len(test_df) if test_df is not None else None,
        "uniformity_sample_seed": args.seed,
        "latency_scope": "full_catalog_scoring_masking_topk_excludes_embedding_forward",
    }

    results_dir = get_model_output_dir("raw", args.model, args.output_root)
    os.makedirs(results_dir, exist_ok=True)
    run_file = os.path.join(results_dir, f"{args.model}_{sparsity_tag}_seed{args.seed}.json")

    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved run results to {run_file}")

    if args.validation_only:
        logger.info("Validation-only run complete; test labels were not read")
        return

    # 10. Save / append to dedicated per-model CSV file (results/aggregated/{model}_results.csv)
    append_to_model_results_csv(results, args.model, args.sparsity, args.seed, args.output_root)

    # 11. Select global best using the configured validation monitor.
    global_best_meta_path = os.path.join(checkpoint_dir, f"{args.model}_best_meta.json")
    global_best_pt_path = os.path.join(checkpoint_dir, f"{args.model}_best.pt")
    
    monitor = results["monitor"]
    current_val_ndcg = results["val_metrics"][monitor]
    current_test_ndcg = results["test_metrics"]["NDCG@10"]
    is_new_global_best = True

    if os.path.exists(global_best_meta_path):
        try:
            with open(global_best_meta_path, "r", encoding="utf-8") as f:
                prev_best = json.load(f)
            if prev_best.get("experiment_family") == results["experiment_family"] and prev_best.get("monitor_value", float("-inf")) >= current_val_ndcg:
                is_new_global_best = False
        except Exception:  # noqa: BLE001 - a malformed previous best must not block training.
            is_new_global_best = True

    if is_new_global_best and os.path.exists(checkpoint_path):
        import shutil
        shutil.copyfile(checkpoint_path, global_best_pt_path)
        with open(global_best_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "experiment_fingerprint": config["experiment_fingerprint"],
                "experiment_family": results["experiment_family"],
                "sparsity": args.sparsity,
                "seed": args.seed,
                "best_epoch": results["best_epoch"],
                "monitor": monitor,
                "monitor_value": current_val_ndcg,
                "Val_NDCG@10": results["val_metrics"]["NDCG@10"],
                "NDCG@10": current_test_ndcg,
                "Recall@10": results["test_metrics"]["Recall@10"],
                "Diversity@10": results["test_metrics"]["Diversity@10"],
                "Novelty@10": results["test_metrics"]["Novelty@10"],
                "source_checkpoint": checkpoint_path,
            }, f, indent=2)
        logger.info(
            f"[GLOBAL BEST] Updated model for {args.model.upper()} -> "
            f"{global_best_pt_path} (Val {monitor}: {current_val_ndcg:.4f})"
        )


if __name__ == "__main__":
    main()
