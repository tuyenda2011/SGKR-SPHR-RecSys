import argparse
import os
import pickle
import sqlite3
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from tqdm import tqdm

from src.data.bundle import BundleError, resolve_bundle
from src.data.export import export_bundle_csvs
from src.utils.logging import setup_logger

logger = setup_logger("export_data")
ROOT = Path(__file__).resolve().parents[1]


def enrich_with_metadata(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Enrich numeric interaction DataFrame with product titles, brands, and categories."""
    item_metadata = mappings.get("item_metadata", {})
    id2user = {v: k for k, v in mappings.get("user2id", {}).items()}

    logger.info("Enriching DataFrame with product titles, brands, and categories...")
    
    # Map user and item details
    enriched_df = df.copy()
    enriched_df["original_user_id"] = enriched_df["u_idx"].map(id2user)

    titles = []
    brands = []
    categories = []
    asins = []

    for i_idx in tqdm(enriched_df["i_idx"], desc="Mapping Metadata", unit=" rows", leave=False):
        meta = item_metadata.get(i_idx, {})
        titles.append(meta.get("title", "Unknown"))
        brands.append(meta.get("brand", "Unknown"))
        categories.append(meta.get("categories", "Unknown"))
        asins.append(meta.get("original_id", "Unknown"))

    enriched_df["asin"] = asins
    enriched_df["product_title"] = titles
    enriched_df["brand"] = brands
    enriched_df["category"] = categories

    return enriched_df


def export_to_csv(df_dict: dict, output_dir: str):
    """Export datasets to CSV format (compatible with Excel & all tools)."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in df_dict.items():
        csv_path = os.path.join(output_dir, f"{name}.csv")
        logger.info(f"Writing CSV: {csv_path} ({len(df):,} rows)...")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Successfully exported {csv_path}")


def export_to_excel(df_dict: dict, output_dir: str, sample_size: int = 50000):
    """Export datasets to formatted Microsoft Excel (.xlsx) file with multiple sheets."""
    os.makedirs(output_dir, exist_ok=True)
    xlsx_path = os.path.join(output_dir, "amazon_electronics_dataset.xlsx")
    logger.info(f"Writing Excel workbook: {xlsx_path} (sampling up to {sample_size:,} rows per sheet)...")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl" if "openpyxl" in sys.modules else None) as writer:
        for name, df in df_dict.items():
            sample_df = df.head(sample_size)
            sample_df.to_excel(writer, sheet_name=name, index=False)
            logger.info(f"  -> Sheet '{name}': {len(sample_df):,} rows written.")

    logger.info(f"Successfully exported Excel workbook to {xlsx_path}")


def export_to_sqlite(df_dict: dict, output_dir: str):
    """Export datasets into a local SQLite database file for SQL queries."""
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "amazon_electronics.sqlite")
    logger.info(f"Writing SQLite database: {db_path}...")

    conn = sqlite3.connect(db_path)
    for name, df in df_dict.items():
        df.to_sql(name, conn, if_exists="replace", index=False)
        logger.info(f"  -> Table '{name}': {len(df):,} records created.")
    conn.close()
    logger.info(f"Successfully exported SQLite database to {db_path}")


def export_to_recsys_txt(df_dict: dict, output_dir: str):
    """Export datasets to standard Tab-Separated RecSys format (RecBole / LightGCN compatible)."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in df_dict.items():
        txt_path = os.path.join(output_dir, f"{name}.inter")
        logger.info(f"Writing RecSys format (.inter): {txt_path}...")
        recsys_df = df[["u_idx", "i_idx", "timestamp"]].rename(columns={
            "u_idx": "user_id:token",
            "i_idx": "item_id:token",
            "timestamp": "timestamp:float",
        })
        recsys_df.to_csv(txt_path, sep="\t", index=False)
        logger.info(f"Successfully exported {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Export processed Parquet dataset to CSV, Excel, SQLite, or RecSys TXT")
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "excel", "sqlite", "recsys_txt", "all"],
        help="Target export format (csv, excel, sqlite, recsys_txt, or all)",
    )
    parser.add_argument(
        "--with_meta",
        action="store_true",
        help="Enrich data with human-readable Product Titles, Brands, and Categories",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50000,
        help="Sample size when exporting to Excel (to stay well below Excel limits)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/exported",
        help="Destination directory for exported files",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Bundle path or data/current.json (default: active bundle when available)",
    )
    args = parser.parse_args()
    fmt = args.format.lower()

    bundle = None
    if args.bundle is not None or (ROOT / "data" / "current.json").exists():
        try:
            bundle = resolve_bundle(args.bundle)
        except BundleError as exc:
            logger.error("Dataset bundle is invalid: %s", exc)
            return

    standard_csv_exported = False
    if bundle is not None and fmt in {"csv", "all"}:
        train_df, val_df, test_df, mappings = bundle.load()
        disliked_path = bundle.artifact("disliked_interactions.parquet", required=False)
        disliked = pd.read_parquet(disliked_path) if disliked_path.exists() else None
        export_bundle_csvs(
            bundle_root=bundle.root,
            train=train_df,
            val=val_df,
            test=test_df,
            mappings=mappings,
            disliked=disliked,
            output_root=args.output_dir,
            seed=42,
        )
        standard_csv_exported = True
        logger.info("Standard bundle CSV export completed in '%s'", args.output_dir)
        if fmt == "csv":
            return

    processed_dir = str(bundle.train_dir) if bundle is not None else "data/processed"
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    mappings_path = os.path.join(processed_dir, "mappings.pkl")

    if not os.path.exists(train_path):
        logger.error(f"Processed dataset not found in {processed_dir}. Run prepare_data.py first.")
        return

    logger.info("Loading Parquet splits...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    if args.with_meta:
        train_df = enrich_with_metadata(train_df, mappings)
        val_df = enrich_with_metadata(val_df, mappings)
        test_df = enrich_with_metadata(test_df, mappings)

    df_dict = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    if (fmt == "csv" or fmt == "all") and not standard_csv_exported:
        export_to_csv(df_dict, args.output_dir)
    if fmt == "excel" or fmt == "all":
        export_to_excel(df_dict, args.output_dir, sample_size=args.sample_size)
    if fmt == "sqlite" or fmt == "all":
        export_to_sqlite(df_dict, args.output_dir)
    if fmt == "recsys_txt" or fmt == "all":
        export_to_recsys_txt(df_dict, args.output_dir)

    logger.info(f"All requested exports completed successfully in '{args.output_dir}'!")


if __name__ == "__main__":
    main()
