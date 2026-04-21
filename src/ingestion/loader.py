"""
Load and merge IEEE-CIS transaction and identity tables.

The IEEE-CIS dataset comes in two files per split:
  - train_transaction.csv  (~590k rows, 394 columns) — core transaction data
  - train_identity.csv     (~145k rows,  41 columns) — device/network identity

Not every transaction has identity data (left join on TransactionID).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"


def load_raw(split: str = "train") -> pd.DataFrame:
    """
    Load and left-join transaction + identity tables for a given split.

    Args:
        split: 'train' or 'test'

    Returns:
        Merged DataFrame.

    Raises:
        FileNotFoundError: if raw CSV files are not found in data/raw/.
    """
    transaction_path = RAW_DIR / f"{split}_transaction.csv"
    identity_path = RAW_DIR / f"{split}_identity.csv"

    _check_files_exist(transaction_path, identity_path)

    logger.info("Loading %s_transaction.csv ...", split)
    transactions = pd.read_csv(transaction_path)
    logger.info("  -> shape: %s", transactions.shape)

    logger.info("Loading %s_identity.csv ...", split)
    identity = pd.read_csv(identity_path)
    logger.info("  -> shape: %s", identity.shape)

    logger.info("Merging on TransactionID (left join) ...")
    df = transactions.merge(identity, on="TransactionID", how="left")
    logger.info("  -> merged shape: %s", df.shape)

    identity_match_rate = transactions["TransactionID"].isin(
        identity["TransactionID"]
    ).mean()
    logger.info(
        "  -> %.1f%% of transactions have identity data",
        identity_match_rate * 100,
    )

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded dataset."""
    print(f"\n{'='*50}")
    print(f"Shape:          {df.shape[0]:,} rows x {df.shape[1]} columns")

    if "isFraud" in df.columns:
        fraud_rate = df["isFraud"].mean()
        fraud_count = int(df["isFraud"].sum())
        print(f"Fraud rate:     {fraud_rate:.4%}  ({fraud_count:,} fraud cases)")

    missing = df.isnull().sum()
    cols_with_missing = int((missing > 0).sum())
    total_missing_pct = missing.sum() / df.size
    print(f"Missing values: {cols_with_missing} columns affected ({total_missing_pct:.1%} of all cells)")
    print(f"Memory usage:   {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"{'='*50}\n")


def save_processed(df: pd.DataFrame, split: str = "train") -> Path:
    """
    Save merged DataFrame as Parquet to data/processed/.

    Parquet is preferred over CSV because:
      - preserves dtypes (no silent int->float conversions)
      - ~5x smaller file size
      - ~10x faster to read back

    Args:
        df:    merged DataFrame to save
        split: 'train' or 'test'

    Returns:
        Path to the saved file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{split}_merged.parquet"
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Saved to %s (%.1f MB)", output_path, size_mb)
    return output_path


def _check_files_exist(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Download the IEEE-CIS dataset from Kaggle:\n"
                "  https://www.kaggle.com/c/ieee-fraud-detection\n"
                "and place the CSV files in data/raw/"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    for split in ("train", "test"):
        df = load_raw(split)
        describe_dataset(df)
        save_processed(df, split)

    print("Phase 1 -> ingestion complete.")
    print("Now run: python src/ingestion/validation.py")
