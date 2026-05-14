"""
Data validation for the IEEE-CIS dataset.

What this checks:
  - Expected columns are present
  - Row count is above minimum (sanity check against broken merges)
  - No duplicate TransactionIDs
  - Fraud rate is within the expected range
  - No negative transaction amounts
  - isFraud is strictly binary

Run after loader.py:
  python src/ingestion/validation.py
  python src/ingestion/validation.py --cleanup-raw   # also frees raw CSVs
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data/processed"

REQUIRED_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
]

FRAUD_RATE_MIN = 0.02
FRAUD_RATE_MAX = 0.10
MIN_ROWS = 500_000


class ValidationError(Exception):
    pass


def validate_train(df: pd.DataFrame) -> dict:
    """
    Run all validation checks on the merged training set.

    Returns:
        dict with validation results summary.

    Raises:
        ValidationError: on any critical check failure, with details in the message.
    """
    results = {}
    errors = []

    # 1. Row count
    results["row_count"] = df.shape[0]
    if df.shape[0] < MIN_ROWS:
        errors.append(
            f"Row count {df.shape[0]:,} is below expected minimum {MIN_ROWS:,}"
        )

    # 2. Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    results["missing_required_cols"] = missing_cols
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 3. No duplicate TransactionIDs
    if "TransactionID" in df.columns:
        n_dupes = int(df["TransactionID"].duplicated().sum())
        results["duplicate_transaction_ids"] = n_dupes
        if n_dupes > 0:
            errors.append(f"Found {n_dupes} duplicate TransactionIDs")
    else:
        # TransactionID is critical (already caught by check 2, but be explicit)
        errors.append("Cannot check duplicates: TransactionID column is missing")

    # 4. Fraud rate within expected range
    if "isFraud" in df.columns:
        fraud_rate = float(df["isFraud"].mean())
        results["fraud_rate"] = fraud_rate
        if not (FRAUD_RATE_MIN <= fraud_rate <= FRAUD_RATE_MAX):
            errors.append(
                f"Fraud rate {fraud_rate:.4%} outside expected range "
                f"[{FRAUD_RATE_MIN:.0%}, {FRAUD_RATE_MAX:.0%}]"
            )
    else:
        errors.append("isFraud column is missing (cannot check fraud rate or binary values)")

    # 5. No negative TransactionAmt
    if "TransactionAmt" in df.columns:
        n_negative = int((df["TransactionAmt"] < 0).sum())
        results["negative_amounts"] = n_negative
        if n_negative > 0:
            errors.append(f"Found {n_negative} negative TransactionAmt values")

    # 6. isFraud is strictly binary (only run if column exists and fraud rate passed)
    if "isFraud" in df.columns:
        unique_vals = set(df["isFraud"].dropna().unique())
        results["isFraud_unique_values"] = sorted(unique_vals)
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            errors.append(f"isFraud has unexpected values: {unique_vals}")

    # 7. Null rate summary (informational only)
    null_rates = (df.isnull().sum() / len(df)).sort_values(ascending=False)
    results["top_10_null_cols"] = null_rates.head(10).to_dict()
    results["cols_over_90pct_null"] = int((null_rates > 0.90).sum())

    _print_report(results, errors)

    if errors:
        raise ValidationError("; ".join(errors))

    return results


def _print_report(results: dict, errors: list) -> None:
    print(f"\n{'='*50}")
    print("DATA VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Row count:              {results.get('row_count', 'N/A'):,}")
    print(f"Fraud rate:             {results.get('fraud_rate', 0):.4%}")
    print(f"Duplicate IDs:          {results.get('duplicate_transaction_ids', 'N/A')}")
    print(f"Negative amounts:       {results.get('negative_amounts', 'N/A')}")
    print(f"Cols > 90% null:        {results.get('cols_over_90pct_null', 'N/A')}")
    print(f"\nTop 5 columns by null rate:")
    for col, rate in list(results.get("top_10_null_cols", {}).items())[:5]:
        print(f"  {col:<30} {rate:.1%}")
    if errors:
        print(f"\n{'─'*50}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗  {e}")
    else:
        print(f"\n  All checks passed.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="IEEE-CIS data validation")
    parser.add_argument(
        "--cleanup-raw",
        action="store_true",
        help=(
            "Delete raw CSV files from data/raw/ after validation passes. "
            "This frees ~1.5 GB of disk space. Irreversible."
        ),
    )
    args = parser.parse_args()

    train_path = PROCESSED_DIR / "train_merged.parquet"
    if not train_path.exists():
        print("ERROR: data/processed/train_merged.parquet not found.")
        print("Run loader.py first:  python src/ingestion/loader.py")
        sys.exit(1)

    logger.info("Loading processed train data ...")
    df = pd.read_parquet(train_path)

    try:
        validate_train(df)
        print("Validation passed.")
    except ValidationError as e:
        print(f"\nValidation FAILED: {e}")
        print("Raw files were NOT deleted.")
        sys.exit(1)

    # Only clean up raw files if validation succeeded
    if args.cleanup_raw:
        # Import here to avoid circular dependency if loader imports validation
        from src.ingestion.loader import cleanup_raw

        logger.info("Validation passed. Cleaning up raw files ...")
        for split in ("train", "test"):
            cleanup_raw(split)
        print("Raw files deleted. Parquet files in data/processed/ are your source of truth.")