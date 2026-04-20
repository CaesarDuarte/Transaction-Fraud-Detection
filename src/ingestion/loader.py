"""Load and merge IEEE-CIS transaction and identity tables."""
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")


def load_raw(split: str = "train") -> pd.DataFrame:
    transactions = pd.read_csv(RAW_DIR / f"{split}_transaction.csv")
    identity = pd.read_csv(RAW_DIR / f"{split}_identity.csv")
    df = transactions.merge(identity, on="TransactionID", how="left")
    return df


if __name__ == "__main__":
    df = load_raw("train")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")
    df.to_parquet("data/processed/train_merged.parquet", index=False)
    print("Saved to data/processed/train_merged.parquet")
