"""
Tests for the ingestion layer.
Uses synthetic data  (no Kaggle files needed in CI).
"""

import pandas as pd
import numpy as np
import pytest

from src.ingestion.loader import describe_dataset, save_processed
from src.ingestion.validation import ValidationError, validate_train


def make_valid_df(n_rows: int = 600_000, fraud_rate: float = 0.035) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fraud = int(n_rows * fraud_rate)
    return pd.DataFrame({
        "TransactionID":  range(n_rows),
        "isFraud":        [1] * n_fraud + [0] * (n_rows - n_fraud),
        "TransactionDT":  rng.integers(0, 15_811_131, n_rows),
        "TransactionAmt": rng.uniform(0.5, 5000, n_rows).round(2),
        "ProductCD":      rng.choice(["W", "H", "C", "S", "R"], n_rows),
        "card1":  rng.integers(1000, 18000, n_rows),
        "card2":  rng.integers(100, 600, n_rows).astype(float),
        "card3":  rng.integers(100, 230, n_rows).astype(float),
        "card4":  rng.choice(["visa", "mastercard"], n_rows),
        "card5":  rng.integers(100, 240, n_rows).astype(float),
        "card6":  rng.choice(["debit", "credit"], n_rows),
        "addr1":  rng.integers(100, 540, n_rows).astype(float),
        "addr2":  rng.integers(10, 102, n_rows).astype(float),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n_rows),
        "R_emaildomain": rng.choice(["gmail.com", None], n_rows),
    })


# ── loader ────────────────────────────────────────────────────────────────────

def test_describe_runs_without_error():
    describe_dataset(make_valid_df(n_rows=1000))


def test_save_processed_creates_parquet(tmp_path, monkeypatch):
    import src.ingestion.loader as m
    monkeypatch.setattr(m, "PROCESSED_DIR", tmp_path)
    df = make_valid_df(n_rows=100)
    out = save_processed(df, "train")
    assert out.exists()
    assert pd.read_parquet(out).shape == df.shape


# ── validation — happy path ───────────────────────────────────────────────────

def test_valid_dataframe_passes():
    results = validate_train(make_valid_df())
    assert results["fraud_rate"] > 0.02
    assert results["duplicate_transaction_ids"] == 0
    assert results["negative_amounts"] == 0


# ── validation — error cases ──────────────────────────────────────────────────

def test_too_few_rows_raises():
    with pytest.raises(ValidationError, match="Row count"):
        validate_train(make_valid_df(n_rows=1000))


def test_negative_amounts_raises():
    df = make_valid_df()
    df.loc[0, "TransactionAmt"] = -10.0
    with pytest.raises(ValidationError, match="negative"):
        validate_train(df)


def test_fraud_rate_too_high_raises():
    with pytest.raises(ValidationError, match="Fraud rate"):
        validate_train(make_valid_df(fraud_rate=0.50))


def test_missing_required_column_raises():
    df = make_valid_df().drop(columns=["card1"])
    with pytest.raises(ValidationError, match="Missing required"):
        validate_train(df)


def test_duplicate_transaction_ids_raises():
    df = make_valid_df()
    df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]
    with pytest.raises(ValidationError, match="duplicate"):
        validate_train(df)
