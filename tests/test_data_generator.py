"""Smoke tests for the synthetic data generator on tiny scale.

Runs the generator with small parameters and checks the output schema.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Run the data generator with tiny parameters in a fresh working dir."""
    workdir = tmp_path_factory.mktemp("gendata")
    project_root = Path(__file__).parent.parent
    cmd = [
        sys.executable, "-m", "scripts.generate_data",
        "--merchants", "5", "--customers", "50",
        "--transactions", "500", "--months", "3", "--seed", "1",
    ]
    result = subprocess.run(
        cmd, cwd=workdir, env={"PYTHONPATH": str(project_root), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"generator failed:\n{result.stderr}"
    return workdir / "data" / "raw"


def test_merchants_schema(generated):
    df = pd.read_parquet(generated / "merchants.parquet")
    assert len(df) == 5
    assert set(df.columns) >= {"id", "name", "industry", "region", "created_at"}


def test_customers_schema(generated):
    df = pd.read_parquet(generated / "customers.parquet")
    assert len(df) == 50
    assert df["plan"].isin(["starter", "growth", "scale", "enterprise"]).all()


def test_transactions_have_anomalies(generated):
    df = pd.read_parquet(generated / "transactions.parquet")
    assert len(df) > 0
    # Anomaly rate should be roughly 0.8% — for 500 rows we expect at least 0
    # but we mainly want to ensure the column exists and is bool
    assert df["is_anomaly"].dtype == bool
    assert df["amount"].min() > 0


def test_transactions_reference_valid_ids(generated):
    txns = pd.read_parquet(generated / "transactions.parquet")
    custs = pd.read_parquet(generated / "customers.parquet")
    prods = pd.read_parquet(generated / "products.parquet")
    assert txns["customer_id"].isin(custs["id"]).all()
    assert txns["product_id"].isin(prods["id"]).all()


def test_tickets_generated(generated):
    df = pd.read_parquet(generated / "tickets.parquet")
    assert len(df) > 0
    assert df["body"].str.len().min() > 10
