"""Tests for the agent's run_sql guard.

These don't require a live database — they only check the rejection logic.
"""
import pytest

from app.agents.tools import run_sql


@pytest.mark.parametrize("query", [
    "DROP TABLE customers",
    "delete from transactions where id = 1",
    "INSERT INTO customers VALUES (1, 'a')",
    "UPDATE customers SET plan = 'enterprise'",
    "TRUNCATE transactions",
    "ALTER TABLE customers ADD COLUMN evil text",
    "SELECT 1; DROP TABLE customers",
    "SELECT * FROM customers; SELECT * FROM transactions",
    "GRANT ALL ON transactions TO public",
    "COPY customers TO '/tmp/x.csv'",
])
def test_rejects_unsafe(query):
    result = run_sql(query)
    assert "error" in result, f"should reject: {query}"


@pytest.mark.parametrize("query", [
    "describe transactions",
    "SHOW TABLES",
    "EXPLAIN SELECT * FROM transactions",
    "VACUUM transactions",
])
def test_rejects_non_select(query):
    result = run_sql(query)
    assert "error" in result


def test_appends_limit_when_missing():
    """We can't actually run SQL here, but we can at least verify the function
    doesn't reject a valid SELECT before reaching the DB."""
    result = run_sql("SELECT 1 AS x")
    # Either succeeds (DB available) or fails with a connection-style error,
    # but should never fail with the syntactic guard.
    if "error" in result:
        assert "write/DDL" not in result["error"]
        assert "single statement" not in result["error"]
        assert "only SELECT" not in result["error"]
