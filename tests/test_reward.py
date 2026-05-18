"""Unit tests for src.reward.reward — Phase 4 sanity checks.

Uses in-memory SQLite via temporary files (sqlite3 doesn't accept :memory:
across processes, so we use tmp_path).
"""

from __future__ import annotations

import sqlite3
import pytest

from src.reward.reward import (
    check_construct,
    compute_reward,
    execute_sql_inproc,
    rows_match,
)


# ─── rows_match ────────────────────────────────────────────
def test_rows_match_simple():
    assert rows_match([(1, "a")], [(1, "a")])
    assert not rows_match([(1, "a")], [(1, "b")])


def test_rows_match_unordered():
    assert rows_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])


def test_rows_match_ordered_strict():
    assert not rows_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=True)


def test_rows_match_float_round():
    assert rows_match([(1.00001,)], [(1.00002,)])  # both round to 1.0000


def test_rows_match_handles_none():
    assert not rows_match(None, [(1,)])
    assert not rows_match([(1,)], None)


# ─── check_construct ──────────────────────────────────────
def test_check_construct_plain():
    assert check_construct("SELECT * FROM t", "plain")
    assert not check_construct("SELECT * FROM t", "CTE")


def test_check_construct_having():
    assert check_construct(
        "SELECT dept FROM emp GROUP BY dept HAVING COUNT(*) > 5", "HAVING"
    )


def test_check_construct_exists():
    assert check_construct(
        "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.aid = a.id)",
        "EXISTS",
    )


def test_check_construct_cte():
    assert check_construct(
        "WITH t AS (SELECT * FROM x) SELECT * FROM t",
        "CTE",
    )


# ─── execute_sql_inproc & compute_reward ──────────────────
@pytest.fixture
def tiny_db(tmp_path):
    """Create a small SQLite DB for testing."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?)",
        [(1, "alice", 30), (2, "bob", 25), (3, "carol", 35)],
    )
    conn.commit()
    conn.close()
    return str(db_path)


def test_execute_sql_inproc_ok(tiny_db):
    rows, err = execute_sql_inproc("SELECT name FROM users ORDER BY id", tiny_db)
    assert err is None
    assert rows == [("alice",), ("bob",), ("carol",)]


def test_execute_sql_inproc_error(tiny_db):
    rows, err = execute_sql_inproc("SELECT * FROM nonexistent_table", tiny_db)
    assert rows is None
    assert err is not None


def test_execute_sql_inproc_empty(tiny_db):
    rows, err = execute_sql_inproc("", tiny_db)
    assert rows is None
    assert err == "empty_sql"


def test_compute_reward_gold_eq_gold(tiny_db):
    """Gold SQL fed as both gen and gold should produce r_exec=1.0, r_construct=1.0."""
    gold = "SELECT name FROM users WHERE age > 28"
    out = compute_reward(
        gen_sql=gold, gold_sql=gold,
        db_path=tiny_db,
        target_construct="plain",
        lambda_construct=0.2,
        use_subprocess=False,
    )
    assert out["r_exec"] == 1.0
    assert out["r_construct"] == 1.0
    assert out["r_total"] == pytest.approx(1.2)


def test_compute_reward_wrong_construct(tiny_db):
    """Gold matches execution but not target construct → r_construct=0."""
    gold = "SELECT name FROM users WHERE age > 28"
    out = compute_reward(
        gen_sql=gold, gold_sql=gold,
        db_path=tiny_db,
        target_construct="CTE",  # gold is plain, so this mismatch
        lambda_construct=0.2,
        use_subprocess=False,
    )
    assert out["r_exec"] == 1.0
    assert out["r_construct"] == 0.0
    assert out["r_total"] == pytest.approx(1.0)


def test_compute_reward_exec_failure(tiny_db):
    """Different gen vs gold should produce r_exec=0."""
    out = compute_reward(
        gen_sql="SELECT name FROM users WHERE age > 100",  # returns []
        gold_sql="SELECT name FROM users WHERE age > 28",  # returns 2 rows
        db_path=tiny_db,
        target_construct="plain",
        lambda_construct=0.2,
        use_subprocess=False,
    )
    assert out["r_exec"] == 0.0
    assert out["r_construct"] == 1.0  # both are plain SELECTs
    assert out["r_total"] == pytest.approx(0.2)


def test_compute_reward_invalid_sql(tiny_db):
    """gen_sql with syntax error → r_exec=0, error captured."""
    out = compute_reward(
        gen_sql="THIS IS NOT SQL",
        gold_sql="SELECT 1",
        db_path=tiny_db,
        target_construct="plain",
        lambda_construct=0.2,
        use_subprocess=False,
    )
    assert out["r_exec"] == 0.0
    assert out["pred_error"] is not None
