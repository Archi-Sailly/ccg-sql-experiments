"""Unit tests for `src.data.labeling.classify_construct`.

The tests verify:

* At least five positive examples per class (plain / HAVING / EXISTS /
  Window / CTE).
* Priority resolution when a query exhibits multiple constructs.
* Robust fallback behaviour for empty, whitespace-only, and unparseable
  inputs.
"""

from __future__ import annotations

import pytest

from src.data.labeling import (
    CONSTRUCT_LABELS,
    DEFAULT_PRIORITY,
    classify_construct,
    classify_construct_with_meta,
)

# ---------------------------------------------------------------------------
# Per-class positive samples (>= 5 per class)
# ---------------------------------------------------------------------------

PLAIN_CASES: list[str] = [
    "SELECT name FROM users",
    "SELECT * FROM employees WHERE salary > 50000",
    "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id",
    "SELECT COUNT(*) FROM products WHERE category = 'books'",
    "SELECT DISTINCT department FROM employees ORDER BY department ASC LIMIT 10",
    "SELECT category, AVG(price) FROM products GROUP BY category",
]

HAVING_CASES: list[str] = [
    "SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5",
    "SELECT category, AVG(price) FROM products GROUP BY category HAVING AVG(price) > 100",
    "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id HAVING SUM(total) >= 1000",
    "SELECT y, COUNT(DISTINCT user_id) FROM logins GROUP BY y HAVING COUNT(DISTINCT user_id) > 1000",
    "SELECT brand, MIN(price) FROM products GROUP BY brand HAVING MIN(price) < 50 AND COUNT(*) > 10",
    "SELECT a, b, COUNT(*) AS c FROM t GROUP BY a, b HAVING c > 1 ORDER BY c DESC",
]

EXISTS_CASES: list[str] = [
    "SELECT name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)",
    "SELECT * FROM employees e WHERE EXISTS (SELECT 1 FROM dependents d WHERE d.emp_id = e.id)",
    "SELECT title FROM books b WHERE NOT EXISTS (SELECT 1 FROM checkouts c WHERE c.book_id = b.id)",
    "SELECT * FROM products p WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.product_id = p.id AND r.rating = 5)",
    "SELECT u.id FROM users u WHERE EXISTS (SELECT 1 FROM purchases p WHERE p.user_id = u.id AND p.amount > 100)",
    "SELECT name FROM artist a WHERE EXISTS (SELECT 1 FROM album al WHERE al.artist_id = a.id)",
]

WINDOW_CASES: list[str] = [
    "SELECT name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rk FROM emp",
    "SELECT id, RANK() OVER (ORDER BY score DESC) AS r FROM students",
    "SELECT symbol, dt, LAG(price, 1) OVER (PARTITION BY symbol ORDER BY dt) AS prev FROM prices",
    "SELECT customer_id, amount, SUM(amount) OVER (PARTITION BY customer_id) AS tot FROM orders",
    "SELECT name, DENSE_RANK() OVER (ORDER BY salary DESC) AS dr FROM employees",
    "SELECT id, AVG(score) OVER (PARTITION BY class_id ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM tests",
]

CTE_CASES: list[str] = [
    "WITH high_earners AS (SELECT * FROM emp WHERE salary > 100000) SELECT * FROM high_earners",
    "WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT * FROM a JOIN b ON a.x = b.y",
    "WITH RECURSIVE cnt(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM cnt WHERE n < 10) SELECT * FROM cnt",
    "WITH dept_avg AS (SELECT dept, AVG(salary) AS avg_sal FROM emp GROUP BY dept) SELECT * FROM dept_avg",
    "WITH t AS (SELECT id, name FROM products) SELECT * FROM t WHERE id < 100",
    "WITH ranked AS (SELECT id FROM x ORDER BY id) SELECT * FROM ranked",
]


@pytest.mark.parametrize("sql", PLAIN_CASES)
def test_plain(sql: str) -> None:
    assert classify_construct(sql) == "plain"


@pytest.mark.parametrize("sql", HAVING_CASES)
def test_having(sql: str) -> None:
    assert classify_construct(sql) == "HAVING"


@pytest.mark.parametrize("sql", EXISTS_CASES)
def test_exists(sql: str) -> None:
    assert classify_construct(sql) == "EXISTS"


@pytest.mark.parametrize("sql", WINDOW_CASES)
def test_window(sql: str) -> None:
    assert classify_construct(sql) == "Window"


@pytest.mark.parametrize("sql", CTE_CASES)
def test_cte(sql: str) -> None:
    assert classify_construct(sql) == "CTE"


# ---------------------------------------------------------------------------
# Priority resolution (>= 3 cases)
# ---------------------------------------------------------------------------


def test_priority_cte_beats_having() -> None:
    sql = (
        "WITH stats AS ("
        "  SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept HAVING COUNT(*) > 5"
        ") "
        "SELECT * FROM stats"
    )
    assert classify_construct(sql) == "CTE"


def test_priority_window_beats_having() -> None:
    sql = (
        "SELECT dept, COUNT(*) AS n, "
        "ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rk "
        "FROM emp GROUP BY dept HAVING COUNT(*) > 5"
    )
    assert classify_construct(sql) == "Window"


def test_priority_exists_beats_having() -> None:
    sql = (
        "SELECT dept, COUNT(*) FROM emp "
        "WHERE EXISTS (SELECT 1 FROM dependents d WHERE d.emp_id = emp.id) "
        "GROUP BY dept HAVING COUNT(*) > 1"
    )
    assert classify_construct(sql) == "EXISTS"


def test_priority_cte_beats_window_and_exists() -> None:
    sql = (
        "WITH t AS ("
        "  SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rk FROM x "
        "  WHERE EXISTS (SELECT 1 FROM y WHERE y.id = x.id)"
        ") "
        "SELECT * FROM t"
    )
    assert classify_construct(sql) == "CTE"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string_returns_plain() -> None:
    label, parsed = classify_construct_with_meta("")
    assert label == "plain"
    assert parsed is False


def test_whitespace_only_returns_plain() -> None:
    label, parsed = classify_construct_with_meta("   \n  \t  ")
    assert label == "plain"
    assert parsed is False


def test_malformed_sql_falls_back_to_plain() -> None:
    label, parsed = classify_construct_with_meta("SELEKT *** WHERE !!! garbage")
    assert label == "plain"
    assert parsed is False


def test_inline_comment_is_handled() -> None:
    sql = "SELECT name -- comment line\nFROM users WHERE age > 18"
    assert classify_construct(sql) == "plain"


def test_having_in_subquery_is_detected() -> None:
    sql = (
        "SELECT * FROM ("
        "  SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept HAVING COUNT(*) > 5"
        ") AS sub"
    )
    assert classify_construct(sql) == "HAVING"


def test_custom_priority_overrides_default() -> None:
    sql = (
        "WITH t AS (SELECT * FROM emp) "
        "SELECT dept, COUNT(*) FROM t GROUP BY dept HAVING COUNT(*) > 1"
    )
    # Promote HAVING above CTE so it wins.
    label = classify_construct(sql, priority=("HAVING", "CTE", "Window", "EXISTS", "plain"))
    assert label == "HAVING"


def test_unknown_priority_label_raises() -> None:
    with pytest.raises(ValueError):
        classify_construct("SELECT 1", priority=("CTE", "BOGUS"))


def test_constants_match_spec() -> None:
    assert CONSTRUCT_LABELS == ("plain", "HAVING", "EXISTS", "Window", "CTE")
    assert DEFAULT_PRIORITY == ("CTE", "Window", "EXISTS", "HAVING", "plain")


def test_parsed_ok_flag_is_true_for_valid_sql() -> None:
    _, parsed = classify_construct_with_meta("SELECT 1 FROM t")
    assert parsed is True
