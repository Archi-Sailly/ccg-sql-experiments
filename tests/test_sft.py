"""Unit tests for src.sft.data — Phase 5 sanity checks.

Note: src.sft.train requires torch+transformers so we don't import it here.
We test only the data preprocessing module which is fast and CPU-only.
"""

from __future__ import annotations

from src.sft.data import (
    DEFAULT_TEMPLATE,
    build_prompt,
    format_few_shot,
    get_question,
    get_schema,
    get_sql,
    select_few_shots,
)


# ─── Key extraction ──────────────────────────────────────
def test_get_question_variants():
    assert get_question({"question": "abc"}) == "abc"
    assert get_question({"sql_prompt": "xyz"}) == "xyz"
    assert get_question({"natural_language_query": "qq"}) == "qq"
    assert get_question({"prompt": "p"}) == "p"
    assert get_question({}) == ""


def test_get_sql_variants():
    assert get_sql({"SQL": "SELECT 1"}) == "SELECT 1"
    assert get_sql({"sql": "SELECT 2"}) == "SELECT 2"
    assert get_sql({}) == ""


def test_get_schema_inline():
    ex = {"sql_context": "CREATE TABLE x (a INT)"}
    assert get_schema(ex) == "CREATE TABLE x (a INT)"

    ex_list = {"create_statements": ["CREATE TABLE a (id INT)", "CREATE TABLE b (id INT)"]}
    assert "CREATE TABLE a" in get_schema(ex_list)


def test_get_schema_unavailable():
    ex = {"db_id": "missing_db"}
    assert get_schema(ex) == "[schema unavailable]"


# ─── Few-shot formatting ─────────────────────────────────
def test_format_few_shot_empty():
    assert format_few_shot([]) == ""


def test_format_few_shot_basic():
    few = [
        {"question": "q1", "SQL": "select 1"},
        {"question": "q2", "SQL": "select 2"},
    ]
    out = format_few_shot(few)
    assert "Examples:" in out
    assert "q1" in out and "q2" in out
    assert "select 1" in out and "select 2" in out


# ─── Prompt build ─────────────────────────────────────────
def test_build_prompt_basic():
    ex = {
        "question": "How many users?",
        "SQL": "SELECT COUNT(*) FROM users",
        "sql_context": "CREATE TABLE users(id INT)",
        "target_construct": "plain",
        "db_id": "x", "source": "Gretel",
    }
    rec = build_prompt(ex, template=DEFAULT_TEMPLATE, few_shots=[])
    assert "CREATE TABLE users" in rec["prompt"]
    assert "How many users?" in rec["prompt"]
    assert rec["prompt"].endswith("SQL:")
    assert rec["completion"].strip() == "SELECT COUNT(*) FROM users"
    assert rec["target_construct"] == "plain"
    assert rec["source"] == "Gretel"


def test_build_prompt_with_few_shots():
    ex = {"question": "main q", "SQL": "main sql",
          "sql_context": "CREATE TABLE t(id INT)",
          "target_construct": "plain"}
    few = [{"question": "shot q1", "SQL": "shot sql1"}]
    rec = build_prompt(ex, template=DEFAULT_TEMPLATE, few_shots=few)
    assert "shot q1" in rec["prompt"]
    assert "shot sql1" in rec["prompt"]


# ─── Few-shot selection ──────────────────────────────────
def test_select_few_shots_same_construct():
    pool = {
        "plain":  [{"question": f"p{i}", "SQL": f"s{i}"} for i in range(10)],
        "EXISTS": [{"question": f"e{i}", "SQL": f"se{i}"} for i in range(5)],
    }
    picked = select_few_shots("EXISTS", pool, k=3, seed=0)
    assert len(picked) == 3
    assert all(p["question"].startswith("e") for p in picked)


def test_select_few_shots_fallback_to_plain():
    pool = {
        "plain":   [{"question": "p", "SQL": "s"}],
        # 'CTE' not in pool
    }
    picked = select_few_shots("CTE", pool, k=3, seed=0)
    assert len(picked) == 1
    assert picked[0]["question"] == "p"


def test_select_few_shots_empty_pool():
    assert select_few_shots("CTE", {}, k=3, seed=0) == []
