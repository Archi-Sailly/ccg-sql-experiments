"""Phase 4 — Reward functions for GSPO RL.

Total reward:
    R_total = R_exec + lambda_construct * R_construct

* R_exec      : execute generated SQL against the BIRD SQLite DB and
                compare row-sets with the gold SQL output (1.0 if match, 0.0 otherwise)
* R_construct : 1.0 if the generated SQL belongs to the target construct class,
                else 0.0 (uses Phase 1 sqlglot 5-class labeller)

CLI sanity test:
    python -m src.reward.reward --sanity-test --train-data data/bird_train_labeled.json \\
        --bird-root data/bird/train/train_databases
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sqlite3
from pathlib import Path
from typing import Any

from src.data.labeling import classify_construct_with_meta

logger = logging.getLogger(__name__)


# ─── Execution ────────────────────────────────────────────
def _run_sql_worker(db_path: str, sql: str, queue):
    """Helper for timed sqlite execution in a subprocess."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        queue.put(("ok", rows))
    except Exception as e:
        queue.put(("err", repr(e)))


def execute_sql(
    sql: str,
    db_path: str,
    timeout: float = 30.0,
) -> tuple[list[tuple] | None, str | None]:
    """Execute SQL and return (rows, error_string). On timeout/error rows is None."""
    if not sql or not sql.strip():
        return None, "empty_sql"
    ctx = mp.get_context("spawn") if mp.get_start_method(allow_none=True) != "fork" else mp
    q = ctx.Queue()
    proc = ctx.Process(target=_run_sql_worker, args=(db_path, sql, q))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        return None, "timeout"
    try:
        status, payload = q.get_nowait()
    except Exception:
        return None, "no_result"
    if status == "ok":
        return payload, None
    return None, payload


def execute_sql_inproc(sql: str, db_path: str) -> tuple[list[tuple] | None, str | None]:
    """In-process variant (no timeout). For unit tests / very fast queries."""
    if not sql or not sql.strip():
        return None, "empty_sql"
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        rows = conn.cursor().execute(sql).fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return None, repr(e)


# ─── Row comparison ────────────────────────────────────────
def _normalize(row: tuple) -> tuple:
    """Normalize each cell — floats round to 4dp, None and NaN to None."""
    out = []
    for v in row:
        if isinstance(v, float):
            out.append(round(v, 4))
        elif isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(v)
    return tuple(out)


def rows_match(
    gold_rows: list[tuple] | None,
    pred_rows: list[tuple] | None,
    ordered: bool = False,
) -> bool:
    """Compare two result sets. Default = set equality (BIRD convention)."""
    if gold_rows is None or pred_rows is None:
        return False
    g = [_normalize(r) for r in gold_rows]
    p = [_normalize(r) for r in pred_rows]
    if ordered:
        return g == p
    # Use multisets to allow duplicate rows
    from collections import Counter
    return Counter(g) == Counter(p)


# ─── Construct check ──────────────────────────────────────
def check_construct(sql: str, target_construct: str) -> bool:
    """True iff sql's classified construct (via Phase 1 labeller) matches target."""
    cls, ok = classify_construct_with_meta(sql)
    if not ok:
        return False
    return cls == target_construct


# ─── Combined reward ──────────────────────────────────────
def compute_reward(
    gen_sql: str,
    gold_sql: str,
    db_path: str,
    target_construct: str,
    lambda_construct: float = 0.2,
    timeout: float = 30.0,
    use_subprocess: bool = True,
) -> dict[str, Any]:
    """Compute R_total = R_exec + lambda * R_construct for a single sample."""
    if use_subprocess:
        pred_rows, pred_err  = execute_sql(gen_sql,  db_path, timeout=timeout)
        gold_rows, gold_err  = execute_sql(gold_sql, db_path, timeout=timeout)
    else:
        pred_rows, pred_err  = execute_sql_inproc(gen_sql,  db_path)
        gold_rows, gold_err  = execute_sql_inproc(gold_sql, db_path)

    r_exec = 1.0 if rows_match(gold_rows, pred_rows) else 0.0
    r_construct = 1.0 if check_construct(gen_sql, target_construct) else 0.0
    r_total = r_exec + lambda_construct * r_construct
    return {
        "r_total": r_total,
        "r_exec": r_exec,
        "r_construct": r_construct,
        "lambda_construct": lambda_construct,
        "pred_error": pred_err,
        "gold_error": gold_err,
    }


# ─── CLI sanity test ──────────────────────────────────────
def _sanity_test(train_path: Path, bird_root: Path, n: int = 20) -> None:
    """Sanity check — feeding gold SQL as both gen and gold should yield R_exec=1 always."""
    examples = json.loads(train_path.read_text(encoding="utf-8"))
    sampled = examples[:n]
    ok, fail = 0, 0
    for ex in sampled:
        db_id = ex.get("db_id", "")
        gold = ex.get("SQL") or ex.get("sql") or ""
        if not (db_id and gold):
            continue
        db_path = bird_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            logger.warning("DB not found: %s", db_path)
            continue
        target = ex.get("target_construct", "plain")
        result = compute_reward(
            gen_sql=gold, gold_sql=gold,
            db_path=str(db_path), target_construct=target,
            use_subprocess=False,  # in-proc is faster for sanity
        )
        if result["r_exec"] == 1.0 and result["r_construct"] == 1.0:
            ok += 1
        else:
            fail += 1
            logger.warning("FAIL: db=%s gold=%s r_exec=%s r_construct=%s pred_err=%s",
                           db_id, gold[:80], result["r_exec"], result["r_construct"],
                           result["pred_error"])
    logger.info("Sanity test:  OK=%d  FAIL=%d  (gold→gold should always give r_exec=r_construct=1)",
                ok, fail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-test", action="store_true", help="Run gold→gold sanity check")
    parser.add_argument("--train-data", type=Path,
                        default=Path("data/bird_train_labeled.json"))
    parser.add_argument("--bird-root",  type=Path,
                        default=Path("data/bird/train/train_databases"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    if args.sanity_test:
        # rglob to handle nested layouts
        candidates = list(args.bird_root.parent.rglob("train_databases"))
        if not args.bird_root.exists() and candidates:
            args.bird_root = candidates[0]
            logger.info("Resolved bird_root via rglob: %s", args.bird_root)
        _sanity_test(args.train_data, args.bird_root, n=args.n)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
