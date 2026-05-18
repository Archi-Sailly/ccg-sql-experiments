"""Phase 5 — Data preprocessing for SFT.

Convert raw labeled examples into uniform (prompt, completion) pairs suitable
for HuggingFace TRL SFTTrainer.

Prompt template (configurable via configs/default.yaml):

    Given the following database schema and question, generate a SQL query.

    Schema: {schema}

    {few_shot_examples}

    Question: {question}
    SQL:

Source handling:
* BIRD examples — schema fetched from <db_id>.sqlite via Phase 1 bird_loader.
* SynSQL examples — schema usually carried inline (db_schema / sql_context).
* Gretel examples — schema in sql_context.
* If schema cannot be obtained, the placeholder "[schema unavailable]" is used
  and the example is still kept (helpful as language modelling signal).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ─── Schema extraction ────────────────────────────────────
def _try_get_schema_inline(ex: dict[str, Any]) -> str | None:
    for k in ("db_schema", "schema", "sql_context", "create_statements", "context"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v and isinstance(v[0], str):
            return "\n".join(v).strip()
    return None


def get_schema(
    ex: dict[str, Any],
    bird_train_root: Path | None = None,
    bird_dev_root: Path | None = None,
    cache: dict[str, str] | None = None,
) -> str:
    """Return CREATE TABLE statements for an example, or '[schema unavailable]'."""
    inline = _try_get_schema_inline(ex)
    if inline:
        return inline

    source = ex.get("source", "BIRD")
    db_id = ex.get("db_id", "")
    if source == "BIRD" and db_id and bird_train_root:
        # cache lookup
        if cache is not None and db_id in cache:
            return cache[db_id]
        # try train then dev
        for root in (bird_train_root, bird_dev_root):
            if root is None:
                continue
            candidates = list(root.rglob(f"{db_id}.sqlite"))
            if candidates:
                import sqlite3
                try:
                    conn = sqlite3.connect(str(candidates[0]))
                    rows = conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
                    ).fetchall()
                    conn.close()
                    schema = "\n".join(r[0] for r in rows if r[0]).strip() or "[schema unavailable]"
                    if cache is not None:
                        cache[db_id] = schema
                    return schema
                except Exception as e:
                    logger.warning("sqlite read failed for %s: %s", db_id, e)
    return "[schema unavailable]"


def get_question(ex: dict[str, Any]) -> str:
    for k in ("question", "sql_prompt", "natural_language_query", "prompt"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def get_sql(ex: dict[str, Any]) -> str:
    for k in ("SQL", "sql", "query", "gold_sql"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ─── Prompt construction ──────────────────────────────────
DEFAULT_TEMPLATE = (
    "Given the following database schema and question, generate a SQL query.\n\n"
    "Schema: {schema}\n\n"
    "{few_shot_examples}"
    "Question: {question}\n"
    "SQL:"
)


def format_few_shot(examples: list[dict[str, Any]]) -> str:
    """Format few-shot examples into the prompt."""
    if not examples:
        return ""
    parts = []
    for ex in examples:
        q = get_question(ex)
        s = get_sql(ex)
        if not q or not s:
            continue
        parts.append(f"Question: {q}\nSQL: {s}")
    if not parts:
        return ""
    return "Examples:\n" + "\n\n".join(parts) + "\n\n"


def build_prompt(
    ex: dict[str, Any],
    template: str,
    few_shots: list[dict[str, Any]] | None = None,
    bird_train_root: Path | None = None,
    bird_dev_root: Path | None = None,
    schema_cache: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build (prompt, completion) for an example."""
    schema = get_schema(ex, bird_train_root, bird_dev_root, cache=schema_cache)
    question = get_question(ex)
    sql = get_sql(ex)
    few_shot_str = format_few_shot(few_shots or [])
    prompt = template.format(
        schema=schema,
        few_shot_examples=few_shot_str,
        question=question,
    )
    return {
        "prompt": prompt,
        "completion": " " + sql,           # leading space for tokenizer
        "target_construct": ex.get("target_construct", "plain"),
        "db_id": ex.get("db_id", ""),
        "source": ex.get("source", "BIRD"),
    }


# ─── Few-shot pool selection ──────────────────────────────
def select_few_shots(
    target_construct: str,
    pool_by_construct: dict[str, list[dict[str, Any]]],
    k: int = 3,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Pick k few-shot examples that share the same target_construct."""
    candidates = pool_by_construct.get(target_construct, [])
    if not candidates:
        # fallback to plain
        candidates = pool_by_construct.get("plain", [])
    if not candidates:
        return []
    rng = random.Random(seed)
    return rng.sample(candidates, min(k, len(candidates)))


# ─── CLI ──────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--train-data", type=Path,
                        default=Path("data/combined_train_labeled.json"),
                        help="Source labeled file")
    parser.add_argument("--out", type=Path,
                        default=Path("data/sft_train.jsonl"),
                        help="Output JSONL with (prompt, completion) per line")
    parser.add_argument("--n-shots", type=int, default=3,
                        help="few-shot examples per prompt (0 to disable)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional limit on examples (0 = use all)")
    parser.add_argument("--bird-train-db",
                        default="data/bird/train/train_databases",
                        type=Path)
    parser.add_argument("--bird-dev-db",
                        default="data/bird/dev/dev_databases",
                        type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    template = config["sft"]["prompt_template"]

    # Resolve nested BIRD layouts (rglob)
    def resolve(root, name):
        if root.exists():
            return root
        found = next(iter(root.parent.rglob(name)), None) if root.parent.exists() else None
        return found if found else root
    bird_train_root = resolve(args.bird_train_db, "train_databases")
    bird_dev_root   = resolve(args.bird_dev_db,   "dev_databases")
    logger.info("BIRD train DB root: %s (exists=%s)", bird_train_root, bird_train_root.exists())
    logger.info("BIRD dev DB root  : %s (exists=%s)", bird_dev_root,   bird_dev_root.exists())

    train_data = json.loads(args.train_data.read_text(encoding="utf-8"))
    if args.limit > 0:
        train_data = train_data[: args.limit]
    logger.info("Loaded %d raw examples from %s", len(train_data), args.train_data)

    # Build few-shot pool (by construct)
    pool_by_construct = defaultdict(list)
    if args.n_shots > 0:
        # Pool from BIRD-only subset to keep schema/style consistent
        for ex in train_data:
            if ex.get("source", "BIRD") == "BIRD":
                pool_by_construct[ex.get("target_construct", "plain")].append(ex)
        logger.info("Few-shot pool sizes: %s",
                    {k: len(v) for k, v in pool_by_construct.items()})

    # Build prompts
    schema_cache: dict[str, str] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    src_counter: Counter[str] = Counter()
    cls_counter: Counter[str] = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(train_data):
            target = ex.get("target_construct", "plain")
            few_shots = select_few_shots(
                target, pool_by_construct, k=args.n_shots, seed=args.seed + i,
            ) if args.n_shots > 0 else []
            # exclude itself if accidentally selected (BIRD case)
            few_shots = [fs for fs in few_shots if get_question(fs) != get_question(ex)]
            rec = build_prompt(
                ex, template=template, few_shots=few_shots,
                bird_train_root=bird_train_root,
                bird_dev_root=bird_dev_root,
                schema_cache=schema_cache,
            )
            if not rec["prompt"] or not rec["completion"].strip():
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            src_counter[rec["source"]] += 1
            cls_counter[rec["target_construct"]] += 1
            if (i + 1) % 500 == 0:
                logger.info("  built %d / %d", i + 1, len(train_data))

    logger.info("Wrote %d prompts to %s", written, args.out)
    logger.info("Sources: %s", dict(src_counter))
    logger.info("Constructs: %s", dict(cls_counter))


if __name__ == "__main__":
    main()
