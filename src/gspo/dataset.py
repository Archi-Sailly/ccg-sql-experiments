"""Phase 6 — Dataset builder for GSPO RL.

Build a HuggingFace Dataset where each row carries the metadata needed to
compute the reward at training time:
    {prompt, gold_sql, db_path, target_construct, db_id}

Only BIRD train examples are used here because we need executable SQLite
databases to compute R_exec. SynSQL / Gretel synthetic rows are excluded
(no underlying DB).
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

from src.sft.data import (DEFAULT_TEMPLATE, build_prompt, get_question,
                          get_sql, select_few_shots)

logger = logging.getLogger(__name__)


def resolve_bird_db_root(args_root: Path, split_name: str) -> Path:
    """rglob to find <split>_databases under data/bird/<split>."""
    if args_root.exists():
        return args_root
    cand = next(iter(args_root.parent.rglob(f"{split_name}_databases")), None)
    return cand if cand else args_root


def build_rl_dataset(
    train_data: list[dict[str, Any]],
    bird_train_root: Path,
    template: str,
    n_shots: int = 3,
    target_per_class: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build RL dataset rows. Skips examples whose DB sqlite file isn't found."""
    rng = random.Random(seed)
    # Filter BIRD only
    bird_only = [ex for ex in train_data if ex.get("source", "BIRD") == "BIRD"]
    logger.info("BIRD-only examples: %d / %d", len(bird_only), len(train_data))

    # Build few-shot pool (BIRD)
    pool_by_construct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if n_shots > 0:
        for ex in bird_only:
            pool_by_construct[ex.get("target_construct", "plain")].append(ex)

    # Group examples by target_construct for stratified sampling
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in bird_only:
        by_class[ex.get("target_construct", "plain")].append(ex)

    # If target_per_class given: stratified sample
    if target_per_class is not None:
        sampled = []
        for cls, want in target_per_class.items():
            avail = by_class.get(cls, [])
            if not avail:
                logger.warning("Class %s has 0 BIRD examples — skipping", cls)
                continue
            n = min(want, len(avail))
            sampled.extend(rng.sample(avail, n))
        rng.shuffle(sampled)
        bird_only = sampled

    schema_cache: dict[str, str] = {}
    rl_rows = []
    missing_db = 0
    for i, ex in enumerate(bird_only):
        db_id = ex.get("db_id", "")
        if not db_id:
            continue
        db_path = bird_train_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            # rglob fallback in case of deeper nesting
            cand = next(iter(bird_train_root.rglob(f"{db_id}.sqlite")), None)
            if cand is None:
                missing_db += 1
                continue
            db_path = cand

        target = ex.get("target_construct", "plain")
        few = select_few_shots(
            target, pool_by_construct, k=n_shots, seed=seed + i
        ) if n_shots > 0 else []
        few = [fs for fs in few if get_question(fs) != get_question(ex)]

        rec = build_prompt(
            ex, template=template, few_shots=few,
            bird_train_root=bird_train_root,
            bird_dev_root=None,
            schema_cache=schema_cache,
        )
        if not rec["prompt"]:
            continue
        rl_rows.append({
            "prompt": rec["prompt"],
            "gold_sql": get_sql(ex),
            "db_path": str(db_path),
            "target_construct": target,
            "db_id": db_id,
        })

    logger.info("Built %d RL rows  (missing DB: %d)", len(rl_rows), missing_db)
    dist = Counter(r["target_construct"] for r in rl_rows)
    logger.info("Construct distribution: %s", dict(dist))
    return rl_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--train-data", type=Path,
                        default=Path("data/combined_train_labeled.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/gspo_train.jsonl"))
    parser.add_argument("--bird-train-db",
                        default="data/bird/train/train_databases",
                        type=Path)
    parser.add_argument("--n-shots", type=int, default=2,
                        help="few-shot examples per prompt (RL: smaller is faster)")
    parser.add_argument("--n-per-class", type=int, default=200,
                        help="examples per construct class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    template = config["sft"]["prompt_template"]
    labels = config["data"]["construct_labels"]
    target_per_class = {lab: args.n_per_class for lab in labels}

    bird_train_root = resolve_bird_db_root(args.bird_train_db, "train")
    logger.info("BIRD train DB root: %s", bird_train_root)

    train_data = json.loads(args.train_data.read_text(encoding="utf-8"))
    logger.info("Loaded %d raw examples", len(train_data))

    rl_rows = build_rl_dataset(
        train_data, bird_train_root, template,
        n_shots=args.n_shots,
        target_per_class=target_per_class,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(rl_rows), args.out)


if __name__ == "__main__":
    main()
