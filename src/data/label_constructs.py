"""Phase 1 CLI — label the BIRD train/dev splits with 5-class constructs.

Reads the YAML config (default ``configs/default.yaml``), loads each
BIRD split via :mod:`src.data.bird_loader`, classifies every example
with :func:`src.data.labeling.classify_construct_with_meta`, and writes:

* ``data/bird_train_labeled.json`` — train examples plus ``target_construct``.
* ``data/bird_dev_labeled.json``   — dev   examples plus ``target_construct``.
* ``results/labeling_stats.json``  — per-split distribution + parse-failure rate.

Usage::

    python -m src.data.label_constructs --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.data.bird_loader import load_bird_split
from src.data.labeling import CONSTRUCT_LABELS, classify_construct_with_meta

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dict.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_sql(example: dict[str, Any]) -> str:
    """Return the SQL string from a BIRD example, accepting common key variants."""
    for key in ("SQL", "sql", "query", "gold_sql"):
        value = example.get(key)
        if isinstance(value, str):
            return value
    return ""


def label_split(
    examples: list[dict[str, Any]],
    priority: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add ``target_construct`` to each example and compute summary stats.

    Args:
        examples: Raw BIRD examples (as returned by ``load_bird_split``).
        priority: Construct priority order, highest first.

    Returns:
        ``(labeled_examples, stats)`` where ``stats`` has keys ``total``,
        ``distribution``, ``fractions``, and ``parse_failure_rate``.
    """
    labeled: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    parse_failures = 0

    for example in tqdm(examples, desc="labeling", unit="ex"):
        sql = _extract_sql(example)
        label, parsed_ok = classify_construct_with_meta(sql, priority=priority)
        if not parsed_ok:
            parse_failures += 1
        enriched = dict(example)
        enriched["target_construct"] = label
        labeled.append(enriched)
        counter[label] += 1

    total = len(labeled) or 1
    stats: dict[str, Any] = {
        "total": len(labeled),
        "distribution": {k: counter.get(k, 0) for k in CONSTRUCT_LABELS},
        "fractions": {k: counter.get(k, 0) / total for k in CONSTRUCT_LABELS},
        "parse_failure_rate": parse_failures / total,
    }
    return labeled, stats


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` to ``path`` as UTF-8 JSON, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _log_stats(split: str, stats: dict[str, Any]) -> None:
    """Pretty-print labeling stats for a split via the logger."""
    logger.info("  total: %d", stats["total"])
    for label in CONSTRUCT_LABELS:
        count = stats["distribution"][label]
        frac = stats["fractions"][label]
        logger.info("    %-7s %5d (%.2f%%)", label, count, 100 * frac)
    logger.info("  parse_failure_rate: %.4f", stats["parse_failure_rate"])
    if stats["parse_failure_rate"] >= 0.05:
        logger.warning(
            "[%s] parse failure rate %.2f%% exceeds 5%% threshold",
            split,
            100 * stats["parse_failure_rate"],
        )


def run(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Run labeling for both train and dev splits.

    Args:
        config: Loaded YAML configuration dict.

    Returns:
        Mapping of split name -> stats dict.
    """
    paths = config["paths"]
    data_dir = Path(paths["data_dir"]) / "bird"
    priority = list(config["data"]["priority"])

    all_stats: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        logger.info("=== Labeling %s split ===", split)
        examples = load_bird_split(split, data_dir)
        labeled, stats = label_split(examples, priority=priority)

        out_path = Path(paths[f"labeled_{split}"])
        _write_json(out_path, labeled)
        logger.info("Wrote %d labeled examples to %s", len(labeled), out_path)
        _log_stats(split, stats)
        all_stats[split] = stats

    stats_path = Path(paths["results_dir"]) / "labeling_stats.json"
    _write_json(stats_path, all_stats)
    logger.info("Wrote summary stats to %s", stats_path)
    return all_stats


def main() -> None:
    """CLI entry point for ``make label-data``."""
    parser = argparse.ArgumentParser(
        description="Phase 1 — sqlglot AST 기반 BIRD 5-class 라벨링",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config path (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
