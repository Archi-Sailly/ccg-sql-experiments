"""Phase 7 — K=8 candidate selection (reranker).

Two strategies:
* argmax       — pick the candidate with highest log-prob (or first if log-probs unavailable)
* weighted_vote — execute all K candidates against the DB; cluster by row-set;
                  pick a representative of the largest cluster (majority voting).
                  Falls back to argmax when no candidate executes.

The weighted_vote strategy is typically stronger on BIRD but requires DB access.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.reward.reward import execute_sql_inproc, rows_match

logger = logging.getLogger(__name__)


def rerank_argmax(
    candidates: list[str],
    log_probs: list[float] | None = None,
) -> tuple[int, str]:
    """Pick highest-logprob candidate (or first if log_probs is None)."""
    if not candidates:
        return -1, ""
    if log_probs is None or len(log_probs) != len(candidates):
        return 0, candidates[0]
    best = max(range(len(candidates)), key=lambda i: log_probs[i])
    return best, candidates[best]


def rerank_weighted_vote(
    candidates: list[str],
    db_path: str,
    log_probs: list[float] | None = None,
) -> tuple[int, str]:
    """Cluster candidates by execution row-set; pick rep of largest cluster.

    Returns (chosen_index, chosen_sql). If no candidate executes successfully,
    falls back to argmax (or first).
    """
    if not candidates:
        return -1, ""

    # Execute each candidate
    rows_per_cand: list[list[tuple] | None] = []
    for c in candidates:
        rows, _ = execute_sql_inproc(c, db_path)
        rows_per_cand.append(rows)

    # Identify candidates that ran successfully
    ok_idx = [i for i, r in enumerate(rows_per_cand) if r is not None]
    if not ok_idx:
        return rerank_argmax(candidates, log_probs)

    # Cluster by row-set equality (multiset on normalized tuples)
    clusters: list[list[int]] = []
    for i in ok_idx:
        placed = False
        for cl in clusters:
            j = cl[0]
            if rows_match(rows_per_cand[i], rows_per_cand[j]):
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    # Largest cluster (ties → highest log_prob within)
    best_cluster = max(clusters, key=lambda cl: len(cl))
    if log_probs is not None and len(log_probs) == len(candidates):
        chosen = max(best_cluster, key=lambda i: log_probs[i])
    else:
        chosen = best_cluster[0]
    return chosen, candidates[chosen]


def rerank(
    candidates: list[str],
    db_path: str | None = None,
    log_probs: list[float] | None = None,
    strategy: str = "argmax",
) -> tuple[int, str]:
    """Top-level dispatcher."""
    if strategy == "argmax" or db_path is None:
        return rerank_argmax(candidates, log_probs)
    if strategy == "weighted_vote":
        return rerank_weighted_vote(candidates, db_path, log_probs)
    raise ValueError(f"Unknown reranker strategy: {strategy}")
