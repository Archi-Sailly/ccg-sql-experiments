"""Phase 1 — Construct labeling via sqlglot AST.

This module maps a SQL query to one of five mutually exclusive *constructs*:

    plain, HAVING, EXISTS, Window, CTE

When a query exhibits multiple constructs at once (e.g. a CTE that itself
contains a HAVING clause), the highest-priority class in
``DEFAULT_PRIORITY`` is returned. The default priority matches the
project specification in ``configs/default.yaml``::

    CTE > Window > EXISTS > HAVING > plain

The classifier is intentionally robust to malformed SQL: any
``sqlglot`` parse failure falls back to ``"plain"`` so that batch
labeling of the BIRD corpus never raises.
"""

from __future__ import annotations

import logging
from typing import Iterable

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

CONSTRUCT_LABELS: tuple[str, ...] = ("plain", "HAVING", "EXISTS", "Window", "CTE")
DEFAULT_PRIORITY: tuple[str, ...] = ("CTE", "Window", "EXISTS", "HAVING", "plain")

_CONSTRUCT_TO_AST: dict[str, type[exp.Expression]] = {
    "CTE": exp.With,
    "Window": exp.Window,
    "EXISTS": exp.Exists,
    "HAVING": exp.Having,
}


def _validate_priority(priority: Iterable[str]) -> tuple[str, ...]:
    """Return ``priority`` as a tuple after checking for unknown labels."""
    seq = tuple(priority)
    unknown = set(seq) - set(CONSTRUCT_LABELS)
    if unknown:
        raise ValueError(
            f"unknown construct labels in priority: {sorted(unknown)}; "
            f"allowed: {list(CONSTRUCT_LABELS)}"
        )
    return seq


def classify_construct_with_meta(
    sql: str,
    priority: Iterable[str] = DEFAULT_PRIORITY,
    dialect: str = "sqlite",
) -> tuple[str, bool]:
    """Classify a SQL query and report whether parsing succeeded.

    The query is parsed with ``sqlglot`` and walked for the AST nodes
    associated with each construct (``exp.With`` for CTEs, ``exp.Window``
    for window functions, ``exp.Exists`` for EXISTS subqueries,
    ``exp.Having`` for HAVING clauses). On parse failure, the function
    returns ``("plain", False)``.

    Args:
        sql: Raw SQL query string. Empty or whitespace-only inputs are
            treated as a parse failure.
        priority: Ordered iterable of class names, highest priority first.
            Must contain only labels in :data:`CONSTRUCT_LABELS`.
            Defaults to ``("CTE", "Window", "EXISTS", "HAVING", "plain")``.
        dialect: sqlglot dialect for parsing. Defaults to ``"sqlite"``
            since BIRD queries target SQLite.

    Returns:
        A ``(label, parsed_ok)`` tuple where ``label`` is one of
        ``"CTE" | "Window" | "EXISTS" | "HAVING" | "plain"`` and
        ``parsed_ok`` indicates a successful ``sqlglot.parse_one`` call.

    Example:
        >>> classify_construct_with_meta("SELECT 1")
        ('plain', True)
        >>> classify_construct_with_meta("SELEKT ???")
        ('plain', False)
    """
    priority = _validate_priority(priority)

    if not sql or not sql.strip():
        return ("plain", False)

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as err:  # noqa: BLE001 — sqlglot raises many leaf types
        logger.debug("sqlglot parse failure (%s): %s", type(err).__name__, err)
        return ("plain", False)

    if tree is None:
        return ("plain", False)

    for label in priority:
        if label == "plain":
            return ("plain", True)
        ast_cls = _CONSTRUCT_TO_AST.get(label)
        if ast_cls is None:
            continue
        if next(tree.find_all(ast_cls), None) is not None:
            return (label, True)
    return ("plain", True)


def classify_construct(
    sql: str,
    priority: Iterable[str] = DEFAULT_PRIORITY,
    dialect: str = "sqlite",
) -> str:
    """Return the construct label for a SQL query.

    Thin wrapper around :func:`classify_construct_with_meta` that drops
    the parse-success flag. Use this as the canonical entry point from
    Phase 2+ modules that only need the label.

    Args:
        sql: Raw SQL query string.
        priority: Ordered iterable of class names, highest priority first.
        dialect: sqlglot dialect for parsing.

    Returns:
        One of ``"CTE" | "Window" | "EXISTS" | "HAVING" | "plain"``.

    Example:
        >>> classify_construct("WITH x AS (SELECT 1) SELECT * FROM x")
        'CTE'
        >>> classify_construct(
        ...     "SELECT d, COUNT(*) FROM t GROUP BY d HAVING COUNT(*) > 5"
        ... )
        'HAVING'
    """
    return classify_construct_with_meta(sql, priority=priority, dialect=dialect)[0]
