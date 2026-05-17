"""Phase 1 — Data Layer.

Public API:

* :func:`classify_construct` — sqlglot AST → one of plain/HAVING/EXISTS/Window/CTE.
* :func:`classify_construct_with_meta` — same, plus a parse-success flag.
* :func:`load_bird_split` — load BIRD train/dev JSON.
* :func:`extract_schema` — read CREATE TABLE DDL for a BIRD database.
"""

from src.data.bird_loader import extract_schema, load_bird_split
from src.data.labeling import (
    CONSTRUCT_LABELS,
    DEFAULT_PRIORITY,
    classify_construct,
    classify_construct_with_meta,
)

__all__ = [
    "CONSTRUCT_LABELS",
    "DEFAULT_PRIORITY",
    "classify_construct",
    "classify_construct_with_meta",
    "extract_schema",
    "load_bird_split",
]
