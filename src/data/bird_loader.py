"""BIRD benchmark loader and SQLite schema extractor.

The BIRD download (see ``scripts/download_bird.sh``) yields a tree of
the form::

    data/bird/
        train/
            train.json
            train_databases/<db_id>/<db_id>.sqlite
        dev/
            dev.json
            dev_databases/<db_id>/<db_id>.sqlite

Real downloads occasionally nest these directories one level deeper, so
the helpers below use :py:meth:`pathlib.Path.rglob` to locate the
canonical files without hard-coding the layout.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_SPLITS = ("train", "dev")


def _find_split_json(split: str, data_dir: Path) -> Path:
    """Locate the ``<split>.json`` file under ``data_dir/<split>``.

    Args:
        split: ``"train"`` or ``"dev"``.
        data_dir: Root of the BIRD download (``./data/bird`` by default).

    Returns:
        Absolute path to the BIRD JSON for the requested split.

    Raises:
        FileNotFoundError: If no ``<split>.json`` exists under
            ``data_dir/<split>``.
    """
    search_root = data_dir / split
    if not search_root.exists():
        raise FileNotFoundError(
            f"{search_root} does not exist. Run `bash scripts/download_bird.sh` first."
        )
    candidates = sorted(search_root.rglob(f"{split}.json"), key=lambda p: len(p.parts))
    if not candidates:
        raise FileNotFoundError(
            f"{split}.json not found under {search_root}. "
            "Did `scripts/download_bird.sh` complete successfully?"
        )
    return candidates[0]


def load_bird_split(
    split: str,
    data_dir: Path = Path("./data/bird"),
) -> list[dict]:
    """Load the BIRD ``train`` or ``dev`` split.

    Args:
        split: One of ``"train"`` or ``"dev"``.
        data_dir: Root of the BIRD download. Defaults to ``./data/bird``.

    Returns:
        List of raw example dicts as shipped by BIRD. Each item
        typically has the keys ``db_id``, ``question``, ``SQL``,
        ``evidence``, and ``difficulty``.

    Raises:
        ValueError: If ``split`` is not ``"train"`` or ``"dev"``.
        FileNotFoundError: If the split JSON cannot be located.

    Example:
        >>> examples = load_bird_split("dev")  # doctest: +SKIP
        >>> examples[0]["db_id"]               # doctest: +SKIP
        'california_schools'
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    path = _find_split_json(split, Path(data_dir))
    with open(path, encoding="utf-8") as f:
        examples = json.load(f)
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def _databases_root(split: str, data_dir: Path) -> Path:
    """Return the directory containing per-db_id folders for a split."""
    base = Path(data_dir) / split
    for name in (f"{split}_databases", "databases"):
        candidate = base / name
        if candidate.is_dir():
            return candidate
    nested = next(base.rglob(f"{split}_databases"), None)
    if nested is not None and nested.is_dir():
        return nested
    raise FileNotFoundError(
        f"Could not locate `{split}_databases/` under {base}. "
        "Re-run `bash scripts/download_bird.sh`."
    )


def extract_schema(
    db_id: str,
    split: str,
    data_dir: Path = Path("./data/bird"),
) -> str:
    """Return concatenated ``CREATE TABLE`` statements for a BIRD database.

    Args:
        db_id: BIRD database identifier (e.g. ``"california_schools"``).
        split: ``"train"`` or ``"dev"``. Used to choose the right
            ``<split>_databases/`` root.
        data_dir: Root of the BIRD download.

    Returns:
        Newline-joined ``CREATE TABLE`` DDL strings, ready to embed in a
        prompt. Tables with no stored ``sql`` (rare) are skipped.

    Raises:
        FileNotFoundError: If the database folder or ``.sqlite`` file is
            missing.

    Example:
        >>> ddl = extract_schema("california_schools", "dev")  # doctest: +SKIP
        >>> ddl.startswith("CREATE TABLE")                       # doctest: +SKIP
        True
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    db_dir = _databases_root(split, Path(data_dir)) / db_id
    if not db_dir.is_dir():
        raise FileNotFoundError(f"database folder not found: {db_dir}")
    sqlite_files = list(db_dir.glob("*.sqlite"))
    if not sqlite_files:
        raise FileNotFoundError(f"no .sqlite file under {db_dir}")

    con = sqlite3.connect(str(sqlite_files[0]))
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(row[0].strip() for row in rows if row[0])
