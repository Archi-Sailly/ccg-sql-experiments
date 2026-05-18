"""Unit tests for src.predictor.train — Phase 2 sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.predictor.train import (
    extract_questions_and_labels,
    run_cv,
    train_final,
)


# ─── extract_questions_and_labels ─────────────────────────
def test_extract_basic():
    examples = [
        {"question": "How many users?", "target_construct": "plain"},
        {"question": "Show top 3 dept",  "target_construct": "Window"},
    ]
    q, y = extract_questions_and_labels(examples)
    assert q == ["How many users?", "Show top 3 dept"]
    assert y == ["plain", "Window"]


def test_extract_alternate_key():
    examples = [
        {"sql_prompt": "List products", "target_construct": "plain"},
        {"natural_language_query": "Total sales", "target_construct": "HAVING"},
    ]
    q, y = extract_questions_and_labels(examples)
    assert q == ["List products", "Total sales"]


def test_extract_skips_empty():
    examples = [
        {"question": "ok", "target_construct": "plain"},
        {"question": "", "target_construct": "plain"},
        {"question": "no label"},
    ]
    q, y = extract_questions_and_labels(examples)
    assert q == ["ok"]
    assert y == ["plain"]


# ─── run_cv ────────────────────────────────────────────────
def _make_data(n_per_class=20, n_classes=3, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    X_parts, y_parts = [], []
    for i in range(n_classes):
        center = rng.randn(dim) * 3
        X_parts.append(center + 0.1 * rng.randn(n_per_class, dim))
        y_parts.append(np.array([f"class_{i}"] * n_per_class))
    return np.vstack(X_parts), np.concatenate(y_parts)


def test_run_cv_perfect_separation():
    X, y = _make_data(n_per_class=30, n_classes=3, dim=8, seed=1)
    labels = ["class_0", "class_1", "class_2"]
    res, oof = run_cv(X, y, labels=labels, C=1.0, n_folds=3, seed=42)
    assert res["overall_macro_f1"] > 0.95
    assert len(res["fold_results"]) == 3
    assert set(res["labels"]) == set(labels)
    assert len(oof) == len(y)


def test_run_cv_imbalanced():
    """CV runs cleanly even with class imbalance (class_weight=balanced should kick in)."""
    rng = np.random.RandomState(7)
    X = rng.randn(60, 8)
    # 50 vs 5 vs 5
    y = np.array(["A"] * 50 + ["B"] * 5 + ["C"] * 5)
    labels = ["A", "B", "C"]
    res, _ = run_cv(X, y, labels=labels, C=1.0, n_folds=5, seed=0)
    assert 0.0 <= res["overall_macro_f1"] <= 1.0
    assert all(lab in res["overall_per_class_f1"] for lab in labels)


# ─── train_final ──────────────────────────────────────────
def test_train_final_predicts():
    X, y = _make_data(n_per_class=20, n_classes=3, dim=8, seed=2)
    clf = train_final(X, y, C=1.0)
    pred = clf.predict(X[:5])
    assert len(pred) == 5
    assert all(isinstance(p, (str, np.str_)) for p in pred)
