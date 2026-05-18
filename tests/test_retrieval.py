"""Unit tests for src.retrieval.build_index — Phase 3 sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.build_index import (
    build_bm25,
    build_corpus,
    build_faiss,
    retrieve,
    tokenize,
)


def test_tokenize_basic():
    assert tokenize("Hello WORLD, foo_bar 42!") == ["hello", "world", "foo_bar", "42"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_build_corpus_basic():
    examples = [
        {"question": "How many users?", "SQL": "SELECT COUNT(*) FROM u", "db_id": "x",
         "target_construct": "plain"},
        {"question": "Show top dept",   "SQL": "SELECT ...", "db_id": "y",
         "target_construct": "Window", "source": "SynSQL-2.5M"},
    ]
    c = build_corpus(examples)
    assert len(c) == 2
    assert c[0]["question"] == "How many users?"
    assert c[0]["target_construct"] == "plain"
    assert c[0]["source"] == "BIRD"  # default
    assert c[1]["source"] == "SynSQL-2.5M"
    assert c[0]["id"] == 0 and c[1]["id"] == 1


def test_build_corpus_skips_empty_question():
    examples = [
        {"question": "", "SQL": "select 1"},
        {"question": "ok", "SQL": "select 2"},
    ]
    c = build_corpus(examples)
    assert len(c) == 1
    assert c[0]["question"] == "ok"


def test_build_bm25_scores():
    corpus = [
        {"id": 0, "question": "list users by name",   "sql": "", "db_id": "",
         "target_construct": "plain", "source": "BIRD"},
        {"id": 1, "question": "count orders total",   "sql": "", "db_id": "",
         "target_construct": "plain", "source": "BIRD"},
        {"id": 2, "question": "top users by spending", "sql": "", "db_id": "",
         "target_construct": "Window", "source": "BIRD"},
    ]
    bm25 = build_bm25(corpus)
    scores = bm25.get_scores(tokenize("users name"))
    # doc 0 should score highest because it contains both "users" and "name"
    assert scores.argmax() == 0


def test_build_faiss_smoke():
    rng = np.random.RandomState(0)
    emb = rng.randn(20, 16).astype("float32")
    # normalize for cosine similarity via IP
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    index = build_faiss(emb)
    assert index.ntotal == 20
    # search top-3
    q = emb[5:6]
    D, I = index.search(q, 3)
    assert I[0][0] == 5  # closest to itself


class _StubEncoder:
    """Deterministic stub encoder for unit testing retrieve() without downloading models."""

    def __init__(self, mapping):
        # mapping: question_str -> np.ndarray
        self.mapping = mapping

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        out = np.stack([self.mapping[t] for t in texts]).astype("float32")
        if normalize_embeddings:
            out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return out


def test_retrieve_hybrid_basic():
    corpus = [
        {"id": 0, "question": "list users by name",     "sql": "", "db_id": "",
         "target_construct": "plain",  "source": "BIRD"},
        {"id": 1, "question": "count orders total",     "sql": "", "db_id": "",
         "target_construct": "plain",  "source": "BIRD"},
        {"id": 2, "question": "top users by spending",  "sql": "", "db_id": "",
         "target_construct": "Window", "source": "BIRD"},
    ]
    # Manual embeddings: 3-d unit vectors aligned per topic.
    emb_map = {
        "list users by name":     np.array([1.0, 0.0, 0.0]),
        "count orders total":     np.array([0.0, 1.0, 0.0]),
        "top users by spending":  np.array([0.0, 0.0, 1.0]),
        "users":                  np.array([1.0, 0.0, 0.5]),
    }
    embeddings = np.stack([emb_map[c["question"]] for c in corpus]).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    bm25 = build_bm25(corpus)
    faiss_index = build_faiss(embeddings)
    encoder = _StubEncoder(emb_map)

    hits = retrieve(
        "users", bm25, faiss_index, embeddings, corpus, encoder,
        stage1_top_k=3, stage2_top_k=3,
        semantic_weight=0.6, lexical_weight=0.4,
    )
    assert len(hits) == 3
    # doc 0 should rank first (contains "users" lexically AND high cosine)
    assert hits[0][2]["id"] == 0


def test_retrieve_alpha_cm_bonus():
    corpus = [
        {"id": 0, "question": "list users by name",     "sql": "", "db_id": "",
         "target_construct": "plain",  "source": "BIRD"},
        {"id": 1, "question": "top users by spending",  "sql": "", "db_id": "",
         "target_construct": "Window", "source": "BIRD"},
    ]
    emb_map = {
        "list users by name":     np.array([1.0, 0.0]),
        "top users by spending":  np.array([0.9, 0.4]),  # slightly worse on cosine
        "users":                  np.array([1.0, 0.0]),
    }
    embeddings = np.stack([emb_map[c["question"]] for c in corpus]).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    bm25 = build_bm25(corpus)
    faiss_index = build_faiss(embeddings)
    encoder = _StubEncoder(emb_map)

    # Without alpha_cm bonus: doc 0 wins
    hits_no_bonus = retrieve(
        "users", bm25, faiss_index, embeddings, corpus, encoder,
        stage1_top_k=2, stage2_top_k=2,
        semantic_weight=0.6, lexical_weight=0.4,
        target_construct="Window", alpha_cm=0.0,
    )
    assert hits_no_bonus[0][2]["id"] == 0

    # With strong alpha_cm bonus and target=Window: doc 1 should jump
    hits_bonus = retrieve(
        "users", bm25, faiss_index, embeddings, corpus, encoder,
        stage1_top_k=2, stage2_top_k=2,
        semantic_weight=0.6, lexical_weight=0.4,
        target_construct="Window", alpha_cm=1.0,
    )
    assert hits_bonus[0][2]["target_construct"] == "Window"
