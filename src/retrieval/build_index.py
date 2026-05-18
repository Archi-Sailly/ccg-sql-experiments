"""Phase 3 — 2-stage retrieval index builder.

Stage 1: BM25 lexical recall (top-K_stage1, default 500)
Stage 2: faiss semantic re-rank + hybrid score (top-K_final, default 8)

Hybrid score:
    s_total = semantic_weight * cos_sim + lexical_weight * normalized_bm25
    (+ alpha_cm bonus for matching target_construct — applied at query time, not here)

Outputs:
    data/retrieval_corpus.json   — list of {id, question, sql, db_id, target_construct, source}
    data/retrieval_bm25.pkl      — pickled rank_bm25.BM25Okapi
    data/retrieval_faiss.index   — faiss IndexFlatIP (cosine via normalized vectors)
    data/retrieval_embeddings.npy
    results/retrieval_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ─── Config & data ────────────────────────────────────────
def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_examples(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Tokenisation for BM25 ────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenisation for BM25."""
    return _TOKEN_RE.findall(text.lower())


# ─── Corpus build ─────────────────────────────────────────
def build_corpus(
    examples: list[dict[str, Any]],
    question_keys: tuple[str, ...] = ("question", "sql_prompt", "natural_language_query"),
    sql_keys: tuple[str, ...] = ("SQL", "sql", "query"),
) -> list[dict[str, Any]]:
    """Turn raw examples into a uniform corpus list with stable ids."""
    corpus = []
    for i, ex in enumerate(examples):
        q = ""
        for k in question_keys:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                q = v.strip()
                break
        sql = ""
        for k in sql_keys:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                sql = v.strip()
                break
        if not q:
            continue
        corpus.append({
            "id": i,
            "question": q,
            "sql": sql,
            "db_id": ex.get("db_id", "") or ex.get("database", ""),
            "target_construct": ex.get("target_construct", "plain"),
            "source": ex.get("source", "BIRD"),
        })
    return corpus


# ─── BM25 index ───────────────────────────────────────────
def build_bm25(corpus: list[dict[str, Any]]):
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(c["question"]) for c in corpus]
    logger.info("Building BM25 over %d documents", len(tokenized))
    return BM25Okapi(tokenized)


# ─── Semantic embeddings + faiss ──────────────────────────
def encode_corpus(
    encoder_name: str,
    corpus: list[dict[str, Any]],
    batch_size: int = 64,
    cache_path: Path | None = None,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        return np.load(cache_path)
    from sentence_transformers import SentenceTransformer

    logger.info("Encoding %d questions with %s", len(corpus), encoder_name)
    model = SentenceTransformer(encoder_name)
    t0 = time.time()
    emb = model.encode(
        [c["question"] for c in corpus],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info("Encoded in %.1fs, shape=%s", time.time() - t0, emb.shape)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb.astype("float32")


def build_faiss(embeddings: np.ndarray):
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("faiss IndexFlatIP built  ntotal=%d  dim=%d", index.ntotal, dim)
    return index


# ─── Retrieval ────────────────────────────────────────────
def retrieve(
    query: str,
    bm25,
    faiss_index,
    embeddings: np.ndarray,
    corpus: list[dict[str, Any]],
    encoder,
    stage1_top_k: int = 500,
    stage2_top_k: int = 8,
    semantic_weight: float = 0.6,
    lexical_weight: float = 0.4,
    target_construct: str | None = None,
    alpha_cm: float = 0.0,
) -> list[tuple[int, float, dict[str, Any]]]:
    """Run 2-stage retrieval. Returns [(corpus_id, score, example), ...] sorted desc."""
    # Stage 1: BM25
    bm25_scores = bm25.get_scores(tokenize(query))
    top1 = np.argsort(-bm25_scores)[:stage1_top_k]

    # Stage 2: semantic re-rank on subset
    q_emb = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    sub_emb = embeddings[top1]                            # (K1, dim)
    cos = (sub_emb @ q_emb[0])                            # (K1,)
    # Normalize BM25 to [0,1] within candidates
    bm25_sub = bm25_scores[top1]
    bm25_norm = (bm25_sub - bm25_sub.min()) / (bm25_sub.max() - bm25_sub.min() + 1e-9)
    score = semantic_weight * cos + lexical_weight * bm25_norm

    # Optional alpha_cm bonus
    if target_construct is not None and alpha_cm > 0:
        for i, idx in enumerate(top1):
            if corpus[int(idx)].get("target_construct") == target_construct:
                score[i] += alpha_cm

    order = np.argsort(-score)[:stage2_top_k]
    results = []
    for j in order:
        cid = int(top1[j])
        results.append((cid, float(score[j]), corpus[cid]))
    return results


# ─── Main ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--corpus-data",
        type=Path,
        default=None,
        help="Few-shot pool data (default: combined_train_labeled.json or bird_train_labeled.json)",
    )
    parser.add_argument("--demo", action="store_true", help="Run a small retrieve demo on BIRD dev")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    config = load_config(args.config)
    paths = config["paths"]
    pred_cfg = config["predictor"]
    retr_cfg = config["retrieval"]

    # Pick corpus source
    if args.corpus_data is not None:
        corpus_path = args.corpus_data
    else:
        combined = Path("data/combined_train_labeled.json")
        corpus_path = combined if combined.exists() else Path(paths["labeled_train"])
    logger.info("Corpus source: %s", corpus_path)

    raw = load_examples(corpus_path)
    corpus = build_corpus(raw)
    logger.info("Built corpus with %d items", len(corpus))

    # Save corpus to disk (compact form)
    corpus_out = Path("data/retrieval_corpus.json")
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    logger.info("Saved corpus to %s", corpus_out)

    # Build BM25
    bm25 = build_bm25(corpus)
    bm25_path = Path("data/retrieval_bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("Saved BM25 to %s", bm25_path)

    # Encode + faiss
    embeddings = encode_corpus(
        pred_cfg["encoder"],
        corpus,
        batch_size=64,
        cache_path=Path("data/retrieval_embeddings.npy"),
    )
    faiss_index = build_faiss(embeddings)
    import faiss
    faiss_path = Path(paths["retrieval_index"])
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(faiss_path))
    logger.info("Saved faiss to %s", faiss_path)

    # Stats
    from collections import Counter
    dist = Counter(c["target_construct"] for c in corpus)
    src_dist = Counter(c["source"] for c in corpus)
    stats = {
        "corpus_path": str(corpus_path),
        "n_corpus": len(corpus),
        "encoder": pred_cfg["encoder"],
        "stage1_top_k": retr_cfg["stage1_top_k"],
        "stage2_top_k": retr_cfg["stage2_top_k"],
        "semantic_weight": retr_cfg["semantic_weight"],
        "lexical_weight": retr_cfg["lexical_weight"],
        "alpha_cm": retr_cfg["alpha_cm"],
        "construct_distribution": dict(dist),
        "source_distribution": dict(src_dist),
        "embedding_dim": int(embeddings.shape[1]),
        "artefacts": {
            "corpus": str(corpus_out),
            "bm25": str(bm25_path),
            "faiss": str(faiss_path),
            "embeddings": "data/retrieval_embeddings.npy",
        },
    }

    # Demo (optional)
    if args.demo:
        dev_path = Path(paths["labeled_dev"])
        if dev_path.exists():
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(pred_cfg["encoder"])
            dev = load_examples(dev_path)
            demo_examples = []
            for ex in dev[:3]:
                q = ex.get("question", "")
                target = ex.get("target_construct", "plain")
                hits = retrieve(
                    q, bm25, faiss_index, embeddings, corpus, encoder,
                    stage1_top_k=retr_cfg["stage1_top_k"],
                    stage2_top_k=retr_cfg["stage2_top_k"],
                    semantic_weight=retr_cfg["semantic_weight"],
                    lexical_weight=retr_cfg["lexical_weight"],
                    target_construct=target,
                    alpha_cm=retr_cfg["alpha_cm"],
                )
                demo_examples.append({
                    "query_question": q,
                    "query_target_construct": target,
                    "top_hits": [
                        {
                            "rank": rank + 1,
                            "score": float(score),
                            "question": hit["question"][:120],
                            "target_construct": hit["target_construct"],
                            "source": hit["source"],
                        }
                        for rank, (_, score, hit) in enumerate(hits)
                    ],
                })
                logger.info("=== Query: %s ... (target=%s) ===", q[:80], target)
                for rank, (_, score, hit) in enumerate(hits):
                    logger.info(
                        "  #%d  score=%.3f  cls=%s  src=%s  q=%s",
                        rank + 1, score, hit["target_construct"], hit["source"], hit["question"][:80],
                    )
            stats["demo"] = demo_examples

    # Save stats
    stats_path = Path(paths["results_dir"]) / "retrieval_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=float)
    logger.info("Saved stats to %s", stats_path)
    logger.info("Phase 3 build_index complete.")


if __name__ == "__main__":
    main()
