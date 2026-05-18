"""Phase 2 — Sentence-BERT + LogisticRegression construct predictor.

Reads labeled train data, encodes questions with Sentence-BERT,
runs 5-fold stratified CV with LogisticRegression(class_weight='balanced'),
then trains a final model on all data and pickles it.

Usage:
    python -m src.predictor.train --config configs/default.yaml
    python -m src.predictor.train --train-data data/combined_train_labeled.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ─── Config & data ────────────────────────────────────────
def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_examples(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_questions_and_labels(
    examples: list[dict[str, Any]],
    question_keys: tuple[str, ...] = ("question", "sql_prompt", "natural_language_query"),
) -> tuple[list[str], list[str]]:
    questions, labels = [], []
    skipped = 0
    for ex in examples:
        q = ""
        for k in question_keys:
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                q = v
                break
        label = ex.get("target_construct")
        if not q or not label:
            skipped += 1
            continue
        questions.append(q)
        labels.append(label)
    if skipped:
        logger.warning("Skipped %d examples with empty question or label", skipped)
    return questions, labels


# ─── Embedding ────────────────────────────────────────────
def encode_questions(
    encoder_name: str,
    questions: list[str],
    batch_size: int = 64,
    cache_path: Path | None = None,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    logger.info("Encoding %d questions with %s ...", len(questions), encoder_name)
    model = SentenceTransformer(encoder_name)
    t0 = time.time()
    emb = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info("Encoded in %.1fs, shape=%s", time.time() - t0, emb.shape)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        logger.info("Cached to %s", cache_path)
    return emb


# ─── Training ─────────────────────────────────────────────
def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    C: float = 1.0,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run stratified k-fold CV. Returns (metrics_dict, oof_predictions)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.empty_like(y)
    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs", class_weight="balanced", n_jobs=-1
        )
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        oof[va_idx] = pred
        macro_f1 = f1_score(y[va_idx], pred, average="macro", labels=labels, zero_division=0)
        per_class = {
            lab: float(
                f1_score(y[va_idx], pred, labels=[lab], average="macro", zero_division=0)
            )
            for lab in labels
        }
        fold_results.append(
            {"fold": fold, "macro_f1": float(macro_f1), "per_class_f1": per_class}
        )
        logger.info(
            "Fold %d/%d  macro_F1=%.4f  per_class=%s",
            fold + 1, n_folds, macro_f1,
            {k: f"{v:.3f}" for k, v in per_class.items()},
        )

    overall_macro = float(f1_score(y, oof, average="macro", labels=labels, zero_division=0))
    overall_per_class = {
        lab: float(f1_score(y, oof, labels=[lab], average="macro", zero_division=0))
        for lab in labels
    }
    cm = confusion_matrix(y, oof, labels=labels).tolist()
    cls_report = classification_report(y, oof, labels=labels, zero_division=0, output_dict=True)

    return {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "overall_macro_f1": overall_macro,
        "overall_per_class_f1": overall_per_class,
        "confusion_matrix": cm,
        "labels": labels,
        "classification_report": cls_report,
    }, oof


def train_final(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    logger.info("Training final model on %d samples", len(X))
    clf = LogisticRegression(
        C=C, max_iter=1000, solver="lbfgs", class_weight="balanced", n_jobs=-1
    )
    clf.fit(X, y)
    return clf


# ─── Main pipeline ────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Override train data path (else use config.paths.labeled_train or combined)",
    )
    parser.add_argument(
        "--dev-data",
        type=Path,
        default=None,
        help="Optional dev set evaluation (e.g. BIRD dev) — runs zero-shot prediction",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    config = load_config(args.config)
    paths = config["paths"]
    pred_cfg = config["predictor"]
    data_cfg = config["data"]
    labels = list(data_cfg["construct_labels"])

    # Resolve train data path:
    #   1. CLI override
    #   2. data/combined_train_labeled.json (Phase 1' output, if exists)
    #   3. config.paths.labeled_train (Phase 1 BIRD-only output)
    if args.train_data is not None:
        train_path = args.train_data
    else:
        combined = Path("data/combined_train_labeled.json")
        train_path = combined if combined.exists() else Path(paths["labeled_train"])
    logger.info("Train data: %s", train_path)

    examples = load_examples(train_path)
    questions, y_list = extract_questions_and_labels(examples)
    logger.info("Loaded %d (question, label) pairs", len(questions))
    logger.info("Label distribution: %s", dict(Counter(y_list)))

    y = np.array(y_list)

    # Embed
    cache_path = Path("data/predictor_train_embeddings.npy")
    X = encode_questions(
        pred_cfg["encoder"],
        questions,
        batch_size=64,
        cache_path=cache_path,
    )

    # CV
    logger.info("=== %d-fold stratified CV ===", pred_cfg["cv_folds"])
    cv_results, _ = run_cv(
        X, y, labels=labels,
        C=pred_cfg["classifier"]["C"],
        n_folds=pred_cfg["cv_folds"],
        seed=config["logging"]["seed"],
    )
    logger.info("Overall macro-F1 = %.4f (target %.2f)",
                cv_results["overall_macro_f1"], pred_cfg["target_f1"])

    # Optional dev eval
    dev_eval = None
    if args.dev_data is not None:
        dev_examples = load_examples(args.dev_data)
        dev_q, dev_y = extract_questions_and_labels(dev_examples)
        if dev_q:
            dev_X = encode_questions(pred_cfg["encoder"], dev_q, batch_size=64,
                                     cache_path=Path("data/predictor_dev_embeddings.npy"))
            # train on full train, evaluate
            full_clf = train_final(X, y, C=pred_cfg["classifier"]["C"])
            dev_pred = full_clf.predict(dev_X)
            dev_macro = float(f1_score(dev_y, dev_pred, average="macro",
                                       labels=labels, zero_division=0))
            dev_per_class = {
                lab: float(f1_score(dev_y, dev_pred, labels=[lab],
                                    average="macro", zero_division=0))
                for lab in labels
            }
            dev_cm = confusion_matrix(dev_y, dev_pred, labels=labels).tolist()
            dev_eval = {
                "n": len(dev_q),
                "macro_f1": dev_macro,
                "per_class_f1": dev_per_class,
                "confusion_matrix": dev_cm,
            }
            logger.info("Dev macro-F1 = %.4f  per_class=%s",
                        dev_macro, {k: f"{v:.3f}" for k, v in dev_per_class.items()})

    # Final train + save
    final_clf = train_final(X, y, C=pred_cfg["classifier"]["C"])
    ckpt = Path(paths["predictor_ckpt"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt, "wb") as f:
        pickle.dump(
            {
                "model": final_clf,
                "encoder_name": pred_cfg["encoder"],
                "labels": labels,
            }, f
        )
    logger.info("Saved final model to %s", ckpt)

    # Save stats
    stats = {
        "train_data": str(train_path),
        "n_train": len(questions),
        "label_distribution": dict(Counter(y_list)),
        "encoder": pred_cfg["encoder"],
        "C": pred_cfg["classifier"]["C"],
        "target_f1": pred_cfg["target_f1"],
        "cv": cv_results,
        "dev_eval": dev_eval,
        "checkpoint": str(ckpt),
    }
    stats_path = Path(paths["results_dir"]) / "predictor_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=float)
    logger.info("Saved stats to %s", stats_path)

    # Threshold check
    if cv_results["overall_macro_f1"] >= pred_cfg["target_f1"]:
        logger.info("✓ Macro-F1 target met (%.4f >= %.2f)",
                    cv_results["overall_macro_f1"], pred_cfg["target_f1"])
    else:
        logger.warning("! Macro-F1 below target (%.4f < %.2f) — proceed to Phase 3 with caveat",
                       cv_results["overall_macro_f1"], pred_cfg["target_f1"])


if __name__ == "__main__":
    main()
