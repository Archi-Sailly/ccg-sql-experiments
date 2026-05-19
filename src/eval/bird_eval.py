"""Phase 8 — BIRD Dev evaluation.

For each BIRD Dev question:
  1. Build the prompt (schema + few-shot examples + question)
  2. Generate K candidates with the trained model (vLLM if available, else HF)
  3. Rerank to choose a final SQL (argmax / weighted_vote)
  4. Execute final SQL against the BIRD Dev DB and compare with gold (EX)
  5. Classify the construct and compare to BIRD-labelled target → per-class F1
  6. Bootstrap CI on EX

Outputs:
  results/eval_predictions.jsonl   — per-question prediction record
  results/eval_metrics.json        — aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────
def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_examples(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dev_db_root(args_root: Path) -> Path:
    if args_root.exists():
        return args_root
    cand = next(iter(args_root.parent.rglob("dev_databases")), None)
    return cand if cand else args_root


def extract_sql_from_completion(completion: str) -> str:
    import re
    if not isinstance(completion, str):
        return ""
    sql = completion.strip()
    for sentinel in ("<|im_end|>", "<|endoftext|>", "</s>"):
        idx = sql.find(sentinel)
        if idx != -1:
            sql = sql[:idx]
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    # Stop at first blank line (heuristic)
    sql = sql.split("\n\n")[0].strip()
    return sql


# ─── Bootstrap ─────────────────────────────────────────────
def bootstrap_ci(values: list[float], n: int = 500, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    means = []
    for _ in range(n):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


# ─── Generation ───────────────────────────────────────────
def generate_candidates_hf(
    model, tokenizer, prompts: list[str],
    num_generations: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 4,
) -> tuple[list[list[str]], list[list[float]]]:
    """Fallback: use HuggingFace .generate() in batches.

    Returns (candidates_per_prompt, mean_logprobs_per_candidate).
    """
    import torch
    all_cands: list[list[str]] = []
    all_lps: list[list[float]] = []
    model.eval()
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        # Tokenize
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                num_return_sequences=num_generations,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = out.sequences          # (B*K, L)
        input_len = enc.input_ids.shape[1]
        gen_tokens = sequences[:, input_len:]
        # Decode
        texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # Compute mean log-probs per candidate
        # out.scores is a tuple of (B*K, vocab) tensors per generated step
        if out.scores:
            log_probs_step = [torch.log_softmax(s, dim=-1) for s in out.scores]
            mean_lps = []
            for k in range(len(texts)):
                lps = []
                for t_idx, lp in enumerate(log_probs_step):
                    tok_id = gen_tokens[k, t_idx].item()
                    if tok_id == tokenizer.pad_token_id:
                        break
                    lps.append(lp[k, tok_id].item())
                mean_lps.append(sum(lps) / max(len(lps), 1))
            del log_probs_step
        else:
            mean_lps = [0.0] * len(texts)

        # Re-group into per-prompt lists
        for b in range(len(batch_prompts)):
            cands = texts[b * num_generations : (b + 1) * num_generations]
            lps = mean_lps[b * num_generations : (b + 1) * num_generations]
            all_cands.append(cands)
            all_lps.append(lps)

        if (i + batch_size) % (batch_size * 5) == 0:
            logger.info("  generated %d / %d prompts", i + batch_size, len(prompts))
    return all_cands, all_lps


# ─── Main eval loop ────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dev-data", type=Path, default=Path("data/bird_dev_labeled.json"))
    parser.add_argument("--dev-db",   type=Path, default=Path("data/bird/dev/dev_databases"))
    parser.add_argument("--adapter",  type=Path, default=Path("checkpoints/gspo/final"),
                        help="LoRA adapter (default: GSPO; falls back to SFT if missing)")
    parser.add_argument("--n", type=int, default=0, help="Limit dev examples (0 = all)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--strategy", default=None, help="reranker strategy (argmax|weighted_vote)")
    parser.add_argument("--n-shots", type=int, default=2)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    config = load_config(args.config)
    sft_cfg = config["sft"]
    eval_cfg = config["eval"]
    rerank_cfg = config["reranker"]
    strategy = args.strategy or rerank_cfg["selection"]

    base_model_id = sft_cfg["base_model"]
    template = sft_cfg["prompt_template"]

    # Resolve adapter — fall back to SFT if GSPO not found
    if not args.adapter.exists():
        sft_fallback = Path("checkpoints/sft/final")
        if sft_fallback.exists():
            logger.warning("GSPO adapter not found, falling back to SFT")
            args.adapter = sft_fallback
        else:
            raise FileNotFoundError("Neither GSPO nor SFT adapter present")
    logger.info("Adapter: %s", args.adapter)

    # Load dev set
    dev_data = load_examples(args.dev_data)
    if args.n > 0:
        dev_data = dev_data[: args.n]
    logger.info("Loaded %d dev examples", len(dev_data))

    # Build few-shot pool from BIRD train labeled
    train_labeled = Path("data/bird_train_labeled.json")
    pool_by_construct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if args.n_shots > 0 and train_labeled.exists():
        for ex in load_examples(train_labeled):
            pool_by_construct[ex.get("target_construct", "plain")].append(ex)

    # Heavy imports
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # for batched generation

    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()
    logger.info("Model + adapter loaded.")

    # Build prompts
    from src.sft.data import build_prompt, get_question, get_sql, select_few_shots
    bird_dev_root = resolve_dev_db_root(args.dev_db)
    bird_train_root = Path("data/bird/train/train_databases")
    if not bird_train_root.exists():
        cand = next(iter(Path("data/bird/train").rglob("train_databases")), None)
        if not cand:
            cand = next(iter(Path("data/bird").rglob("train_databases")), None)
        if cand: bird_train_root = cand
    logger.info("bird_train_root: %s (exists=%s)", bird_train_root, bird_train_root.exists())
    schema_cache: dict[str, str] = {}
    prompts = []
    meta = []
    for i, ex in enumerate(dev_data):
        target = ex.get("target_construct", "plain")
        few = select_few_shots(target, pool_by_construct, k=args.n_shots, seed=42 + i) if args.n_shots > 0 else []
        rec = build_prompt(ex, template=template, few_shots=few,
                            bird_train_root=bird_train_root,
                            bird_dev_root=bird_dev_root,
                            schema_cache=schema_cache)
        prompts.append(rec["prompt"])
        db_id = ex.get("db_id", "")
        db_path = bird_dev_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            cand = next(iter(bird_dev_root.rglob(f"{db_id}.sqlite")), None)
            if cand: db_path = cand
        meta.append({
            "db_id": db_id,
            "db_path": str(db_path),
            "question": get_question(ex),
            "gold_sql": get_sql(ex),
            "target_construct": target,
        })
    logger.info("Built %d prompts", len(prompts))

    # Generate
    t0 = time.time()
    K = rerank_cfg["K"]
    candidates_per_prompt, log_probs_per_prompt = generate_candidates_hf(
        model, tokenizer, prompts,
        num_generations=K,
        max_new_tokens=512,
        temperature=rerank_cfg["temperature"],
        top_p=rerank_cfg["top_p"],
        batch_size=args.batch_size,
    )
    logger.info("Generation done in %.1f min", (time.time() - t0) / 60)

    # Rerank + execute + score
    from src.data.labeling import classify_construct_with_meta
    from src.reranker.select import rerank
    from src.reward.reward import execute_sql_inproc, rows_match

    predictions = []
    construct_labels = config["data"]["construct_labels"]
    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total:  dict[str, int] = defaultdict(int)
    per_class_pred:    dict[str, int] = defaultdict(int)
    ex_flags = []

    for i, m in enumerate(meta):
        cands = [extract_sql_from_completion(c) for c in candidates_per_prompt[i]]
        lps = log_probs_per_prompt[i]
        chosen_idx, chosen_sql = rerank(
            cands, db_path=m["db_path"], log_probs=lps, strategy=strategy,
        )
        # EX evaluation
        gold_rows, _ = execute_sql_inproc(m["gold_sql"], m["db_path"])
        pred_rows, pred_err = execute_sql_inproc(chosen_sql, m["db_path"])
        ex_ok = bool(rows_match(gold_rows, pred_rows)) if gold_rows is not None else False
        ex_flags.append(1.0 if ex_ok else 0.0)

        pred_cls, _ = classify_construct_with_meta(chosen_sql)
        per_class_total[m["target_construct"]] += 1
        per_class_pred[pred_cls] += 1
        if pred_cls == m["target_construct"]:
            per_class_correct[m["target_construct"]] += 1

        predictions.append({
            "i": i, "db_id": m["db_id"],
            "question": m["question"][:200],
            "gold_sql": m["gold_sql"],
            "pred_sql": chosen_sql,
            "pred_idx": chosen_idx,
            "ex": int(ex_ok),
            "target_construct": m["target_construct"],
            "pred_construct": pred_cls,
            "pred_error": pred_err,
        })

    # Aggregate
    ex_mean, ex_lo, ex_hi = bootstrap_ci(ex_flags, n=eval_cfg["bootstrap_n"])
    cx_flags = [1.0 if (p["ex"] == 1 and p["pred_construct"] == p["target_construct"]) else 0.0
                for p in predictions]
    cx_mean, cx_lo, cx_hi = bootstrap_ci(cx_flags, n=eval_cfg["bootstrap_n"])
    per_class_f1 = {}
    for c in construct_labels:
        tp = per_class_correct[c]
        fp = per_class_pred[c] - tp
        fn = per_class_total[c] - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[c] = {"precision": prec, "recall": rec, "f1": f1,
                           "support": per_class_total[c]}

    metrics = {
        "adapter": str(args.adapter),
        "strategy": strategy,
        "K": K,
        "n_eval": len(predictions),
        "EX": {"mean": ex_mean, "ci_lo": ex_lo, "ci_hi": ex_hi},
        "Cx_EX": {"mean": cx_mean, "ci_lo": cx_lo, "ci_hi": cx_hi},
        "per_class_f1": per_class_f1,
        "construct_distribution_pred": {k: int(per_class_pred[k]) for k in construct_labels},
        "construct_distribution_gold": {k: int(per_class_total[k]) for k in construct_labels},
    }

    # Save
    pred_path = Path("results/eval_predictions.jsonl")
    metrics_path = Path("results/eval_metrics.json")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved predictions to %s", pred_path)
    logger.info("Saved metrics    to %s", metrics_path)

    # Pretty print
    logger.info("=== EVAL RESULTS ===")
    logger.info("EX     = %.4f  (CI [%.4f, %.4f])", ex_mean, ex_lo, ex_hi)
    logger.info("Cx-EX  = %.4f  (CI [%.4f, %.4f])", cx_mean, cx_lo, cx_hi)
    logger.info("--- per-class F1 ---")
    for c in construct_labels:
        m = per_class_f1[c]
        logger.info("  %-7s F1=%.4f  P=%.4f  R=%.4f  support=%d",
                    c, m["f1"], m["precision"], m["recall"], m["support"])


if __name__ == "__main__":
    main()
