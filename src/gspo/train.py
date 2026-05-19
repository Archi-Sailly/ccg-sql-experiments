"""Phase 6 — GSPO RL training.

Loads the Phase 5 LoRA adapter and continues training with TRL's GRPO trainer
under sequence-level importance sampling (GSPO).

Reward = R_exec + lambda_construct * R_construct, computed via src.reward.reward.

Resumable: checkpoints/gspo/checkpoint-* — pass --resume.

Usage:
    python -m src.gspo.train --config configs/default.yaml \\
        --train-jsonl data/gspo_train.jsonl \\
        --sft-adapter checkpoints/sft/final --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src.reward.reward import compute_reward

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────
def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_sql_from_completion(completion: str) -> str:
    """Take the model's completion text and return the SQL fragment.

    The SFT prompt ends in 'SQL:' so completions usually start with the SQL
    statement directly. We strip leading whitespace and stop at the first
    obvious sentinel (newline+blank or '<|...|>' tokens).
    """
    if not isinstance(completion, str):
        return ""
    sql = completion.strip()
    # Stop at end-of-turn tokens that Qwen may emit
    for sentinel in ("<|im_end|>", "<|endoftext|>", "</s>"):
        idx = sql.find(sentinel)
        if idx != -1:
            sql = sql[:idx]
    # Heuristic: stop at a blank line after the first SELECT/WITH
    sql = sql.strip()
    # Optional: remove trailing markdown fences if present
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    return sql


# ─── Reward closure ───────────────────────────────────────
def make_reward_fn(
    lambda_construct: float = 0.2,
    timeout: float = 10.0,
):
    """Return a reward function compatible with TRL GRPOTrainer.

    TRL passes:
        completions: list[str]   (length = batch * num_generations)
        **kwargs:    any additional dataset columns, each as a list
    The dataset columns we attach: prompt, gold_sql, db_path, target_construct.
    """
    def reward_fn(completions, **kwargs):
        gold_sqls = kwargs.get("gold_sql", [])
        db_paths = kwargs.get("db_path", [])
        targets = kwargs.get("target_construct", [])
        # In GRPO, each prompt is repeated num_generations times. TRL handles
        # the broadcasting of metadata via dataset columns automatically.
        rewards = []
        for i, comp in enumerate(completions):
            gen_sql = extract_sql_from_completion(comp)
            gold = gold_sqls[i] if i < len(gold_sqls) else ""
            db_path = db_paths[i] if i < len(db_paths) else ""
            target = targets[i] if i < len(targets) else "plain"
            if not gen_sql or not gold or not db_path:
                rewards.append(0.0)
                continue
            try:
                r = compute_reward(
                    gen_sql=gen_sql, gold_sql=gold,
                    db_path=db_path, target_construct=target,
                    lambda_construct=lambda_construct,
                    timeout=timeout, use_subprocess=True,
                )
                rewards.append(float(r["r_total"]))
            except Exception as e:
                logger.warning("reward error: %s", e)
                rewards.append(0.0)
        return rewards

    return reward_fn


# ─── Main ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/gspo_train.jsonl"))
    parser.add_argument("--sft-adapter", type=Path, default=Path("checkpoints/sft/final"))
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    config = load_config(args.config)
    sft_cfg = config["sft"]
    gspo_cfg = config["gspo"]
    rew_cfg = config["reward"]
    train_cfg = gspo_cfg["config"]
    output_dir = Path(gspo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    base_model_id = sft_cfg["base_model"]
    logger.info("Base model:    %s", base_model_id)
    logger.info("SFT adapter:   %s", args.sft_adapter)
    logger.info("GSPO out dir:  %s", output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading base model in 4-bit (QLoRA)")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    # Attach SFT LoRA adapter, trainable
    logger.info("Loading SFT adapter at %s", args.sft_adapter)
    model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
    model.print_trainable_parameters()

    # Dataset
    rows = load_jsonl(args.train_jsonl)
    logger.info("Loaded %d RL rows", len(rows))
    ds = Dataset.from_list(rows)

    # GRPO config
    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        num_generations=train_cfg["num_generations"],
        max_prompt_length=train_cfg["max_prompt_length"],
        max_completion_length=train_cfg["max_completion_length"],
        beta=train_cfg["kl_coef"],
        temperature=train_cfg["temperature"],
        top_p=train_cfg["top_p"],
        importance_sampling_level="sequence",  # GSPO
        logging_steps=10,
        save_steps=gspo_cfg["monitoring"]["save_every"],
        save_total_limit=3,
        report_to="none",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=config["logging"]["seed"],
    )

    # Reward
    reward_fn = make_reward_fn(
        lambda_construct=rew_cfg["lambda_construct"],
        timeout=rew_cfg["execution_timeout"],
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Resume
    resume_from = None
    if args.resume:
        ckpts = sorted(output_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume_from = str(ckpts[-1])
            logger.info("Resuming from %s", resume_from)
        else:
            logger.info("No GSPO checkpoint — starting fresh from SFT")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final GSPO adapter to %s", final_dir)

    # Stats
    stats = {
        "base_model": base_model_id,
        "sft_adapter": str(args.sft_adapter),
        "n_train": len(rows),
        "gspo_config": train_cfg,
        "reward": rew_cfg,
        "output_dir": str(output_dir),
        "final_adapter": str(final_dir),
        "use_4bit": args.use_4bit,
    }
    out = Path("results/gspo_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote stats to %s", out)


if __name__ == "__main__":
    main()
