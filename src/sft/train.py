"""Phase 5 — LoRA SFT for Qwen3-8B on construct-conditioned BIRD+SynSQL data.

Resumable: HuggingFace Trainer auto-resumes from the latest checkpoint
in output_dir when --resume is set.

Usage:
    python -m src.sft.train --config configs/default.yaml --train-jsonl data/sft_train.jsonl
    python -m src.sft.train --config configs/default.yaml --train-jsonl data/sft_train.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_dataset(records, tokenizer, max_seq_length: int):
    """Format (prompt, completion) for SFTTrainer text mode."""
    from datasets import Dataset

    rows = []
    for r in records:
        text = r["prompt"] + r["completion"] + tokenizer.eos_token
        rows.append({"text": text})
    ds = Dataset.from_list(rows)
    logger.info("Built HF dataset with %d rows", len(ds))
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/sft_train.jsonl"))
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output_dir")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Load base model in 4-bit (QLoRA) to fit smaller GPUs")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override max training steps (-1 = use config epochs)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    config = load_config(args.config)
    sft_cfg = config["sft"]
    train_cfg = sft_cfg["training"]
    lora_cfg = sft_cfg["lora"]

    # Imports (lazy so the module loads on CPU-only environments for tests)
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    base_model_id = sft_cfg["base_model"]
    output_dir = Path(sft_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Base model: %s", base_model_id)
    logger.info("Output dir: %s", output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading model in 4-bit (QLoRA)")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if train_cfg.get("bf16", True) else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    # LoRA
    lora = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Data
    records = load_jsonl(args.train_jsonl)
    logger.info("Loaded %d training records", len(records))
    train_ds = make_dataset(records, tokenizer, max_seq_length=train_cfg["max_seq_length"])

    # Training config (use TRL SFTConfig)
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        max_seq_length=train_cfg["max_seq_length"],
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="none",                # add 'wandb' if WANDB_API_KEY set
        dataset_text_field="text",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=config["logging"]["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    # Resume detection
    resume_from = None
    if args.resume:
        ckpts = sorted(output_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume_from = str(ckpts[-1])
            logger.info("Resuming from %s", resume_from)
        else:
            logger.info("No checkpoint found — starting fresh")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final adapter
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final LoRA adapter to %s", final_dir)

    # Quick stats
    stats = {
        "base_model": base_model_id,
        "n_train": len(records),
        "lora": lora_cfg,
        "training": train_cfg,
        "output_dir": str(output_dir),
        "final_adapter": str(final_dir),
        "use_4bit": args.use_4bit,
    }
    out_stats = Path("results/sft_stats.json")
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    logger.info("Wrote stats to %s", out_stats)


if __name__ == "__main__":
    main()
