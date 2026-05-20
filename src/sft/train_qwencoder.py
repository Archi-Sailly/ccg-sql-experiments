"""Phase 5'' — LoRA SFT on top of Qwen2.5-Coder-7B (vanilla) for BIRD-only data.

Key differences from Phase 5:
* Base model: Qwen/Qwen2.5-Coder-7B-Instruct (vanilla code LLM, no SQL pretraining)
* Training data: BIRD-only (4,549 examples, 100% schema-aware)
* Goal: train construct-conditioned model entirely from vanilla code LLM (academic clarity)

Usage:
    python -m src.sft.train_omnisql --config configs/default.yaml \\
        --train-jsonl data/sft_bird_only.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import logging
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


def make_dataset(records, tokenizer):
    from datasets import Dataset
    rows = []
    for r in records:
        text = r["prompt"] + r["completion"] + tokenizer.eos_token
        rows.append({"text": text})
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--train-jsonl", type=Path,
                        default=Path("data/sft_bird_only.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Base model (default: OmniSQL-7B)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("checkpoints/sft_omnisql"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    config = load_config(args.config)
    sft_cfg = config["sft"]
    train_cfg = sft_cfg["training"]
    lora_cfg = sft_cfg["lora"]

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Base model: %s", args.base_model)
    logger.info("Output dir: %s", args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.model_max_length = train_cfg["max_seq_length"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading in 4-bit (QLoRA)")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    lora = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    records = load_jsonl(args.train_jsonl)
    logger.info("Loaded %d records", len(records))
    train_ds = make_dataset(records, tokenizer)

    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=10, save_steps=200, save_total_limit=3,
        report_to="none", dataset_text_field="text",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=config["logging"]["seed"],
    )

    trainer = SFTTrainer(
        model=model, args=sft_args,
        train_dataset=train_ds, processing_class=tokenizer,
    )

    resume_from = None
    if args.resume:
        ckpts = sorted(args.output_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume_from = str(ckpts[-1])
            logger.info("Resuming from %s", resume_from)

    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final to %s", final_dir)

    # Stats
    stats = {
        "base_model": args.base_model,
        "n_train": len(records),
        "num_epochs": args.num_epochs,
        "lora": lora_cfg,
        "output_dir": str(args.output_dir),
        "final_adapter": str(final_dir),
    }
    out = Path("results/sft_omnisql_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
