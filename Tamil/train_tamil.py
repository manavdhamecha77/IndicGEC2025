#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)


try:
    import torch
except ImportError:
    torch = None

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune mT5 on Tamil GEC from a Malayalam-finetuned checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Malayalam-finetuned checkpoint (base for continued FT)")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to Tamil train CSV (UTF-8)")
    parser.add_argument("--val_csv", type=str, default=None, help="Optional path to Tamil dev/val CSV; if omitted, a split from train is used")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints and final model")

    parser.add_argument("--source_col", type=str, default="source", help="Input column name in CSV")
    parser.add_argument("--target_col", type=str, default="target", help="Target column name in CSV")
    parser.add_argument("--prefix", type=str, default="gec ta: ", help="Instruction/prefix added to inputs")
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=256)

    # Training hyperparams
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Precision
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available (recommended on H100)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")

    # LoRA options
    parser.add_argument("--use_lora", action="store_true", help="Train a LoRA adapter instead of full finetune")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Validation split ratio if val_csv not provided
    parser.add_argument("--val_ratio", type=float, default=0.05)

    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, use_lora: bool, lora_cfg_params: dict):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)

    if use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is not installed but --use_lora was set. Install peft or disable --use_lora.")
        lora_cfg = LoraConfig(
            r=lora_cfg_params.get("r", 8),
            lora_alpha=lora_cfg_params.get("lora_alpha", 16),
            lora_dropout=lora_cfg_params.get("lora_dropout", 0.1),
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q", "k", "v", "o"],
        )
        model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def build_datasets(train_csv, val_csv, source_col, target_col, prefix, tokenizer, max_source_len, max_target_len, val_ratio, seed):
    df_train = pd.read_csv(train_csv)

    # Basic column sanity
    if source_col not in df_train.columns or target_col not in df_train.columns:
        raise ValueError(f"CSV must contain columns '{source_col}' and '{target_col}'. Found: {list(df_train.columns)}")

    ds = Dataset.from_pandas(df_train)

    if val_csv is None:
        ds = ds.train_test_split(test_size=val_ratio, seed=seed)
        train_ds, eval_ds = ds["train"], ds["test"]
    else:
        df_val = pd.read_csv(val_csv)
        if source_col not in df_val.columns or target_col not in df_val.columns:
            raise ValueError(f"Val CSV must contain columns '{source_col}' and '{target_col}'. Found: {list(df_val.columns)}")
        train_ds = ds
        eval_ds = Dataset.from_pandas(df_val)

    def preprocess(row):
        inp = prefix + str(row[source_col])
        out = str(row[target_col])
        model_inputs = tokenizer(inp, max_length=max_source_len, truncation=True)
        labels = tokenizer(out, max_length=max_target_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        use_lora=args.use_lora,
        lora_cfg_params={"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout},
    )

    train_ds, eval_ds = build_datasets(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        source_col=args.source_col,
        target_col=args.target_col,
        prefix=args.prefix,
        tokenizer=tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Prefer bf16 on H100; if both flags set, bf16 wins
    bf16 = bool(args.bf16)
    fp16 = bool(args.fp16) and not bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=max(50, args.eval_steps // 5),
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=fp16,
        bf16=bf16,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to=["none"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save final artifacts
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also save a small README with command
    with open(os.path.join(args.output_dir, "COMMAND.txt"), "w", encoding="utf-8") as f:
        f.write("Run command and args used are stored by your Slurm script.")


if __name__ == "__main__":
    main()
