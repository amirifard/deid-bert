"""Fine‑tune BERT on de‑identification."""

import argparse, pathlib, json, torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
from .utils import set_seed, get_logger

logger = get_logger(__name__)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Name of processed dataset dir")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


def main():
    args = get_args()
    set_seed()
    dataset = load_from_disk(f"data/processed/tokenised/{args.dataset}")
    label_list = dataset["train"].features["labels"].feature.names
    num_labels = len(label_list)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.dataset}",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    metrics = trainer.evaluate(dataset["test"])
    with open(
        f"{args.output_dir}/{args.dataset}/metrics.json", "w", encoding="utf8"
    ) as f:
        json.dump(metrics, f, indent=2)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
