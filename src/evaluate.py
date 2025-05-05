"""Compute evaluation metrics and export CSV."""

import argparse, json, pathlib, pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score

from .utils import get_logger

logger = get_logger(__name__)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    return p.parse_args()


def main():
    args = get_args()
    dataset = load_from_disk(f"data/processed/tokenised/{args.dataset}")
    label_list = dataset["test"].features["labels"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)

    # TODO: run predictions
    logger.info("Evaluation skeleton — add prediction loop.")


if __name__ == "__main__":
    main()
