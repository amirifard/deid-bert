"""Dataset download & preprocessing pipeline.

Functions:
    download_datasets()
    harmonise()
    tokenise()
    sample_segments()
"""

import os, pathlib, re, json
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer
from .utils import get_logger, set_seed

logger = get_logger(__name__)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


def download_datasets(root="data/raw"):
    """Download or symlink i2b2/PhysioNet corpora.

    NOTE: You must hold a signed DUA. Place the raw XML files under
    `data/raw/<corpus>/train/`, `test/` before running.
    """
    os.makedirs(root, exist_ok=True)
    logger.info("Assuming datasets already placed per DUA instructions.")


def harmonise(raw_root="data/raw", out_root="data/processed/harmonised"):
    """Convert all corpora into a common JSON Lines span format."""
    pathlib.Path(out_root).mkdir(parents=True, exist_ok=True)
    # Placeholder: iterate over corpora and convert.
    for corpus in pathlib.Path(raw_root).iterdir():
        logger.info(f"Skipping harmonisation for {corpus} (TODO).")


def tokenise(in_root="data/processed/harmonised", out_root="data/processed/tokenised"):
    """Run BERT WordPiece tokenisation and align labels."""
    pathlib.Path(out_root).mkdir(parents=True, exist_ok=True)
    logger.info("Tokenising (TODO).")


def sample_segments(in_root="data/processed/tokenised", window=256, stride=128):
    """Clip long notes into sliding windows."""
    logger.info("Segment sampling (TODO).")
