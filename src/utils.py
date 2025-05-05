"""Utility helpers (seed fixing, config loading, logging)."""

import random, os, numpy as np, torch, json, logging


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def get_logger(name: str = "deid"):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(name)
