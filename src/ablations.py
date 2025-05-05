"""Run a sweep over different model variants."""

import itertools, subprocess, argparse, json, pathlib
from .utils import get_logger

logger = get_logger(__name__)


def main():
    models = ["bert-base-uncased", "bert-large-uncased", "biobert-base-cased-v1.2"]
    datasets = ["i2b2_2014"]
    for model_name, ds in itertools.product(models, datasets):
        logger.info(f"Running {model_name} on {ds}")
        subprocess.run(
            [
                "python",
                "src/train_model.py",
                "--dataset",
                ds,
                "--model_name",
                model_name,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
