# Reproducing “Deidentification of free-text medical records using pre-trained bidirectional transformers”


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository **reproduces** Johnson et al. (2020)’s BERT‑deid system on the publicly
available **PhysioNet 2010 De‑ID** corpus, adds a lightweight regex overlay that
halves false‑negatives at 99 % recall, and contributes a reusable
`DeidTransformer` task wrapper to **PyHealth**.

| Deliverable | Link |
|-------------|------|
| **Final report (PDF)** | [`report/Final Report.pdf`](report/Final Report.pdf) |
| **slide deck (PPTX)** | [`slides/deid_presentation.pptx`](slides/deid_presentation.pptx) |
| **PyHealth pull‑request** | <https://github.com/sunlabuiuc/PyHealth/pull/412> |

> **Author:** Mohamadhossein Amirifardchime (`ma144`)   •    

---

## Quick start

```bash
# Conda environment
conda env create -f environment.yml
conda activate deid-final

# 1 Pre‑process raw notes into a tokenised dataset
python -m src.data_processing harmonise   # align id.text with id.res
python -m src.data_processing tokenise    # build HF DatasetDict

# 2 Fine‑tune BERT (1 epoch demo)
python src/train_model.py --dataset physionet --epochs 1

# 3 Predict on test set + hybrid regex overlay
python src/evaluate.py --dataset physionet --checkpoint outputs/physionet/checkpoint

# 4 Optional: tiny model sweep
python src/ablations.py

# 5 Generate demo visual
python src/visualize.py                   # outputs slides/token_lengths.png
