# Deidentification BERT Reproduction

This repository reproduces **“Deidentification of Free‑Text Medical Records using Pre‑trained Bidirectional Transformers”** (Johnson *et al.*, 2020) and adds a lightweight rules layer plus a PyHealth task wrapper.

> **Course:** DL4H — Spring 2025  
> **Author:** Mohamadhossein Amirifardchime (`ma144`)

See the [`report`](report/) and [`slides`](slides/) folders for the final deliverables.

## Quick start

```bash
conda env create -f environment.yml
conda activate deid-bert
python src/train_model.py --dataset i2b2_2014
python src/evaluate.py --checkpoint outputs/i2b2_2014/model.ckpt
```

