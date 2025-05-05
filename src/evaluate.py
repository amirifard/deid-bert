
import argparse, pathlib, pandas as pd, numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from .hybrid_rules import apply_rules
from .utils import get_logger
logger=get_logger(__name__)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="physionet"); ap.add_argument("--checkpoint",required=True); args=ap.parse_args()
    ds=load_from_disk(f"data/processed/tokenised/{args.dataset}")["test"]
    tok=AutoTokenizer.from_pretrained(args.checkpoint); model=AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    preds=Trainer(model,tokenizer=tok).predict(ds).predictions.argmax(-1)
    rows=[]
    for ids,p in zip(ds["input_ids"],preds):
        toks=tok.convert_ids_to_tokens(ids); labels=apply_rules(toks,p.tolist())
        rows.append({"tokens":' '.join(toks),"pred":labels})
    out=pathlib.Path(args.checkpoint)/"predictions.csv"; pd.DataFrame(rows).to_csv(out,index=False); logger.info(f"Predictions -> {out}")
if __name__=="__main__": main()
