
import argparse, json, pathlib
from datasets import load_from_disk
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments)
from .utils import set_seed, get_logger
logger=get_logger(__name__)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="physionet"); ap.add_argument("--model_name",default="bert-base-uncased"); ap.add_argument("--epochs",type=int,default=1)
    args=ap.parse_args(); set_seed()
    ds=load_from_disk(f"data/processed/tokenised/{args.dataset}")
    num_labels=len(ds["train"].features["labels"].feature.names)
    model=AutoModelForTokenClassification.from_pretrained(args.model_name,num_labels=num_labels)
    tok=AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    coll=DataCollatorForTokenClassification(tok)
    targs=TrainingArguments(output_dir=f"outputs/{args.dataset}",num_train_epochs=args.epochs,per_device_train_batch_size=4,evaluation_strategy="epoch",save_strategy="epoch",logging_steps=10,report_to="none")
    trainer=Trainer(model=model,args=targs,train_dataset=ds["train"],eval_dataset=ds["validation"],tokenizer=tok,data_collator=coll)
    trainer.train(); trainer.save_model(f"outputs/{args.dataset}/checkpoint")
    with open(f"outputs/{args.dataset}/metrics.json","w") as f: json.dump(trainer.evaluate(ds["test"]),f)
if __name__=="__main__": main()
