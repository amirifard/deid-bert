
"""
Build HuggingFace token classification dataset from PhysioNet De‑ID plain‑text
files: `id.text` (raw notes) and `id.res` (de‑identified notes with PHI wrapped
in [** **]).

Strategy
--------
For each *line* we align `text` with `res`:
* Tokenise both with whitespace.
* A token that starts with "[**" in `id.res` is PHI -> label 1.
* We remove the "[**" and "**]" markers so the model sees the clean token.

Output:
    data/processed/harmonised/physionet.jsonl
        {"id": <int>, "tokens": [...], "labels": [...]}
    data/processed/tokenised/physionet/
        HuggingFace DatasetDict arrow dir
"""

import pathlib, json, re
from typing import List, Tuple
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from .utils import get_logger

logger = get_logger(__name__)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

RAW_DIR = pathlib.Path("data/raw/physionet")
HARM_PATH = pathlib.Path("data/processed/harmonised/physionet.jsonl")
TOK_DIR = pathlib.Path("data/processed/tokenised/physionet")

PHI_LABELS = ["O", "PHI"]  # 0=non‑PHI, 1=PHI
label_feature = ClassLabel(num_classes=2, names=PHI_LABELS)


def _process_line(raw_line: str, res_line: str) -> Tuple[List[str], List[int]]:
    raw_toks = raw_line.strip().split()
    res_toks = res_line.strip().split()
    labels = [0] * len(raw_toks)
    j = 0
    for i, tok in enumerate(res_toks):
        if tok.startswith("[**"):
            # consume until we hit closing **]
            collected = [tok]
            while not tok.endswith("**]"):
                j += 1
                tok = res_toks[j]
                collected.append(tok)
            # mark PHI for corresponding raw token(s)
            # assume PHI spans exactly one token in raw text
            labels[i] = 1
            res_token_clean = re.sub(r"\[\*\*|\*\*\]", "", collected[0])
            res_toks[i] = res_token_clean
        j += 1
    assert len(raw_toks) == len(res_toks), "Token mis‑alignment"
    return raw_toks, labels


def harmonise():
    txt = (RAW_DIR/"id.text").read_text(encoding="utf8").splitlines()
    res = (RAW_DIR/"id.res").read_text(encoding="utf8").splitlines()
    assert len(txt) == len(res), "Line count mismatch between text and res"
    HARM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HARM_PATH.open("w", encoding="utf8") as f:
        rec_id = 0
        for raw, r in zip(txt, res):
            tokens, labels = _process_line(raw, r)
            json.dump({"id": rec_id, "tokens": tokens, "labels": labels}, f)
            f.write("\n")
            rec_id += 1
    logger.info(f"Wrote {rec_id} records -> {HARM_PATH}")


def tokenise(block_size=256):
    # read jsonl
    records = [json.loads(l) for l in HARM_PATH.open()]
    # explode into word-level dataset
    examples = []
    for rec in records:
        examples.append({"tokens": rec["tokens"], "labels": rec["labels"]})
    ds = Dataset.from_list(examples)

    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(label_feature),
        }
    )

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            [' '.join(toks) for toks in batch["tokens"]],
            truncation=True,
            max_length=block_size,
            padding=False,
            return_offsets_mapping=True,
        )
        all_labels = []
        for label_seq, offsets in zip(batch["labels"], tokenized["offset_mapping"]):
            token_labels = []
            for offset in offsets:
                if offset == (0, 0):  # CLS or SEP
                    token_labels.append(-100)
                else:
                    idx = offset[0] // 1000  # dummy mapping not used
                    # simple approach: use first label
                    token_labels.append(label_seq[0] if label_seq else 0)
            all_labels.append(token_labels)
        tokenized.pop("offset_mapping")
        tokenized["labels"] = all_labels
        return tokenized

    tokenised = ds.map(tokenize_and_align, batched=True, remove_columns=["tokens"])
    split = tokenised.train_test_split(test_size=0.2, seed=42)
    val_test = split["test"].train_test_split(test_size=0.5, seed=42)
    dsd = DatasetDict({"train": split["train"], "validation": val_test["train"], "test": val_test["test"]})
    dsd.save_to_disk(TOK_DIR)
    logger.info(f"Tokenised dataset saved to {TOK_DIR}")


if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["harmonise", "tokenise", "test"])
    args = p.parse_args()
    if args.stage == "harmonise":
        harmonise()
    elif args.stage == "tokenise":
        tokenise()
    elif args.stage == "test":
        harmonise()
        recs = [json.loads(l) for l in HARM_PATH.open()]
        phi_cnt = sum(sum(lbl) for lbl in (r["labels"] for r in recs))
        print(f"PHI tokens found: {phi_cnt}")
