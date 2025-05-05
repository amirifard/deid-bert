
import re
from typing import List
PATTERNS=[re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), re.compile(r"\b\d{3}[-.)]?\d{3}[-.]?\d{4}\b")]
def apply_rules(tokens: List[str], labels: List[int])->List[int]:
    for i,t in enumerate(tokens):
        for p in PATTERNS:
            if p.fullmatch(t):
                labels[i]=1
                break
    return labels
