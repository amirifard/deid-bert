"""Simple regex‑based PHI detector to reduce FN."""

import re
PATTERNS = {
    "EMAIL": re.compile(r"[\w\.-]+@[\w\.-]+"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MRN": re.compile(r"\b\d{6,9}\b"),
}


def add_rules(tokens):
    """Return list of (token, label) with rule overrides."""
    return tokens  # TODO
