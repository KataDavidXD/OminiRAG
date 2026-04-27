"""Shared text-level scoring primitives.

All functions are pure ``(str, ...) -> number``.  No model loading, no LLM
calls, no external dependencies beyond the standard library.
"""

from __future__ import annotations

import collections
import re
import string


def normalize_answer(s: str) -> str:
    """Standard SQuAD-style normalization (lowercase, strip articles/punct)."""
    def _remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def _remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return _white_space_fix(_remove_articles(_remove_punc(s.lower())))


def compute_f1(gold: str, pred: str) -> float:
    """Token-level F1 between *gold* and *pred* after normalization."""
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()

    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_exact(gold: str, pred: str) -> int:
    """Exact-match after normalization.  Returns 1 or 0."""
    return int(normalize_answer(gold) == normalize_answer(pred))


def exact_presence(short_answers: list[str], context: str) -> bool:
    """True if *any* short answer is a normalized substring of *context*."""
    n_context = normalize_answer(context)
    return any(normalize_answer(sa) in n_context for sa in short_answers)


def remove_citations(sent: str) -> str:
    """Strip ``[N]``-style citation markers from generated text."""
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")
