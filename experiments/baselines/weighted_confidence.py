"""Confidence-weighted vote -- the standard 'industrial' baseline.

Reference:
    Grofman, B., Owen, G., & Feld, S. L. (1983).
    "Thirteen theorems in search of the truth."
    Theory and Decision 15(3): 261-278. DOI: 10.1007/BF00125672
    (Theorem 6: optimal aggregation weight is log(p_i / (1 - p_i)); the
    confidence-weighted majority is the practitioner approximation.)
"""
from __future__ import annotations
import math
from typing import Sequence
from . import DebaterOutput, Decision


def aggregate(question: str, debater_outputs: Sequence[DebaterOutput]) -> Decision:
    if not debater_outputs:
        raise ValueError("Need >=1 debater output.")
    log_odds = 0.0
    for d in debater_outputs:
        p = min(max(d.p_true, 1e-6), 1 - 1e-6)
        log_odds += d.confidence * math.log(p / (1 - p))
    p_agg = 1.0 / (1.0 + math.exp(-log_odds))
    return Decision(question, 1 if p_agg >= 0.5 else 0, p_agg, False, "weighted_confidence")
