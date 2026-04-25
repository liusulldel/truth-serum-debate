"""Self-consistency: majority vote over k samples per debater.

Reference:
    Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A.,
    & Zhou, D. (2022). "Self-Consistency Improves Chain of Thought Reasoning in
    Language Models." arXiv:2203.11171.

Per-debater modal answer first, then majority over debaters (two-level plurality).
"""
from __future__ import annotations
from collections import Counter
from typing import Sequence
from . import DebaterOutput, Decision


def aggregate(question: str, debater_outputs: Sequence[DebaterOutput]) -> Decision:
    """Two-level plurality: per-debater modal sample, then majority over debaters.

    Args:
        question: The question being decided.
        debater_outputs: One or more debater outputs, optionally with samples.

    Returns:
        A Decision aggregated by self-consistency majority voting.
    """
    if not debater_outputs:
        raise ValueError("Need >=1 debater output.")
    modes, p1s = [], []
    for d in debater_outputs:
        if d.samples:
            c = Counter(d.samples)
            modes.append(1 if c[1] > c[0] else 0)
            p1s.append(c[1] / len(d.samples))
        else:
            modes.append(d.answer)
            p1s.append(d.p_true)
    pop = Counter(modes)
    return Decision(question, 1 if pop[1] > pop[0] else 0,
                    sum(p1s) / len(p1s), False, "self_consistency")
