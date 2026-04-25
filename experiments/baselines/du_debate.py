"""Du et al. multi-agent debate: R rounds of cross-agent revision + majority vote.

Reference:
    Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023).
    "Improving Factuality and Reasoning in Language Models through Multiagent
    Debate." arXiv:2305.14325 (ICML 2024).

Protocol: N agents propose answers + reasoning, see each other's responses,
revise over R rounds, then majority-vote. The empirical regularity Du measures
is that per-round confidences drift toward the population mean (agents update
toward what they see others saying). After R rounds, mode answer wins.

Mock implementation: at each round, each debater's p_true moves a fraction
DRIFT toward the population mean p_true. After R rounds, hard-pick the mode of
the resulting answers; p_agg is the mean of post-debate p_true values.
"""
from __future__ import annotations
from collections import Counter
from typing import Sequence
from . import DebaterOutput, Decision

DRIFT = 0.4  # per-round shrinkage toward population mean (Du fig. 3 regularity)


def aggregate(question: str, debater_outputs: Sequence[DebaterOutput],
              rounds: int = 2) -> Decision:
    """Du-style multi-round drift toward population mean, then majority vote.

    Args:
        question: The question being decided.
        debater_outputs: One or more debater outputs.
        rounds: Number of revision rounds to apply.

    Returns:
        A Decision tagged ``du_debate``.
    """
    if not debater_outputs:
        raise ValueError("Need >=1 debater output.")
    ps = [d.p_true for d in debater_outputs]
    for _ in range(rounds):
        mean_p = sum(ps) / len(ps)
        ps = [(1.0 - DRIFT) * p + DRIFT * mean_p for p in ps]
    answers = [1 if p >= 0.5 else 0 for p in ps]
    c = Counter(answers)
    answer = 1 if c[1] > c[0] else 0
    p_agg = sum(ps) / len(ps)
    return Decision(question, answer, p_agg, False, "du_debate")
