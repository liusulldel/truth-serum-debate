"""Naive plurality / majority vote baseline.

Reference:
    Boland, P. J. (1989). "Majority systems and the Condorcet jury theorem."
    Journal of the Royal Statistical Society Series D 38(3): 181-189.
    DOI: 10.2307/2348873  (Generalises Condorcet 1785; majority is the optimal
    Bayes rule iff voters are i.i.d. and individually accurate above 1/2.)
"""
from __future__ import annotations
from collections import Counter
from typing import Sequence
from . import DebaterOutput, Decision


def aggregate(question: str, debater_outputs: Sequence[DebaterOutput]) -> Decision:
    """Plurality vote over debater answers; ties break to FALSE.

    Args:
        question: The question being decided.
        debater_outputs: One or more debater outputs.

    Returns:
        A Decision tagged ``majority_vote``.
    """
    if not debater_outputs:
        raise ValueError("Need >=1 debater output.")
    votes = Counter(d.answer for d in debater_outputs)
    answer = 1 if votes[1] > votes[0] else 0  # tie-break to FALSE
    return Decision(question, answer, votes[1] / len(debater_outputs), False, "majority_vote")
