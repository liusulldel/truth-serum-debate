"""Vanilla debate: judge picks the more-rebuttable claim.

References:
    Irving, G., Christiano, P., & Amodei, D. (2018). "AI safety via debate."
        arXiv:1805.00899.
    Khan, A., Hughes, J., Valentine, D., et al. (2024). "Debating with More
        Persuasive LLMs Leads to More Truthful Answers." arXiv:2402.06782.

Each debater commits to an answer plus rebuttal_strength in [0,1] (higher =
position survives more rounds). Judge picks the answer of the side with
greater rebuttal strength. Soft p_true derived from the margin.
"""
from __future__ import annotations
from typing import Sequence
from . import DebaterOutput, Decision


def aggregate(question: str, debater_outputs: Sequence[DebaterOutput]) -> Decision:
    """Vanilla debate aggregator: pick the side with greater rebuttal strength.

    Args:
        question: The question being decided.
        debater_outputs: At least two debater outputs with rebuttal_strength.

    Returns:
        A Decision tagged ``vanilla_debate``.
    """
    if len(debater_outputs) < 2:
        raise ValueError("Vanilla debate needs >=2 debaters.")
    s_t = max((d.rebuttal_strength for d in debater_outputs if d.answer == 1), default=0.0)
    s_f = max((d.rebuttal_strength for d in debater_outputs if d.answer == 0), default=0.0)
    if s_t == s_f:
        n1 = sum(1 for d in debater_outputs if d.answer == 1)
        return Decision(question, 1 if n1 > len(debater_outputs) - n1 else 0,
                        n1 / len(debater_outputs), False, "vanilla_debate")
    answer = 1 if s_t > s_f else 0
    margin = abs(s_t - s_f)
    p = 0.5 + 0.5 * margin if answer == 1 else 0.5 - 0.5 * margin
    return Decision(question, answer, p, False, "vanilla_debate")
