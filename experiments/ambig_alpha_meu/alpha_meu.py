"""alpha-MEU aggregation of debater probability claims.

Reference:
    Ghirardato, P., Maccheroni, F., & Marinacci, M. (2004).
    "Differentiating ambiguity and ambiguity attitude."
    Journal of Economic Theory 118(2): 133-173.
    (Generalizes Gilboa-Schmeidler 1989 maxmin EU; alpha=1 recovers MEU.)

Setup
-----
Each debater r in {1..N} reports a probability vector p_r over m mutually
exclusive answer options for a question. The judge does not know which
debater's prior is correct, so it treats {p_1, ..., p_N} as a *set* of
priors C (Gilboa-Schmeidler's "multiple priors").

For a binary "is the claim true?" question (option 0 = false, option 1 = true),
let q_r = p_r[1] = debater r's probability of "true".

alpha-MEU aggregated probability of "true":

    P_alpha = alpha * min_r q_r  +  (1 - alpha) * max_r q_r

with alpha in [0, 1] the ambiguity-aversion coefficient. alpha = 1 is pure
maxmin (Gilboa-Schmeidler); alpha = 0 is pure maxmax (optimist); alpha = 0.5
is Hurwicz's classical criterion.

Ambiguity index
---------------
    A = max_r q_r - min_r q_r   in [0, 1]

This is the diameter of the set of priors projected onto the "true" event.
Bewley (2002) inertia: when A exceeds a threshold tau, the priors disagree
too much for any unanimous-preference action -- the judge ABSTAINS.

Decision rule
-------------
    if A > tau:           ABSTAIN     (Bewley inertia)
    elif P_alpha >= 0.5:  decide TRUE
    else:                 decide FALSE

This is the central safety property: out-of-distribution inputs cause
debaters to disagree, A spikes, and the judge withholds judgment instead
of confidently averaging to the wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AmbiguityDecision:
    """Result of alpha-MEU aggregation."""
    question: str
    p_alpha: float            # aggregated P(true) under alpha-MEU
    p_min: float              # min_r q_r  (maxmin / Gilboa-Schmeidler)
    p_max: float              # max_r q_r  (maxmax)
    ambiguity_index: float    # p_max - p_min
    decision: str             # "TRUE", "FALSE", or "ABSTAIN"
    alpha: float
    tau: float
    bts_baseline: float       # arithmetic mean of q_r (what BTS judge would use)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "p_alpha": self.p_alpha,
            "p_min": self.p_min,
            "p_max": self.p_max,
            "ambiguity_index": self.ambiguity_index,
            "decision": self.decision,
            "alpha": self.alpha,
            "tau": self.tau,
            "bts_baseline_mean": self.bts_baseline,
        }


def alpha_meu_aggregate(
    question: str,
    debater_distributions: Sequence[Sequence[float]],
    *,
    alpha: float = 1.0,
    tau: float = 0.4,
    true_index: int = 1,
) -> AmbiguityDecision:
    """Aggregate debater probability vectors into an alpha-MEU decision.

    Args:
        question: human-readable question label (passed through).
        debater_distributions: N x m probability matrix, one row per debater.
            Each row must sum to ~1 (will be re-normalised).
        alpha: ambiguity-aversion in [0, 1]. 1.0 = Gilboa-Schmeidler maxmin,
            0.5 = Hurwicz, 0.0 = maxmax.
        tau: Bewley abstention threshold on the ambiguity index. If
            max_r q_r - min_r q_r > tau the judge abstains.
        true_index: which option (0..m-1) is "the claim is true".

    Returns:
        AmbiguityDecision with diagnostic fields.

    Raises:
        ValueError: on shape / range violations.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0,1], got {tau}")
    arr = np.asarray(debater_distributions, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(
            f"debater_distributions must be N x m with N>=1; got shape {arr.shape}"
        )
    if not (0 <= true_index < arr.shape[1]):
        raise ValueError(f"true_index {true_index} out of range [0,{arr.shape[1]})")
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each debater distribution must have positive mass")
    arr = arr / row_sums

    q = arr[:, true_index]                   # (N,) probabilities of "true"
    p_min = float(q.min())
    p_max = float(q.max())
    p_alpha = alpha * p_min + (1.0 - alpha) * p_max
    ambiguity = p_max - p_min
    bts_baseline = float(q.mean())

    if ambiguity > tau:
        decision = "ABSTAIN"
    elif p_alpha >= 0.5:
        decision = "TRUE"
    else:
        decision = "FALSE"

    return AmbiguityDecision(
        question=question,
        p_alpha=p_alpha,
        p_min=p_min,
        p_max=p_max,
        ambiguity_index=ambiguity,
        decision=decision,
        alpha=alpha,
        tau=tau,
        bts_baseline=bts_baseline,
    )


def bts_style_mean_decision(
    debater_distributions: Sequence[Sequence[float]],
    *,
    true_index: int = 1,
) -> tuple[float, str]:
    """Reference baseline: arithmetic mean of debater P(true), threshold 0.5.

    Mirrors what a BTS-trained judge does after eliciting calibrated
    probabilities from each debater (no ambiguity discount).
    """
    arr = np.asarray(debater_distributions, dtype=float)
    arr = arr / arr.sum(axis=1, keepdims=True)
    p = float(arr[:, true_index].mean())
    return p, ("TRUE" if p >= 0.5 else "FALSE")
