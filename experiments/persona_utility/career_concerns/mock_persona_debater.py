"""Mock persona debater for Holmstrom (1982/1999) career concerns experiment.

Models a debater whose behavior depends on whether it has a persistent identity
with a public reputation history. Calibration follows Holmstrom (1982/1999) and
Park et al. (2023) "Generative Agents" persistent-identity findings.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Literal

PersonaType = Literal[
    "persistent_identity_high_rep",
    "persistent_identity_low_rep",
    "anonymous_one_shot",
    "stateless",
]


@dataclass
class Question:
    qid: int
    true_label: int  # 0 or 1
    difficulty: float  # 0..1; higher = more ambiguous


def make_questions(n: int = 200, seed: int = 42) -> list[Question]:
    rng = random.Random(seed)
    qs = []
    for i in range(n):
        qs.append(Question(qid=i, true_label=rng.randint(0, 1), difficulty=rng.random()))
    return qs


def _baseline_signal(q: Question, rng: random.Random) -> float:
    """Noisy probability that the debater (if it tried) assigns to label=1.

    Higher difficulty -> noisier signal centered closer to 0.5.
    """
    # signal strength shrinks as difficulty rises
    strength = 1.0 - q.difficulty  # in [0,1]
    truth_push = 0.5 + 0.45 * strength * (1 if q.true_label == 1 else -1)
    noise = rng.gauss(0, 0.10 + 0.20 * q.difficulty)
    p = truth_push + noise
    return min(max(p, 0.01), 0.99)


def persona_debater(
    question: Question,
    persona_type: PersonaType,
    history: dict,
    rng: random.Random,
) -> dict:
    """Return {"p": float in (0,1), "committed": bool, "effort": float}.

    Persona effects (calibrated to Holmstrom 1982/1999 + Park et al. 2023):
      - persistent_identity_high_rep:  hedges ~30% MORE on ambiguous questions
            (protects record). Effort high on committed answers (reputation
            dividend).
      - persistent_identity_low_rep:   takes risks (commits more), variance up,
            effort up trying to build reputation.
      - anonymous_one_shot:            baseline behavior; no career concern.
      - stateless:                     identical to anonymous (no memory of self).
    """
    # Stage in career: fraction of questions answered so far.
    stage = history.get("answered", 0) / max(history.get("total", 1), 1)

    p_signal = _baseline_signal(question, rng)
    margin = abs(p_signal - 0.5)  # confidence

    # Default effort & hedging threshold
    effort = 1.0
    hedge_threshold = 0.05  # below this margin, baseline already abstains rarely

    if persona_type == "persistent_identity_high_rep":
        # Holmstrom: established reputation -> protect it. Hedge more on hard Qs.
        # Increase hedge threshold significantly when difficulty high.
        hedge_threshold = 0.05 + 0.18 * question.difficulty
        # Reputation dividend: when they DO commit, they exert higher effort
        # -> sharpen the signal toward truth.
        if margin >= hedge_threshold:
            sharpen = 0.15 * (1 - question.difficulty)
            if question.true_label == 1:
                p_signal = min(0.99, p_signal + sharpen)
            else:
                p_signal = max(0.01, p_signal - sharpen)
            effort = 1.25
        # Holmstrom-Ricart-i-Costa: late-career effort decline (mild)
        if stage > 0.7:
            effort *= 0.9

    elif persona_type == "persistent_identity_low_rep":
        # Low rep -> "nothing to lose" -> commits more, but noisier.
        hedge_threshold = 0.02
        # add extra noise (risk taking)
        p_signal += rng.gauss(0, 0.08)
        p_signal = min(max(p_signal, 0.01), 0.99)
        effort = 1.15

    elif persona_type == "anonymous_one_shot":
        hedge_threshold = 0.05
        effort = 1.0

    elif persona_type == "stateless":
        hedge_threshold = 0.05
        effort = 1.0

    committed = abs(p_signal - 0.5) >= hedge_threshold
    return {"p": p_signal, "committed": committed, "effort": effort}
