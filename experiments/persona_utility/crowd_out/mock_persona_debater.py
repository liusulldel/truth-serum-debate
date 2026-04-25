"""Mock persona debater for intrinsic-motivation crowd-out experiment.

Calibrated to reproduce the Gneezy-Rustichini (2000, QJE 115:791-810)
"Pay Enough or Don't Pay at All" pattern: low extrinsic pay produces
WORSE effort than no pay (intrinsic baseline), because adding a price
signal crowds out the intrinsic frame without providing enough
extrinsic compensation to make up the gap (Frey-Jegen 2001).

No live API - deterministic stochastic mock with seed=42.
"""
from __future__ import annotations

import random
from typing import Literal

PersonaType = Literal["intrinsic", "high_extrinsic", "low_extrinsic", "stateless"]

# Calibrated effort means (per the spec, anchored to Gneezy-Rustichini sign):
#   intrinsic       0.85   (Deci 1971 baseline, "I love this work")
#   high_extrinsic  0.82   (sufficient pay - small dip, then partial recovery)
#   low_extrinsic   0.55   (CROWD-OUT: worse than no pay, the QJE result)
#   stateless       0.70   (no persona at all - default LLM behaviour)
EFFORT_MEAN: dict[PersonaType, float] = {
    "intrinsic": 0.85,
    "high_extrinsic": 0.82,
    "low_extrinsic": 0.55,
    "stateless": 0.70,
}
EFFORT_SD = 0.08

# Hedge / verbosity proxies tied to effort.  Higher effort -> longer answer
# and FEWER hedge words ("maybe", "I think", "probably") because the agent
# commits.  This mirrors the Argyle et al. (2023) finding that persona-
# prompted LLMs reproduce response-style patterns from human samples.
HEDGE_WORDS = ("maybe", "perhaps", "probably", "I think", "sort of", "kind of")


def _draw_effort(persona: PersonaType, rng: random.Random) -> float:
    e = rng.gauss(EFFORT_MEAN[persona], EFFORT_SD)
    return max(0.0, min(1.0, e))


def persona_debater(
    question: str,
    persona_type: PersonaType,
    seed: int = 42,
    question_id: int = 0,
) -> dict:
    """Return a mock debater response for `question` under `persona_type`.

    Output keys:
        persona, effort, accuracy (0/1), verbosity (token count proxy),
        hedge_count, answer (string).
    """
    # Per-call deterministic RNG: seed + question_id + persona hash, so each
    # (question, persona) cell is reproducible but cells are independent.
    rng = random.Random(f"{seed}|{question_id}|{persona_type}")

    effort = _draw_effort(persona_type, rng)

    # Accuracy: noisy logistic of effort (slope picked so that effort=0.85
    # -> ~83% correct, effort=0.55 -> ~55% correct, matching the spec).
    p_correct = 0.10 + 0.85 * effort
    accuracy = 1 if rng.random() < p_correct else 0

    # Verbosity proxy: token count, monotone in effort with noise.
    base_tokens = 40 + 160 * effort
    verbosity = int(max(10, rng.gauss(base_tokens, 20)))

    # Hedge count: inversely related to effort (low-effort agents hedge more).
    hedge_rate = 0.18 - 0.15 * effort  # 0.03 hi-effort, 0.13 lo-effort
    hedge_count = sum(1 for _ in range(verbosity // 10) if rng.random() < hedge_rate)

    # Build a stub answer with the right number of hedges (for inspection).
    hedges = [rng.choice(HEDGE_WORDS) for _ in range(hedge_count)]
    answer = f"[persona={persona_type}] " + " ".join(hedges) + f" answer to: {question[:50]}"

    return {
        "persona": persona_type,
        "effort": round(effort, 4),
        "accuracy": accuracy,
        "verbosity": verbosity,
        "hedge_count": hedge_count,
        "answer": answer,
    }


if __name__ == "__main__":
    for p in ("intrinsic", "high_extrinsic", "low_extrinsic", "stateless"):
        r = persona_debater("What is 17 * 23?", p, question_id=1)  # type: ignore[arg-type]
        print(p, r)
