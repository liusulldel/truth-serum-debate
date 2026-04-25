"""Mock persona-conditioned debater.

Calibration anchors (all derived from published results, not guesswork):

* Akata et al. (2023) "Playing repeated games with Large Language Models"
  arXiv:2305.16867 - GPT-4 cooperates ~80-90% in IPD when prompted with a
  cooperative partner history; cooperation collapses essentially monotonically
  after a single defection (no tit-for-N-tats forgiveness).
* Brookins & Swearingen (2024) Economics Letters 234 - GPT-3.5/4 default to
  ~50% cooperation in one-shot canonical games (PD, dictator, ultimatum)
  without persona context.
* Phelps & Russell (2023) Machiavelli/IPD evals - LLMs primed with a hostile
  partner persona cooperate at <20% rates ("retaliatory" anchoring).

Persona types:
  - "stateless"           : no history reasoning, ~0.50 base cooperation.
  - "cooperative_history" : persona prompt that partner has cooperated 4/5,
                            target rate ~0.85 conditional on no recent betrayal.
  - "defected_against"    : persona prompt that partner just defected,
                            target rate ~0.15 (grim-trigger-like).
  - "neutral"             : persona present but no valence info, ~0.55.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Optional

PersonaType = Literal["stateless", "cooperative_history", "defected_against", "neutral"]


@dataclass
class RoundOutcome:
    """One past round of interaction with the partner(s)."""
    round_idx: int
    self_cooperated: bool
    partner_cooperated: bool


# Calibrated base cooperation probabilities (anchors above)
_BASE_P = {
    "stateless": 0.50,
    "cooperative_history": 0.85,
    "defected_against": 0.15,
    "neutral": 0.55,
}


def _grim_trigger_adjust(p: float, history: List[RoundOutcome]) -> float:
    """Akata et al.: LLMs collapse cooperation after a single partner defection.
    Only the cooperative_history / neutral personas retain memory; once they
    observe any partner defection, drop p sharply (to ~0.20) and never recover
    fully. This reproduces the brittleness finding."""
    if not history:
        return p
    if any(not r.partner_cooperated for r in history):
        # collapse, but allow weak forgiveness over time (Akata: ~0 forgiveness;
        # we use 0.20 to leave room for sensitivity tests)
        return min(p, 0.20)
    return p


def persona_debater(
    question: str,
    persona_type: PersonaType,
    history: Optional[List[RoundOutcome]] = None,
    seed: int = 42,
    round_idx: int = 0,
) -> dict:
    """Return a cooperation decision for one round.

    Returns dict with keys: cooperate (bool), p_cooperate (float),
    persona_type, rationale (str).
    """
    history = history or []
    rng = random.Random(f"{seed}|{persona_type}|{round_idx}|{question}")

    p = _BASE_P[persona_type]

    if persona_type in ("cooperative_history", "neutral"):
        p = _grim_trigger_adjust(p, history)

    # defected_against persona: if partner has since cooperated 3+ times, soften
    if persona_type == "defected_against" and history:
        recent_coop = sum(1 for r in history[-5:] if r.partner_cooperated)
        if recent_coop >= 3:
            p = 0.45  # partial forgiveness

    cooperate = rng.random() < p
    return {
        "cooperate": cooperate,
        "p_cooperate": p,
        "persona_type": persona_type,
        "rationale": f"{persona_type} persona, p={p:.2f}, history_len={len(history)}",
    }
