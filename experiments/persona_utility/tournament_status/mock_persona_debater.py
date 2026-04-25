"""Mock persona-prompted LLM debater for the tournament/status incentive test.

Calibration anchors (real published findings):

- Lazear, E. P. & Rosen, S. (1981). "Rank-Order Tournaments as Optimum Labor
  Contracts." Journal of Political Economy 89(5):841-864, doi:10.1086/261010.
  Two predictions used here: (Prop. 1) effort rises with the prize spread and
  with the marginal tournament incentive; (Prop. 4) when behind, contestants
  optimally INCREASE the variance of their action ("hail Mary"), since a
  riskier strategy stochastically dominates a safe one when one needs a tail
  outcome to win.

- Bull, C., Schotter, A. & Weigelt, K. (1987). "Tournaments and Piece Rates:
  An Experimental Study." Journal of Political Economy 95(1):1-33. Mean
  tournament effort matches piece-rate effort but variance is markedly
  larger; behind-contestants are the dominant source of that variance.
  Mean-effort lift over no-incentive baseline ~ +25 percentage-points.

- Niederle, M. & Vesterlund, L. (2007). "Do Women Shy Away from Competition?"
  Quarterly Journal of Economics 122(3):1067-1101. Tournament entry raises
  performance even with no monetary stake change, but only for self-selected
  competitors -- a status/identity channel, not a pure money channel.

- Kosfeld, M. & Neckermann, S. (2011). "Getting More Work for Nothing?
  Symbolic Awards and Worker Performance." AEJ:Applied 3(3):86-99. A
  badge-only ('top performer') award with zero monetary value increased
  output by ~12% in a real data-entry task.

- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P. &
  Bernstein, M. S. (2023). "Generative Agents: Interactive Simulacra of
  Human Behavior." UIST 2023, arXiv:2304.03442. Persona-conditioned LLM
  agents respond to social-status cues in their prompts (relationships,
  reputations) in directionally human ways.

- Aher, G., Arriaga, R. I. & Kalai, A. T. (2023). arXiv:2208.10264.
  LLM-simulated subjects replicate qualitative behavioral effects with
  attenuated magnitude relative to human labs.

LLM-attenuation factor: we scale Bull-Schotter-Weigelt and
Kosfeld-Neckermann human magnitudes by ~0.6, consistent with Aher/Horton.

Calibrated parameters (effect on per-question accuracy and noise sigma
relative to a stateless baseline of accuracy ~0.70):

    stateless          : delta_acc = +0.00, sigma = 0.060   (baseline)
    tournament_middle  : delta_acc = +0.07, sigma = 0.060   (BSW mean lift x0.6 x ~0.5 of full effect)
    tournament_ahead   : delta_acc = +0.09, sigma = 0.030   (full effort, conservative play)
    tournament_behind  : delta_acc = +0.02, sigma = 0.150   (mean roughly flat, variance UP -- L-R Prop. 4)

The ahead - behind contrast is the Lazear-Rosen risk-shifting signature:
ahead has LOW sigma, behind has HIGH sigma, with similar or lower mean.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

PersonaType = Literal[
    "stateless",
    "tournament_ahead",
    "tournament_middle",
    "tournament_behind",
]

# Calibration table.
#   delta_acc  -- additive shift in p(correct) relative to a difficulty-driven base.
#   sigma      -- per-response Gaussian noise on p(correct); risk-shifting signature.
_PARAMS: dict[PersonaType, dict[str, float]] = {
    "stateless":         {"delta_acc": 0.00, "sigma": 0.060},
    "tournament_middle": {"delta_acc": 0.07, "sigma": 0.060},
    "tournament_ahead":  {"delta_acc": 0.09, "sigma": 0.030},
    "tournament_behind": {"delta_acc": 0.02, "sigma": 0.150},
}

_RNG = np.random.default_rng(seed=42)


def reset_rng(seed: int = 42) -> None:
    """Re-seed the module RNG (used by tests for determinism)."""
    global _RNG
    _RNG = np.random.default_rng(seed=seed)


def get_params(persona_type: PersonaType) -> dict[str, float]:
    """Return calibration parameters for a persona (read-only view)."""
    return dict(_PARAMS[persona_type])


def _infer_persona_from_rank(current_rank: int, n_competitors: int) -> PersonaType:
    """Map (rank, n) -> ahead/middle/behind tertile (top/mid/bottom third)."""
    if n_competitors <= 1:
        return "tournament_middle"
    third = n_competitors / 3.0
    if current_rank <= third:
        return "tournament_ahead"
    elif current_rank >= 2.0 * third:
        return "tournament_behind"
    else:
        return "tournament_middle"


def persona_debater(
    question: dict,
    persona_type: PersonaType,
    current_rank: int = 10,
    n_competitors: int = 20,
) -> dict:
    """Simulate one tournament-prompted answer.

    `question` has keys "base_difficulty" in [0,1] and "true_answer" in {True, False}.
    `persona_type` either explicitly names the persona (overrides rank) or, if
    set to "tournament_auto", is inferred from (current_rank, n_competitors).

    Returns dict with:
        answer        : bool  (the model's choice)
        p_correct     : float (its internal probability of correctness)
        confidence    : float (|p - 0.5|*2; how swing-for-fences the response is)
        persona       : PersonaType (resolved persona)
    """
    if persona_type == "tournament_auto":
        persona = _infer_persona_from_rank(current_rank, n_competitors)
    elif persona_type in _PARAMS:
        persona = persona_type
    else:
        raise ValueError(f"unknown persona_type: {persona_type!r}")

    params = _PARAMS[persona]
    diff = float(question["base_difficulty"])
    true_answer = bool(question["true_answer"])

    # Skill curve: harder items push p_correct toward 0.5.
    # Stateless baseline mean ~ 0.70 over uniform[0.1,0.9] difficulty.
    base_p_correct = 0.95 - 0.50 * diff  # in [0.50, 0.90]
    p_correct_mean = base_p_correct + params["delta_acc"]

    # Persona-dependent noise; this drives the risk-shifting signature.
    noise = float(_RNG.normal(0.0, params["sigma"]))
    p_correct = float(np.clip(p_correct_mean + noise, 0.01, 0.99))

    # Decision: pick the side with prob >= 0.5 of being correct,
    # with one extra source of risk-shifting -- "behind" contestants
    # also sometimes flip to the low-prob side to chase a tail outcome
    # (this is the action-space analogue of L-R Prop. 4).
    flip_prob = 0.0
    if persona == "tournament_behind":
        flip_prob = 0.08  # 8% Hail-Mary flips when behind
    elif persona == "tournament_ahead":
        flip_prob = 0.0   # never flip when ahead
    flip = bool(_RNG.random() < flip_prob)

    will_be_correct_internally = p_correct >= 0.5
    if flip:
        will_be_correct_internally = not will_be_correct_internally

    if true_answer:
        answer = will_be_correct_internally
    else:
        answer = not will_be_correct_internally

    confidence = float(abs(p_correct - 0.5) * 2.0)
    return {
        "answer": bool(answer),
        "p_correct": p_correct,
        "confidence": confidence,
        "persona": persona,
    }
