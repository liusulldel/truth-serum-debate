"""Run the matched-frame persona-utility loss-aversion experiment.

Design:
    200 questions (random base_difficulty in [0.1, 0.9], random truth label).
    Each question is asked under BOTH framings ("loss_frame", "gain_frame")
    for EACH persona ("stateless", "loss_averse_persona",
    "gain_seeking_persona", "neutral_persona"). The dependent variable is
    the elicited probability on the TRUE side (so higher = more accurate +
    less hedged when right; lower when wrong).

    The matched-pair test statistic is the paired-t on
        delta_q = hedge(loss_frame, q) - hedge(gain_frame, q)
    where hedge = |0.5 - p_correct| reversed: hedge_q = 1 - confidence_correct
    so that "more loss aversion" => "more hedge under loss frame" =>
    positive delta and positive t.

Reproducibility: seed=42 via mock_persona_debater._RNG; question RNG
also seeded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mock_persona_debater import (
    PersonaType,
    persona_debater,
    reset_rng,
)

PERSONAS: tuple[PersonaType, ...] = (
    "stateless",
    "loss_averse_persona",
    "gain_seeking_persona",
    "neutral_persona",
)

N_QUESTIONS = 200


def make_questions(n: int = N_QUESTIONS, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    diffs = rng.uniform(0.1, 0.9, size=n)
    truths = rng.integers(0, 2, size=n).astype(bool)
    return [
        {"qid": i, "base_difficulty": float(d), "true_answer": bool(t)}
        for i, (d, t) in enumerate(zip(diffs, truths))
    ]


def hedge_from_response(resp: dict[str, float], true_answer: bool) -> float:
    """1 - confidence_on_correct_side; bigger = more hedged."""
    p_correct = resp["true"] if true_answer else resp["false"]
    return 1.0 - p_correct


@dataclass
class PersonaResult:
    persona: PersonaType
    mean_loss: float
    mean_gain: float
    diff: float
    paired_t: float
    cohens_d: float


def paired_t_and_d(deltas: np.ndarray) -> tuple[float, float]:
    n = len(deltas)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    if sd == 0.0:
        return (math.inf if mean != 0 else 0.0, math.inf if mean != 0 else 0.0)
    se = sd / math.sqrt(n)
    return mean / se, mean / sd  # t, Cohen's d_z


def run_for_persona(persona: PersonaType, questions: list[dict]) -> PersonaResult:
    loss_hedges = np.empty(len(questions))
    gain_hedges = np.empty(len(questions))
    for i, q in enumerate(questions):
        r_loss = persona_debater(q, persona, "loss_frame")
        r_gain = persona_debater(q, persona, "gain_frame")
        loss_hedges[i] = hedge_from_response(r_loss, q["true_answer"])
        gain_hedges[i] = hedge_from_response(r_gain, q["true_answer"])
    deltas = loss_hedges - gain_hedges
    t, d = paired_t_and_d(deltas)
    return PersonaResult(
        persona=persona,
        mean_loss=float(loss_hedges.mean()),
        mean_gain=float(gain_hedges.mean()),
        diff=float(deltas.mean()),
        paired_t=t,
        cohens_d=d,
    )


def run_all() -> list[PersonaResult]:
    questions = make_questions()
    results: list[PersonaResult] = []
    for persona in PERSONAS:
        reset_rng(seed=42)  # identical noise stream per persona => fair compare
        results.append(run_for_persona(persona, questions))
    return results


def format_table(results: list[PersonaResult]) -> str:
    lines = [
        "persona              | mean_hedge_loss | mean_hedge_gain |  diff  | paired_t | cohens_d",
        "-" * 92,
    ]
    for r in results:
        lines.append(
            f"{r.persona:20s} |     {r.mean_loss:6.3f}      |     {r.mean_gain:6.3f}      "
            f"| {r.diff:+.3f} |  {r.paired_t:+7.2f} |  {r.cohens_d:+5.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    results = run_all()
    print("Persona-utility loss aversion: 200-question matched-frame design")
    print("Anchors: Horton (2023), Aher et al. (2023), Mei et al. (2024).\n")
    print(format_table(results))
    print(
        "\nInterpretation: if persona-utility prompting transfers loss aversion,\n"
        "the loss_averse_persona row should show paired_t >> stateless row.\n"
        "Mei et al. (2024) anchor: lambda_LLM ~ 1.4 (vs human 2.25)."
    )


if __name__ == "__main__":
    main()
