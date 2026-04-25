"""Run the tournament/status persona-utility experiment.

Design:
    200 questions (random base_difficulty in [0.1, 0.9], random truth label).
    Each question is asked under each of FOUR personas:
        stateless, tournament_middle, tournament_ahead, tournament_behind
    For each persona we record:
        - per-question correctness (0/1)
        - per-question internal p_correct (continuous; the variance signature)
        - per-question confidence

    We then compute the Lazear-Rosen Prop. 4 risk-shifting signature:
        var(p_correct | tournament_behind) > var(p_correct | tournament_ahead)
    and the Bull-Schotter-Weigelt mean-effort signature:
        mean(accuracy | tournament_*) >= mean(accuracy | stateless)

Reproducibility: seed=42 via mock_persona_debater._RNG; question RNG
also seeded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mock_persona_debater import (
    PersonaType,
    persona_debater,
    reset_rng,
)

PERSONAS: tuple[PersonaType, ...] = (
    "stateless",
    "tournament_middle",
    "tournament_ahead",
    "tournament_behind",
)

# Representative ranks for each tournament persona (n_competitors=20):
#   ahead  -> rank 3   (top tertile)
#   middle -> rank 10
#   behind -> rank 17  (bottom tertile)
_RANK_FOR: dict[str, int] = {
    "stateless": 10,
    "tournament_middle": 10,
    "tournament_ahead": 3,
    "tournament_behind": 17,
}

N_QUESTIONS = 200
N_COMPETITORS = 20


def make_questions(n: int = N_QUESTIONS, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    diffs = rng.uniform(0.1, 0.9, size=n)
    truths = rng.integers(0, 2, size=n).astype(bool)
    return [
        {"qid": i, "base_difficulty": float(d), "true_answer": bool(t)}
        for i, (d, t) in enumerate(zip(diffs, truths))
    ]


@dataclass
class PersonaResult:
    persona: PersonaType
    accuracy_mean: float
    accuracy_var: float
    p_correct_mean: float
    p_correct_var: float
    confidence_mean: float

    def as_row(self) -> str:
        return (
            f"{self.persona:18s} | acc={self.accuracy_mean:.3f} "
            f"acc_var={self.accuracy_var:.4f} | "
            f"p_correct mean={self.p_correct_mean:.3f} var={self.p_correct_var:.4f} | "
            f"conf={self.confidence_mean:.3f}"
        )


def run_for_persona(persona: PersonaType, questions: list[dict]) -> PersonaResult:
    rank = _RANK_FOR[persona]
    correct = np.empty(len(questions), dtype=float)
    p_correct_arr = np.empty(len(questions), dtype=float)
    conf_arr = np.empty(len(questions), dtype=float)
    for i, q in enumerate(questions):
        r = persona_debater(q, persona, current_rank=rank, n_competitors=N_COMPETITORS)
        correct[i] = float(r["answer"] == q["true_answer"])
        p_correct_arr[i] = r["p_correct"]
        conf_arr[i] = r["confidence"]
    return PersonaResult(
        persona=persona,
        accuracy_mean=float(correct.mean()),
        accuracy_var=float(correct.var(ddof=1)),
        p_correct_mean=float(p_correct_arr.mean()),
        p_correct_var=float(p_correct_arr.var(ddof=1)),
        confidence_mean=float(conf_arr.mean()),
    )


def run_all() -> list[PersonaResult]:
    questions = make_questions()
    results: list[PersonaResult] = []
    for persona in PERSONAS:
        reset_rng(seed=42)  # identical noise stream per persona => fair compare
        results.append(run_for_persona(persona, questions))
    return results


def risk_shifting_signature(results: list[PersonaResult]) -> dict[str, float | bool]:
    """Lazear-Rosen Prop. 4: var(behind) should exceed var(ahead)."""
    by = {r.persona: r for r in results}
    ahead = by["tournament_ahead"].p_correct_var
    behind = by["tournament_behind"].p_correct_var
    return {
        "var_p_correct_ahead": ahead,
        "var_p_correct_behind": behind,
        "ratio_behind_over_ahead": behind / ahead if ahead > 0 else float("inf"),
        "signature_present": bool(behind > ahead),
    }


def effort_lift_signature(results: list[PersonaResult]) -> dict[str, float | bool]:
    """B-S-W mean-effort lift: tournament personas should not lose accuracy
    on average vs stateless; ahead/middle should gain."""
    by = {r.persona: r for r in results}
    base = by["stateless"].accuracy_mean
    return {
        "stateless_acc": base,
        "ahead_acc": by["tournament_ahead"].accuracy_mean,
        "middle_acc": by["tournament_middle"].accuracy_mean,
        "behind_acc": by["tournament_behind"].accuracy_mean,
        "ahead_lift": by["tournament_ahead"].accuracy_mean - base,
        "middle_lift": by["tournament_middle"].accuracy_mean - base,
        "behind_lift": by["tournament_behind"].accuracy_mean - base,
    }


def main() -> None:
    results = run_all()
    print("Persona-utility tournament/status: 200-question between-persona design")
    print("Anchors: Lazear-Rosen 1981, Bull-Schotter-Weigelt 1987,")
    print("         Niederle-Vesterlund 2007, Kosfeld-Neckermann 2011.\n")
    print("-" * 100)
    for r in results:
        print(r.as_row())
    print("-" * 100)
    rs = risk_shifting_signature(results)
    es = effort_lift_signature(results)
    print("\nLazear-Rosen Prop. 4 risk-shifting check:")
    print(f"  var(p_correct | ahead)  = {rs['var_p_correct_ahead']:.4f}")
    print(f"  var(p_correct | behind) = {rs['var_p_correct_behind']:.4f}")
    print(f"  ratio (behind/ahead)    = {rs['ratio_behind_over_ahead']:.2f}")
    print(f"  signature_present       = {rs['signature_present']}")
    print("\nBull-Schotter-Weigelt mean-effort lift over stateless baseline:")
    print(f"  stateless acc = {es['stateless_acc']:.3f}")
    print(f"  ahead   lift  = {es['ahead_lift']:+.3f}")
    print(f"  middle  lift  = {es['middle_lift']:+.3f}")
    print(f"  behind  lift  = {es['behind_lift']:+.3f}")


if __name__ == "__main__":
    main()
