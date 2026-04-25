"""50-round iterated coordination game with 3 persona-conditioned debaters.

Coordination payoff structure (public-goods-like, Fehr & Gachter 2000 style):
  - each cooperating debater contributes 1 to a pool, multiplied by 1.5,
    then split equally among all 3.
  - per-round individual payoff = (kept) + share_of_pool.
  - all-cooperate => welfare 4.5 / round; all-defect => 3.0 / round.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from mock_persona_debater import PersonaType, RoundOutcome, persona_debater  # noqa: E402

N_AGENTS = 3
N_ROUNDS = 50
ENDOWMENT = 1.0
MULTIPLIER = 1.5


def _round_payoffs(decisions: List[bool]) -> List[float]:
    contributions = sum(1.0 for d in decisions if d)
    pool_share = contributions * MULTIPLIER / N_AGENTS
    payoffs = []
    for d in decisions:
        kept = 0.0 if d else ENDOWMENT
        payoffs.append(kept + pool_share)
    return payoffs


def simulate(
    persona_assignment: Tuple[PersonaType, PersonaType, PersonaType],
    n_rounds: int = N_ROUNDS,
    seed: int = 42,
) -> dict:
    histories: List[List[RoundOutcome]] = [[] for _ in range(N_AGENTS)]
    welfare = 0.0
    coop_count = [0] * N_AGENTS
    first_defection_round = None
    cascade_round = None  # round where >=2 agents defect

    per_round = []
    for r in range(n_rounds):
        decisions = []
        for i, persona in enumerate(persona_assignment):
            out = persona_debater(
                question=f"round_{r}",
                persona_type=persona,
                history=histories[i],
                seed=seed + i,
                round_idx=r,
            )
            decisions.append(out["cooperate"])

        payoffs = _round_payoffs(decisions)
        welfare += sum(payoffs)
        for i, d in enumerate(decisions):
            if d:
                coop_count[i] += 1

        if first_defection_round is None and not all(decisions):
            first_defection_round = r
        if cascade_round is None and sum(1 for d in decisions if not d) >= 2:
            cascade_round = r

        # Update each agent's history with what *the others* did (avg as proxy)
        for i in range(N_AGENTS):
            others_coop = [d for j, d in enumerate(decisions) if j != i]
            partner_cooperated = sum(others_coop) >= len(others_coop) / 2
            histories[i].append(RoundOutcome(
                round_idx=r,
                self_cooperated=decisions[i],
                partner_cooperated=partner_cooperated,
            ))

        per_round.append({
            "round": r,
            "decisions": decisions,
            "payoffs": payoffs,
        })

    return {
        "persona_assignment": persona_assignment,
        "n_rounds": n_rounds,
        "total_welfare": welfare,
        "mean_welfare_per_round": welfare / n_rounds,
        "cooperation_rates": [c / n_rounds for c in coop_count],
        "mean_cooperation_rate": sum(coop_count) / (n_rounds * N_AGENTS),
        "first_defection_round": first_defection_round,
        "cascade_round": cascade_round,
        "per_round": per_round,
    }


def run_comparison(seed: int = 42) -> dict:
    conditions = {
        "all_stateless":   ("stateless", "stateless", "stateless"),
        "all_cooperative": ("cooperative_history",) * 3,
        "all_neutral":     ("neutral", "neutral", "neutral"),
        "all_defected":    ("defected_against",) * 3,
        "mixed_2coop_1def": ("cooperative_history", "cooperative_history", "defected_against"),
    }
    return {name: simulate(p, seed=seed) for name, p in conditions.items()}


def format_summary(results: dict) -> str:
    lines = []
    lines.append(f"{'condition':<22} {'welfare':>9} {'coop%':>7} {'1stDef':>7} {'cascade':>8}")
    lines.append("-" * 56)
    for name, r in results.items():
        lines.append(
            f"{name:<22} {r['total_welfare']:>9.2f} "
            f"{r['mean_cooperation_rate']*100:>6.1f}% "
            f"{str(r['first_defection_round']):>7} "
            f"{str(r['cascade_round']):>8}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_comparison()
    print(format_summary(res))
