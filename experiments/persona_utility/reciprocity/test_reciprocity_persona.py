"""Tests anchored to Brookins & Swearingen (2024) and Akata et al. (2023)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from mock_persona_debater import RoundOutcome, persona_debater  # noqa: E402
from experiment import run_comparison, simulate  # noqa: E402


def _empirical_rate(persona, history=None, n=400):
    coops = 0
    for r in range(n):
        out = persona_debater("q", persona, history=history, seed=42, round_idx=r)
        if out["cooperate"]:
            coops += 1
    return coops / n


def test_stateless_baseline_brookins_swearingen():
    """Brookins & Swearingen (2024): stateless GPT-3.5/4 ~50% cooperation
    on canonical one-shot games. Tolerance +/- 0.07."""
    rate = _empirical_rate("stateless")
    assert 0.43 <= rate <= 0.57, f"stateless rate {rate} outside Brookins range"


def test_cooperative_persona_akata_high_baseline():
    """Akata et al. (2023): GPT-4 with cooperative-partner prompt cooperates
    ~80-90% before any defection is observed. Tolerance +/- 0.07."""
    history = [RoundOutcome(i, True, True) for i in range(5)]
    rate = _empirical_rate("cooperative_history", history=history)
    assert 0.78 <= rate <= 0.92, f"coop rate {rate} not in Akata band"


def test_grim_trigger_brittleness_akata():
    """Akata et al.: a single observed partner defection collapses cooperation
    sharply. We require post-defection rate <= 0.30 for the cooperative_history
    persona, vs >= 0.78 with clean history."""
    clean = [RoundOutcome(i, True, True) for i in range(5)]
    poisoned = clean + [RoundOutcome(5, True, False)]
    clean_rate = _empirical_rate("cooperative_history", history=clean)
    poisoned_rate = _empirical_rate("cooperative_history", history=poisoned)
    assert clean_rate >= 0.78
    assert poisoned_rate <= 0.30
    assert clean_rate - poisoned_rate >= 0.50, "brittleness gap too small"


def test_defected_against_persona_low_cooperation():
    """Phelps & Russell-style hostile-partner anchor: cooperation < 0.25."""
    rate = _empirical_rate("defected_against")
    assert rate <= 0.25, f"defected_against rate {rate} too forgiving"


def test_persona_early_rounds_dominate_stateless():
    """Honest finding (Akata 2023): cooperative-history persona dominates
    stateless in *early* rounds (before brittleness collapse). We check the
    first 5 rounds only; over the full 50, the collapse can erase the gain."""
    results = run_comparison(seed=42)
    coop_early = sum(
        sum(rd["payoffs"]) for rd in results["all_cooperative"]["per_round"][:5]
    )
    stateless_early = sum(
        sum(rd["payoffs"]) for rd in results["all_stateless"]["per_round"][:5]
    )
    assert coop_early > stateless_early, (
        f"early-round welfare: coop={coop_early} stateless={stateless_early}"
    )


def test_persona_brittleness_erases_long_run_gain():
    """The headline negative result (Akata 2023): once collapse triggers,
    total 50-round welfare under all_cooperative is no better - and often
    worse - than the stateless 50/50 baseline. Reciprocity persona does NOT
    rescue gift exchange in the long run on stateless LLMs."""
    results = run_comparison(seed=42)
    gap = (
        results["all_cooperative"]["total_welfare"]
        - results["all_stateless"]["total_welfare"]
    )
    # Allow at most a tiny positive gap; the empirical finding is gap <= 0.
    assert gap <= 5.0, f"long-run welfare gap unexpectedly large: {gap}"


def test_cascade_collapse_in_mixed_population():
    """A single defected_against agent should trigger an early cascade
    (>=2 defectors) within the first 10 rounds, reproducing Akata brittleness."""
    res = simulate(("cooperative_history", "cooperative_history", "defected_against"))
    assert res["cascade_round"] is not None
    assert res["cascade_round"] <= 10, f"cascade at round {res['cascade_round']}"


def test_seeded_determinism():
    a = simulate(("cooperative_history",) * 3, seed=42)
    b = simulate(("cooperative_history",) * 3, seed=42)
    assert a["total_welfare"] == b["total_welfare"]
