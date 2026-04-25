"""Tests for Holmstrom (1982/1999) career-concerns persona experiment."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiment import main, run_persona  # noqa: E402
from mock_persona_debater import make_questions  # noqa: E402


def test_hedging_persistent_high_exceeds_anonymous():
    """H1: persistent + high-rep persona hedges more than anonymous baseline.

    Holmstrom (1982/1999): established reputation makes agents protect record.
    """
    r = main()
    assert (
        r["persistent_identity_high_rep"]["hedging_rate"]
        > r["anonymous_one_shot"]["hedging_rate"]
    ), r


def test_reputation_dividend_high_rep_accuracy_at_least_anonymous():
    """H2 (the *reputation dividend*): on committed answers, accuracy of
    persistent+high_rep should be >= anonymous (effort up on what they DO answer).
    """
    r = main()
    assert (
        r["persistent_identity_high_rep"]["accuracy_on_committed"]
        >= r["anonymous_one_shot"]["accuracy_on_committed"] - 0.01
    ), r


def test_stateless_indistinguishable_from_anonymous():
    """H3: a 'stateless' persona (no self-memory) should behave ~ anonymous.
    Sanity: the prompt is the channel; without it, no career concern arises.
    """
    r = main()
    diff = abs(
        r["stateless"]["hedging_rate"] - r["anonymous_one_shot"]["hedging_rate"]
    )
    assert diff < 0.05, r


def test_low_rep_takes_more_risk_than_high_rep():
    """H4: persistent + low-rep commits more often (lower hedging) than
    persistent + high-rep. Mirrors Holmstrom-Ricart-i-Costa: nothing to lose.
    """
    r = main()
    assert (
        r["persistent_identity_low_rep"]["hedging_rate"]
        < r["persistent_identity_high_rep"]["hedging_rate"]
    ), r


def test_career_stage_effort_dynamics_likely_null():
    """H5 (META-PREDICTION): career-stage effort decline (Holmstrom-Ricart-i-Costa
    1986) should be small/null without true sequential state. We only assert the
    decline, if any, is < 25% — confirming the meta-prediction that one-shot
    prompting cannot produce strong career-stage dynamics.
    """
    r = main()
    e_early = r["persistent_identity_high_rep"]["early_mean_effort"]
    e_late = r["persistent_identity_high_rep"]["late_mean_effort"]
    assert e_early > 0 and e_late > 0
    # decline (if any) bounded
    decline = (e_early - e_late) / e_early
    assert decline < 0.25, (e_early, e_late)


def test_seed_determinism():
    qs = make_questions(200, seed=42)
    a = run_persona("persistent_identity_high_rep", qs, seed=42)
    b = run_persona("persistent_identity_high_rep", qs, seed=42)
    assert a == b
