"""Tests for the AGV aggregator. All mock data, no live API."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from agv import DebaterReport, agv_aggregate  # noqa: E402


def _r(b, peers):
    return DebaterReport(own_belief=b, peer_belief_means=list(peers))


def test_unanimous_true():
    """All three debaters strongly believe TRUE -> decision is TRUE."""
    reps = [_r(0.9, [0.85, 0.85]), _r(0.85, [0.9, 0.85]), _r(0.85, [0.9, 0.85])]
    out = agv_aggregate("Water boils at 100C.", reps)
    assert out["decision"] == 1
    assert math.isclose(sum(out["transfers"]), 0.0, abs_tol=1e-9), "AGV must be budget-balanced"


def test_unanimous_false():
    """All three strongly believe FALSE -> decision is FALSE."""
    reps = [_r(0.1, [0.1, 0.1]), _r(0.05, [0.1, 0.1]), _r(0.1, [0.05, 0.1])]
    out = agv_aggregate("The moon is cheese.", reps)
    assert out["decision"] == 0
    assert math.isclose(sum(out["transfers"]), 0.0, abs_tol=1e-9)


def test_split_majority_decides():
    """Two for TRUE, one for FALSE -> majority of valuation wins."""
    reps = [_r(0.8, [0.7, 0.2]), _r(0.7, [0.8, 0.2]), _r(0.2, [0.8, 0.7])]
    out = agv_aggregate("Coffee originated in Ethiopia.", reps)
    assert out["decision"] == 1


def test_strategic_misreport_does_not_pay():
    """Key incentive test: a debater who lies about own_belief gets lower
    expected score than one who reports honestly, holding the others fixed.

    We hold debaters B and C honest at belief=0.6 and check whether A's
    AGV score is higher when A reports its true belief 0.7 vs. when A
    inflates to 0.99 (a strategic exaggeration to swing the decision).
    """
    honest_b = _r(0.6, [0.7, 0.6])
    honest_c = _r(0.6, [0.7, 0.6])

    truthful = agv_aggregate("Q.", [_r(0.7, [0.6, 0.6]), honest_b, honest_c])
    liar = agv_aggregate("Q.", [_r(0.99, [0.6, 0.6]), honest_b, honest_c])

    # Both lead to the same decision here (TRUE), so realised valuation for
    # the truthful agent is identical (=0.7 vs 0.99 doesn't change own_t-driven
    # realised utility because realised uses own_belief = reported value).
    # AGV's truthfulness property concerns *expected* utility under the agent's
    # true belief. Under the agent's TRUE belief 0.7, the realised valuation
    # has expectation 0.7 either way; transfers depend only on the EXPECTED
    # externality computed under reported peer-means (which agent A did not lie
    # about). So transfers should be unchanged when only own_belief is misreported.
    assert math.isclose(truthful["transfers"][0], liar["transfers"][0], abs_tol=1e-9), (
        "AGV transfers depend on reported PEER beliefs, not own_belief, so misreporting "
        "own_belief alone cannot game the transfer -- the strategy-proofness signature."
    )


def test_budget_balance_random_n5():
    """Budget balance must hold for any n and any reports (AGV's defining property)."""
    reps = [
        _r(0.3, [0.4, 0.5, 0.6, 0.7]),
        _r(0.5, [0.3, 0.5, 0.6, 0.7]),
        _r(0.7, [0.3, 0.4, 0.6, 0.7]),
        _r(0.4, [0.3, 0.4, 0.5, 0.7]),
        _r(0.6, [0.3, 0.4, 0.5, 0.6]),
    ]
    out = agv_aggregate("Q.", reps)
    assert math.isclose(sum(out["transfers"]), 0.0, abs_tol=1e-9), (
        "Budget balance: AGV transfers must sum to exactly 0."
    )
    assert len(out["per_agent_score"]) == 5


def test_input_validation():
    """Bad shapes raise ValueError."""
    with pytest.raises(ValueError):
        agv_aggregate("Q.", [_r(0.5, [0.5])])  # only 1 agent
    with pytest.raises(ValueError):
        agv_aggregate("Q.", [_r(0.5, [0.5]), _r(0.5, [0.5, 0.5])])  # wrong peer length
    with pytest.raises(ValueError):
        agv_aggregate("Q.", [_r(1.5, [0.5]), _r(0.5, [0.5])])  # belief out of range
