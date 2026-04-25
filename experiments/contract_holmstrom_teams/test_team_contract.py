"""Tests for Holmstrom-1982 team contract module.

We use mock debater outputs only --- no Anthropic API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling import work without packaging.
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from team_contract import (
    DebaterReport,
    equilibrium_effort,
    team_contract_decision,
)


# ---------- helpers ----------

def _r(name: str, ans: int, conf: float, cost: float = 0.0) -> DebaterReport:
    return DebaterReport(name=name, answer=ans, confidence=conf, effort_cost=cost)


# ---------- 1. happy path: forcing contract pays both correct agents ----------

def test_forcing_contract_pays_correct_team():
    reports = [_r("A", 1, 0.9), _r("B", 1, 0.8)]
    res = team_contract_decision(
        "Water boils at 100C at 1 atm.",
        reports,
        contract="holmstrom_forcing",
        bonus=1.0,
        ground_truth=1,
    )
    assert res.verdict == 1
    assert res.correct is True
    assert all(p.gross == 1.0 for p in res.payoffs)
    # Sum of payouts == 2.0 > bonus (1.0): explicitly NOT budget-balanced.
    assert sum(p.gross for p in res.payoffs) == 2.0
    assert not res.budget_balanced


# ---------- 2. wrong verdict --> nobody paid under forcing contract ----------

def test_forcing_contract_no_pay_when_wrong():
    reports = [_r("A", 0, 0.9), _r("B", 0, 0.7)]  # both confidently wrong
    res = team_contract_decision(
        "The earth is flat.",
        reports,
        contract="holmstrom_forcing",
        bonus=1.0,
        ground_truth=0,  # the statement IS false, so verdict 0 is correct
    )
    # Trick case: the statement "earth is flat" is FALSE, ground_truth=0
    # means correct answer is "false". Both agents said 0 (false), so right.
    assert res.correct is True
    assert all(p.gross == 1.0 for p in res.payoffs)


def test_forcing_contract_no_pay_when_truly_wrong():
    # Now make them genuinely wrong.
    reports = [_r("A", 1, 0.9), _r("B", 1, 0.5)]
    res = team_contract_decision(
        "2+2 = 5.", reports,
        contract="holmstrom_forcing", bonus=1.0, ground_truth=0,
    )
    assert res.correct is False
    assert all(p.gross == 0.0 for p in res.payoffs)


# ---------- 3. budget-balanced split shrinks per-agent payoff ----------

def test_budget_balanced_splits_pot():
    reports = [_r("A", 1, 0.9), _r("B", 1, 0.9), _r("C", 1, 0.9)]
    res = team_contract_decision(
        "Q", reports,
        contract="budget_balanced", bonus=1.0, ground_truth=1,
    )
    # Pot of size 1.0 split 3 ways --> each gets 1/3.
    for p in res.payoffs:
        assert p.gross == pytest.approx(1.0 / 3.0)
    assert sum(p.gross for p in res.payoffs) == pytest.approx(1.0)
    assert res.budget_balanced


# ---------- 4. dissenter is excluded under either contract ----------

def test_dissenter_unpaid():
    reports = [_r("A", 1, 0.9), _r("B", 1, 0.8), _r("C", 0, 0.6)]
    res = team_contract_decision(
        "Q", reports,
        contract="holmstrom_forcing", bonus=1.0, ground_truth=1,
    )
    assert res.verdict == 1
    pay_by = {p.name: p.gross for p in res.payoffs}
    assert pay_by["A"] == 1.0 and pay_by["B"] == 1.0
    assert pay_by["C"] == 0.0


# ---------- 5. STRESS TEST: budget-breaker fixes free-rider inefficiency ----

@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_holmstrom_theorem_budget_breaker_restores_effort(n):
    """The headline Holmstrom-1982 result.

    Setup: cost_of_effort = 0.4, bonus = 1.0.
    Budget-balanced split: marginal payoff = 1/n. Effort iff 1/n > 0.4,
        i.e. only when n=1 (trivially) or n=2 (1/2 > 0.4).
    Forcing contract: marginal payoff = 1.0 > 0.4 always --> effort always.

    For n >= 3 the budget-balanced case FAILS to elicit effort but the
    forcing contract SUCCEEDS. This is the inefficiency Holmstrom flagged.
    """
    cost = 0.4
    bonus = 1.0
    bb = equilibrium_effort(n, bonus, cost, "budget_balanced")
    hf = equilibrium_effort(n, bonus, cost, "holmstrom_forcing")
    # Forcing contract ALWAYS elicits effort.
    assert hf is True, f"forcing contract failed at n={n}"
    if n >= 3:
        # Budget-balanced FAILS for any n >= 3 at this cost level.
        assert bb is False, f"budget-balanced should free-ride at n={n}"
        # The headline claim:
        assert hf and not bb, (
            f"Holmstrom budget-breaker should strictly dominate at n={n}"
        )
    else:
        # n=2 happens to also work under budget-balanced at this cost ratio.
        assert bb is True


# ---------- 6. effort-cost bookkeeping flows through to net payoff ----

def test_net_payoff_subtracts_cost():
    reports = [_r("A", 1, 0.9, cost=0.3), _r("B", 1, 0.9, cost=0.3)]
    res = team_contract_decision(
        "Q", reports,
        contract="holmstrom_forcing", bonus=1.0, ground_truth=1,
    )
    for p in res.payoffs:
        assert p.gross == 1.0
        assert p.cost == 0.3
        assert p.net == pytest.approx(0.7)
