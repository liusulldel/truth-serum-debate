"""Holmstrom (1982) team moral hazard with budget-breaker, applied to LLM debate.

Reference
---------
Holmstrom, B. (1982). "Moral Hazard in Teams."
    The Bell Journal of Economics, 13(2): 324-340. JSTOR 3003457.
    https://doi.org/10.2307/3003457

Theorem (informal)
------------------
When n risk-neutral agents jointly produce output x = f(e_1,...,e_n) + noise
and share x via budget-balancing rules sum_i s_i(x) = x, no Nash equilibrium
implements first-best effort. PROOF: each agent's marginal incentive is the
derivative of their share, but these must sum to 1; under symmetry each gets
1/n of marginal output, so under-invests by factor n. FIX: drop budget
balance --- introduce a "principal" (budget-breaker) who is residual claimant
and pays each agent the FULL marginal output when group hits target, zero
otherwise (a forcing contract). Holmstrom shows this restores first-best.

Transfer to multi-agent LLM debate
----------------------------------
Two debaters jointly produce a "team verdict". Truthful, calibrated reporting
is costly effort (the debater must reason hard, override priors). Under BTS
the two debaters split a fixed-sum information score: A's gain is B's loss,
which is the *opposite* of team production --- it is zero-sum *competition*,
not joint output. Holmstrom-1982 says joint-output settings need a budget
breaker. Here we model the JUDGE as the budget-breaker / principal:
- Group output g = correctness of the aggregated verdict (1 if matches truth).
- Forcing contract: each agent gets bonus B if g = 1 AND their report agreed
  with the verdict; 0 otherwise. The judge funds the bonus from outside the
  team (residual claimant of being-correct), so payments do NOT have to sum
  to a fixed pot.

Property to test
----------------
Under budget-balanced equal-split (BTS-style), if both debaters have a small
private cost c of "thinking hard", neither has incentive to exert effort
because the marginal payoff is c/2 < c. Under Holmstrom forcing contract
with bonus B > 2c, both exert effort --> verdict accuracy strictly higher.

This module is pure-Python, no LLM call. Mock debater outputs in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Verdict = Literal[0, 1]  # 0=false, 1=true


@dataclass
class DebaterReport:
    """A single mock debater output."""
    name: str
    answer: Verdict
    confidence: float  # in [0,1], debater's stated p that answer is correct
    effort_cost: float = 0.0  # private cost paid (bookkeeping, not used in agg)


@dataclass
class TeamPayoff:
    """Per-agent net payoff under a contract (gross bonus minus effort cost)."""
    name: str
    gross: float
    cost: float
    net: float


@dataclass
class TeamResult:
    verdict: Verdict
    confidence: float
    correct: bool | None  # None if ground_truth not supplied
    payoffs: list[TeamPayoff]
    contract: str
    budget_balanced: bool


def _aggregate(reports: Sequence[DebaterReport]) -> tuple[Verdict, float]:
    """Confidence-weighted majority vote --> (verdict, aggregate-confidence)."""
    s = 0.0
    w = 0.0
    for r in reports:
        s += (1 if r.answer == 1 else -1) * r.confidence
        w += r.confidence
    verdict: Verdict = 1 if s >= 0 else 0
    conf = abs(s) / w if w > 0 else 0.0
    return verdict, conf


def team_contract_decision(
    question: str,
    reports: Sequence[DebaterReport],
    *,
    contract: Literal["budget_balanced", "holmstrom_forcing"] = "holmstrom_forcing",
    bonus: float = 1.0,
    ground_truth: Verdict | None = None,
    target_confidence: float = 0.5,
) -> TeamResult:
    """Aggregate debaters and pay each per the chosen contract.

    Args:
        question: Question text (carried for traceability; not used in math).
        reports: List of DebaterReport objects (>=1).
        contract:
          - "budget_balanced": fixed pot of size `bonus` split equally among
            agents whose report matches the verdict. This is the Holmstrom
            *negative* result: budget balance breaks incentives.
          - "holmstrom_forcing": each agreeing agent receives `bonus` IFF
            verdict is judged correct (group output = 1) AND aggregate
            confidence exceeds `target_confidence`. Judge is the budget-
            breaker / residual claimant.
        bonus: Per-agent bonus (forcing) or pot size (budget_balanced).
        ground_truth: 0/1 if known (eval harness); None means judge uses its
            own confidence threshold instead (works without ground truth).
        target_confidence: Threshold for the forcing contract when no ground
            truth is supplied. Default 0.5 = mere majority.

    Returns:
        TeamResult with verdict, payoffs, and bookkeeping.
    """
    if not reports:
        raise ValueError("Need at least one debater report.")
    if contract not in {"budget_balanced", "holmstrom_forcing"}:
        raise ValueError(f"Unknown contract: {contract}")

    verdict, conf = _aggregate(reports)
    correct: bool | None
    if ground_truth is None:
        # Use confidence as a proxy for "judge believes verdict is correct".
        group_output = 1 if conf >= target_confidence else 0
        correct = None
    else:
        correct = (verdict == ground_truth)
        group_output = 1 if correct else 0

    agreeing = [r for r in reports if r.answer == verdict]
    payoffs: list[TeamPayoff] = []

    if contract == "budget_balanced":
        # Fixed pot `bonus` split equally among agreeing agents, regardless
        # of correctness --- this is the pathological case Holmstrom rules
        # out. It's budget-balanced (sum of payouts == bonus, exactly).
        share = (bonus / len(agreeing)) if agreeing else 0.0
        for r in reports:
            gross = share if r in agreeing else 0.0
            payoffs.append(TeamPayoff(r.name, gross, r.effort_cost, gross - r.effort_cost))
    else:
        # Holmstrom forcing contract: full bonus to each agreeing agent IFF
        # group_output == 1. Sum of payouts can exceed `bonus` (judge funds
        # it from outside the team) --- explicitly NOT budget balanced.
        for r in reports:
            paid = (r in agreeing) and (group_output == 1)
            gross = bonus if paid else 0.0
            payoffs.append(TeamPayoff(r.name, gross, r.effort_cost, gross - r.effort_cost))

    return TeamResult(
        verdict=verdict,
        confidence=conf,
        correct=correct,
        payoffs=payoffs,
        contract=contract,
        budget_balanced=(contract == "budget_balanced"),
    )


def equilibrium_effort(
    n: int,
    bonus: float,
    cost_of_effort: float,
    contract: Literal["budget_balanced", "holmstrom_forcing"],
) -> bool:
    """Symmetric Nash effort decision under the two contracts.

    Returns True iff a representative agent strictly prefers to exert effort
    (pay `cost_of_effort` for marginal +1 chance of being on the winning team).

    Holmstrom (1982) shows:
      budget_balanced -> marginal payoff = bonus/n  (free-ride)
      holmstrom_forcing -> marginal payoff = bonus  (first-best)
    """
    if contract == "budget_balanced":
        marginal = bonus / n
    else:
        marginal = bonus
    return marginal > cost_of_effort
