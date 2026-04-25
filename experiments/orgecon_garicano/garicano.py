"""Garicano (2000) knowledge-based hierarchy aggregator for LLM debate.

Reference:
    Garicano, L. (2000). "Hierarchies and the Organization of Knowledge
    in Production." Journal of Political Economy 108(5): 874-904.
    DOI: 10.1086/317671

Original setting (paraphrased):
  Workers (production layer) face a stream of "problems" of varying
  difficulty. A worker who knows the answer solves it locally at zero
  marginal cost. A worker who does not knows the answer "passes up" the
  problem to a manager (referral cost h per kick-up). The manager has
  broader knowledge but scarce attention -- referrals consume capacity.
  The optimal organisation maximises throughput by matching the
  hardest-known-knowledge layer to each problem. Key comparative static
  (Garicano 2000, Prop. 2-3): the *referral / communication cost h*
  determines the optimal pyramid depth and the manager's span of control.

Transfer to LLM debate-judge:
  - "Workers" = debaters. Each emits an answer plus a self-reported
    confidence in [0,1]. If max debater confidence >= worker_threshold,
    the worker layer "solves" the problem and the judge is NOT consulted.
    This saves referral_cost h (judge tokens / latency / attention).
  - "Manager" = judge. Only consulted when no debater is confident.
    The judge has higher accuracy but each call costs h.
  - "Delegation rule" = the worker_threshold tau in [0,1]. Low tau
    means workers handle everything (flat org); high tau means
    everything escalates (centralised org). Garicano's prediction:
    optimal tau* is interior and strictly decreasing in h.

This module exposes ``garicano_decide`` (single decision) and
``garicano_throughput`` (expected net-utility = accuracy minus referral
cost) so we can sweep tau and reproduce the interior-optimum prediction.

This is a deliberately minimal, mock-friendly implementation: no live
API calls. Debater outputs and the judge are passed in as plain
callables / dataclasses so unit tests can drive every branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class DebaterReport:
    """One debater's output: an option index and a self-reported confidence."""
    answer: int
    confidence: float  # in [0, 1]


@dataclass
class Decision:
    """Aggregator output."""
    answer: int
    used_judge: bool          # True iff problem was escalated
    realised_cost: float      # 0 if worker-solved, else h
    chosen_confidence: float  # confidence of the layer that decided


def garicano_decide(
    reports: Sequence[DebaterReport],
    judge: Callable[[Sequence[DebaterReport]], int],
    worker_threshold: float,
    referral_cost: float = 0.1,
) -> Decision:
    """One Garicano-style decision.

    Args:
        reports: debater outputs.
        judge: callable mapping reports -> chosen option (only invoked
            when no debater clears the threshold; this is the Garicano
            "kick the problem upstairs" event).
        worker_threshold: tau in [0,1]. Worker layer keeps the decision
            iff at least one debater reports confidence >= tau.
        referral_cost: h, the cost (judge tokens / latency) charged when
            the problem escalates. Must be >= 0.

    Returns:
        Decision dataclass. ``used_judge`` lets the caller verify the
        delegation event matches the threshold rule.
    """
    if not reports:
        raise ValueError("Need at least one debater report.")
    if not 0.0 <= worker_threshold <= 1.0:
        raise ValueError("worker_threshold must be in [0,1].")
    if referral_cost < 0.0:
        raise ValueError("referral_cost must be non-negative.")

    # Worker layer: pick the *most confident* debater's answer if any
    # clears the threshold. (Garicano: workers self-select on knowledge.)
    best_idx = max(range(len(reports)), key=lambda i: reports[i].confidence)
    best = reports[best_idx]
    if best.confidence >= worker_threshold:
        return Decision(
            answer=best.answer,
            used_judge=False,
            realised_cost=0.0,
            chosen_confidence=best.confidence,
        )

    # Otherwise escalate.
    judged = judge(reports)
    return Decision(
        answer=judged,
        used_judge=True,
        realised_cost=referral_cost,
        chosen_confidence=1.0,  # judge resolves uncertainty by assumption
    )


def garicano_throughput(
    problems: Sequence[tuple[Sequence[DebaterReport], int]],
    judge: Callable[[Sequence[DebaterReport]], int],
    worker_threshold: float,
    referral_cost: float = 0.1,
) -> dict:
    """Expected net throughput across a batch of problems.

    Args:
        problems: list of (reports, ground_truth_answer) pairs.
        judge: judge callable (assumed strictly more accurate than any
            single debater; this is the Garicano "managers know more"
            primitive).
        worker_threshold: tau as above.
        referral_cost: h as above.

    Returns:
        Dict with ``accuracy`` (fraction correct), ``escalation_rate``
        (fraction sent up), ``net_utility`` (= accuracy - h * escalation),
        and ``decisions`` (the per-problem Decision objects).
    """
    decisions = [
        garicano_decide(rep, judge, worker_threshold, referral_cost)
        for rep, _ in problems
    ]
    correct = sum(1 for d, (_, gt) in zip(decisions, problems) if d.answer == gt)
    escalated = sum(1 for d in decisions if d.used_judge)
    n = len(problems)
    accuracy = correct / n
    esc_rate = escalated / n
    return {
        "accuracy": accuracy,
        "escalation_rate": esc_rate,
        "net_utility": accuracy - referral_cost * esc_rate,
        "decisions": decisions,
    }
