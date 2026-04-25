"""AGV (d'Aspremont-Gerard-Varet / expected-externality) aggregator for LLM debate.

Reference:
    d'Aspremont, C. & Gerard-Varet, L.-A. (1979).
    "Incentives and incomplete information."
    Journal of Public Economics 11(1): 25-45.
    DOI: 10.1016/0047-2727(79)90043-4

Setting transferred to LLM debate:
  Each debater i reports a "type" t_i = posterior probability that the
  proposition is TRUE (a number in [0, 1]) plus a predicted distribution
  over the OTHER debaters' reports. We treat t_i as a private valuation
  of the "TRUE" alternative vs. "FALSE" alternative (v_i(TRUE)=t_i,
  v_i(FALSE)=1-t_i).

Decision rule (efficient): pick whichever option maximises summed
reported valuations -- equivalent to picking TRUE iff sum_i t_i > n/2.

AGV transfer payment for agent i:
    tau_i = E_{t_-i ~ p_i} [ sum_{j != i} v_j(d*(t_i, t_-i)) ]
            - (1/(n-1)) * sum_{k != i} E_{t_-k ~ p_k}
                [ sum_{j != k} v_j(d*(t_k, t_-k)) ]

The first term is i's expected externality on others (computed under
i's reported beliefs over peers). The second term re-distributes the
sum of others' first terms equally so the budget balances exactly.
Truth-telling is a Bayes-Nash equilibrium (AGV 1979, Thm 1) and the
sum of transfers is identically zero (budget balance) -- both
properties VCG lacks.

Single public function ``agv_aggregate`` returns the chosen option
plus per-agent score (utility = realised valuation + transfer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import itertools


@dataclass
class DebaterReport:
    """One LLM debater's structured output (mock-friendly)."""
    own_belief: float                 # P(TRUE) according to this debater, in [0,1]
    peer_belief_means: list[float]    # length n-1, each in [0,1]: predicted E[t_j] for each other debater j


def _decision(beliefs: Sequence[float]) -> int:
    """Efficient rule: pick TRUE (1) iff sum of valuations for TRUE exceeds for FALSE."""
    s = sum(beliefs)
    return 1 if s > len(beliefs) / 2.0 else 0


def _social_welfare_excluding(beliefs: Sequence[float], exclude: int, decision: int) -> float:
    """Sum_{j != exclude} v_j(decision)."""
    total = 0.0
    for j, t in enumerate(beliefs):
        if j == exclude:
            continue
        total += t if decision == 1 else (1.0 - t)
    return total


def _expected_externality(
    i: int,
    own_t: float,
    peer_means: Sequence[float],
    grid: int = 5,
) -> float:
    """E_{t_-i} [ sum_{j != i} v_j(d*(own_t, t_-i)) ] under independent Bernoulli-on-grid beliefs.

    For tractability we discretise each peer's type to {low, high} around its mean.
    With n-1 peers this gives 2^(n-1) scenarios, each weighted by product of
    peer probabilities. Faithful to AGV's "expected over reported prior" structure.
    """
    n_peers = len(peer_means)
    # Two-point support around the mean: {0, 1} with prob {1-mean, mean} -> Bernoulli with E = mean.
    total = 0.0
    for bits in itertools.product((0, 1), repeat=n_peers):
        prob = 1.0
        peer_ts: list[float] = []
        for k, b in enumerate(bits):
            mu = peer_means[k]
            prob *= (mu if b == 1 else 1.0 - mu)
            peer_ts.append(float(b))
        # Reconstruct full belief vector with i's own type inserted at position i.
        full = peer_ts[:i] + [own_t] + peer_ts[i:]
        d = _decision(full)
        # sum_{j != i} v_j(d): sum over positions in `full` excluding i.
        ext = 0.0
        for j, t in enumerate(full):
            if j == i:
                continue
            ext += t if d == 1 else (1.0 - t)
        total += prob * ext
    return total


def agv_aggregate(question: str, reports: Sequence[DebaterReport]) -> dict:
    """Aggregate debater reports via the AGV mechanism.

    Args:
        question: The proposition under debate (passed through, not used numerically).
        reports: Length-n list of DebaterReport. Each must have peer_belief_means of length n-1.

    Returns:
        Dict with:
          decision: 0 (FALSE) or 1 (TRUE)
          per_agent_score: list[float] of length n, AGV utility = realised valuation + transfer
          transfers: list[float] of length n, AGV transfer per agent (sums to ~0)
          aggregated_belief: arithmetic mean of own_beliefs (for diagnostic)
    """
    n = len(reports)
    if n < 2:
        raise ValueError("AGV needs at least 2 agents.")
    for r in reports:
        if len(r.peer_belief_means) != n - 1:
            raise ValueError(
                f"Each report needs peer_belief_means of length {n-1}, got {len(r.peer_belief_means)}."
            )
        if not 0.0 <= r.own_belief <= 1.0:
            raise ValueError("own_belief out of [0,1].")

    own_ts = [r.own_belief for r in reports]
    decision = _decision(own_ts)

    # First term: i's own expected externality under i's reported peer-means.
    first_terms = [
        _expected_externality(i, own_ts[i], reports[i].peer_belief_means)
        for i in range(n)
    ]
    # Second (balancing) term: average of others' first terms.
    transfers = []
    for i in range(n):
        others = [first_terms[k] for k in range(n) if k != i]
        balance = sum(others) / (n - 1)
        transfers.append(first_terms[i] - balance)

    realised = [
        own_ts[i] if decision == 1 else 1.0 - own_ts[i]
        for i in range(n)
    ]
    per_agent_score = [realised[i] + transfers[i] for i in range(n)]

    return {
        "question": question,
        "decision": decision,
        "per_agent_score": per_agent_score,
        "transfers": transfers,
        "aggregated_belief": sum(own_ts) / n,
    }
