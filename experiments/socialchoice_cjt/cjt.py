"""Condorcet Jury Theorem (CJT) aggregation for binary LLM-debater votes.

Theory references
-----------------
- Condorcet, M. de (1785). *Essai sur l'application de l'analyse a la
  probabilite des decisions rendues a la pluralite des voix.* Paris.
- Ladha, K. K. (1992). "The Condorcet Jury Theorem, Free Speech, and
  Correlated Votes." *American Journal of Political Science* 36(3): 617-634.
  (Extension of CJT to correlated voters; gives the variance-of-vote-share
  bound used in `correlated_majority_correct_prob` below.)
- Boland, P. J. (1989). "Majority Systems and the Condorcet Jury Theorem."
  *The Statistician* 38(3): 181-189.

Classical (independent) CJT: if N voters each independently get the binary
truth right with probability p > 1/2, then majority-vote correctness
P_N -> 1 monotonically as N -> inf.

Correlated CJT (Ladha 1992, Thm 1): with pairwise vote correlation rho,
the variance of the sample mean of votes is

    Var(V_bar) = p(1-p)/N * (1 + (N-1)*rho)

so as N -> inf the variance floor is p(1-p)*rho, and majority correctness
plateaus *below* 1 whenever rho > 0. This is the failure mode we expose.

Public API
----------
- `aggregate(question, votes)`: majority-rule decision + diagnostics
  (winner, margin, vote share, would-be Condorcet pairwise winner trivially
  equals the majority answer in the binary case).
- `independent_majority_correct_prob(p, N)`: closed-form for the canonical
  CJT curve.
- `correlated_majority_correct_prob(p, N, rho, n_mc)`: Monte-Carlo
  estimate using a Beta-Bernoulli (exchangeable) generator that matches
  the Ladha (1992) variance formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CJTResult:
    """Diagnostics from a single CJT aggregation."""

    decision: bool                 # majority answer
    yes_votes: int
    no_votes: int
    n: int
    margin: int                    # yes - no
    yes_share: float
    tie: bool                      # True iff yes_votes == no_votes


def aggregate(question: str, votes: Sequence[bool]) -> CJTResult:
    """Aggregate a sequence of binary votes by majority rule (CJT).

    Args:
        question: The question being voted on (kept for traceability,
            not used in math). The CJT is *content-blind* -- this is one
            of its honest weaknesses vs BTS.
        votes: Iterable of booleans, one per debater. True = "the
            statement is true".

    Returns:
        `CJTResult`. Ties resolve to ``decision=False`` and ``tie=True``
        so that downstream code can detect and re-poll.
    """
    arr = np.asarray(list(votes), dtype=bool)
    n = int(arr.size)
    if n == 0:
        raise ValueError("aggregate() requires at least one vote.")
    yes = int(arr.sum())
    no = n - yes
    tie = yes == no
    decision = yes > no  # strict majority; tie -> False
    return CJTResult(
        decision=decision,
        yes_votes=yes,
        no_votes=no,
        n=n,
        margin=yes - no,
        yes_share=yes / n,
        tie=tie,
    )


def independent_majority_correct_prob(p: float, n: int) -> float:
    """P(majority correct) for n IID voters each correct w.p. p.

    Closed form: sum_{k=ceil((n+1)/2)}^{n} C(n,k) p^k (1-p)^(n-k).
    For even n we count strict majority only (matches `aggregate`'s
    tie-rule, which counts ties as failures).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")
    threshold = n // 2 + 1  # strict majority
    return float(
        sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(threshold, n + 1))
    )


def correlated_majority_correct_prob(
    p: float,
    n: int,
    rho: float,
    n_mc: int = 20_000,
    seed: int | None = 0,
) -> float:
    """Monte-Carlo P(majority correct) under exchangeable correlated votes.

    Generates each trial's votes as Bernoulli draws with a *shared* latent
    success rate Q ~ Beta(alpha, beta), where alpha, beta are picked so
    that E[Q] = p and the implied pairwise correlation is rho. This is
    the standard Beta-Bernoulli (Polya) exchangeable model whose pairwise
    correlation equals 1 / (alpha + beta + 1); see e.g. Ladha (1992) eq. 4.

    Args:
        p: Marginal per-voter correctness.
        n: Jury size.
        rho: Pairwise correlation in [0, 1). rho=0 reduces (in expectation)
            to the IID case.
        n_mc: Monte-Carlo trials.
        seed: RNG seed for reproducibility.

    Returns:
        Estimated P(strict majority correct).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0,1]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0,1)")
    if n < 1 or n_mc < 1:
        raise ValueError("n and n_mc must be >= 1")
    rng = np.random.default_rng(seed)
    if rho == 0.0:
        # Pure IID; use closed form for stability.
        return independent_majority_correct_prob(p, n)
    # Beta-Bernoulli moment match: corr = 1/(a+b+1) -> a+b = 1/rho - 1.
    s = 1.0 / rho - 1.0
    alpha = p * s
    beta = (1.0 - p) * s
    if alpha <= 0 or beta <= 0:
        raise ValueError(
            "Beta parameters non-positive; pick p in (0,1) and rho < 1."
        )
    q = rng.beta(alpha, beta, size=n_mc)              # latent shared rate
    votes = rng.random((n_mc, n)) < q[:, None]        # (n_mc, n) booleans
    yes = votes.sum(axis=1)
    threshold = n // 2 + 1
    return float(np.mean(yes >= threshold))
