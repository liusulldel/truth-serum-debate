"""Tests for `cjt.py`.

Five mock cases; the third (`test_correlation_caps_convergence`) is the
failure-mode case the global note asked for: it shows that with shared
priors / shared training data (rho > 0) the CJT n -> inf guarantee
breaks down, capping accuracy strictly below 1.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from cjt import (  # noqa: E402
    aggregate,
    correlated_majority_correct_prob,
    independent_majority_correct_prob,
)


# ---------- Case 1: basic majority correctness on a 5-debater poll ----------
def test_aggregate_basic_majority():
    votes = [True, True, False, True, False]
    res = aggregate("Water boils at 100C at sea level.", votes)
    assert res.decision is True
    assert res.yes_votes == 3 and res.no_votes == 2
    assert res.margin == 1
    assert not res.tie
    assert res.yes_share == pytest.approx(0.6)


# ---------- Case 2: ties resolve to False and are flagged ----------
def test_aggregate_tie_handling():
    res = aggregate("Coin flip.", [True, False, True, False])
    assert res.tie is True
    assert res.decision is False
    assert res.margin == 0


# ---------- Case 3: failure-mode -- correlation breaks n->inf convergence
# This is the "shared training data" critique. With rho=0 accuracy -> 1;
# with rho=0.3 it plateaus well below 1 even at N=101. ----------
def test_correlation_caps_convergence():
    p = 0.65
    p_iid_101 = independent_majority_correct_prob(p, 101)
    p_corr_101 = correlated_majority_correct_prob(p, 101, rho=0.3, n_mc=10_000)
    # IID accuracy is essentially 1 at N=101, p=0.65.
    assert p_iid_101 > 0.999
    # Correlated accuracy gets stuck far below 1; Ladha (1992) variance
    # floor is p(1-p)*rho = 0.0683, so the law of large numbers can't
    # push the sample mean past 0.5 reliably.
    assert p_corr_101 < 0.95
    # And it does NOT improve much over a smaller jury -- the hallmark
    # of correlated CJT failure.
    p_corr_21 = correlated_majority_correct_prob(p, 21, rho=0.3, n_mc=10_000)
    assert abs(p_corr_101 - p_corr_21) < 0.05


# ---------- Case 4: classical CJT monotonicity in n (independent case) ----------
@pytest.mark.parametrize("p", [0.55, 0.6, 0.7])
def test_independent_cjt_monotone(p: float):
    ns = (3, 5, 11, 31, 101, 401)
    probs = [independent_majority_correct_prob(p, n) for n in ns]
    # Strictly increasing in n for p > 0.5 across odd jury sizes
    # (Boland 1989, Thm 1).
    for a, b in zip(probs, probs[1:]):
        assert b > a - 1e-12  # allow tiny FP slack
    # Convergence to 1: at N=401 even the slow p=0.55 curve clears 0.95.
    assert probs[-1] > 0.95


# ---------- Case 5: anti-CJT regime -- if p < 0.5, more voters HURTS ----------
def test_independent_cjt_reverse_for_bad_jurors():
    # Symmetric case: p=0.4 jurors should converge to wrong answer.
    p = 0.4
    probs = [independent_majority_correct_prob(p, n) for n in (3, 11, 51, 201)]
    for a, b in zip(probs, probs[1:]):
        assert b < a + 1e-12
    assert probs[-1] < 0.1
