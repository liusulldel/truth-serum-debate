"""Property-based tests for core mechanisms.

Verifies *mathematical* invariants of:
  - src/bts.py::bts_scores                 (Prelec 2004 zero-sum identity)
  - experiments/ambig_alpha_meu/alpha_meu  (alpha-MEU bracketing + ambiguity)
  - experiments/orgecon_garicano/garicano  (delegation extremes + cost monotonicity)
  - experiments/baselines/majority_vote    (Condorcet jury monotonicity in N)

Run with:  pytest experiments/test_properties.py -v
"""
from __future__ import annotations

import random

import pytest
from hypothesis import given, settings, strategies as st

from src.bts import bts_scores
from experiments.ambig_alpha_meu.alpha_meu import alpha_meu_aggregate
from experiments.orgecon_garicano.garicano import (
    DebaterReport,
    garicano_throughput,
)
from experiments.baselines import DebaterOutput
from experiments.baselines.majority_vote import aggregate as majority_aggregate


# ---------- Reusable strategies --------------------------------------------

# A normalised probability vector over m options.
def prob_vec(m: int):
    return st.lists(
        st.floats(min_value=0.01, max_value=1.0), min_size=m, max_size=m
    ).map(lambda xs: [x / sum(xs) for x in xs])


# A list of N probability vectors of length m.
def prob_matrix(n_min: int = 1, n_max: int = 6, m_min: int = 2, m_max: int = 4):
    return st.integers(min_value=m_min, max_value=m_max).flatmap(
        lambda m: st.lists(prob_vec(m), min_size=n_min, max_size=n_max)
    )


# =========================================================================
# 1. BTS  (Prelec 2004)
# =========================================================================

@given(
    n_options=st.integers(min_value=2, max_value=4),
    bad_offset=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=200, deadline=None)
def test_bts_rejects_out_of_range_answers(n_options, bad_offset):
    """(1a) input validation: any answer >= n_options must raise."""
    answers = [0, n_options - 1 + bad_offset]   # second one is out of range
    preds = [[1.0 / n_options] * n_options] * 2
    with pytest.raises(ValueError):
        bts_scores(answers, preds, n_options=n_options)


@given(
    n_options=st.integers(min_value=2, max_value=4),
    n=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200, deadline=None)
def test_bts_zero_sum_identity(n_options, n, seed):
    """(1b) Prelec's identity: sum_r BTS(r) == 0 exactly.

    Derivation:
        sum_r info(r) = N * KL(x_bar || y_bar)
        sum_r pred(r) = - N * KL(x_bar || y_bar)
        => sum is identically zero, regardless of distributions.

    This is the *strong* form of the translation/scale invariance the user
    asked about: it says BTS is a pure redistribution rule, so any uniform
    additive shift to predictions must wash out in aggregate.
    """
    rng = random.Random(seed)
    answers = [rng.randrange(n_options) for _ in range(n)]
    preds = []
    for _ in range(n):
        v = [rng.uniform(0.05, 1.0) for _ in range(n_options)]
        s = sum(v)
        preds.append([x / s for x in v])
    scores = bts_scores(answers, preds, n_options=n_options)
    # Float arithmetic + clipping accumulates ~1e-7 per respondent; loosen.
    assert sum(scores) == pytest.approx(0.0, abs=1e-6)


@given(
    n_options=st.integers(min_value=2, max_value=4),
    n=st.integers(min_value=2, max_value=6),
    scale=st.floats(min_value=0.1, max_value=10.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200, deadline=None)
def test_bts_renormalisation_invariance(n_options, n, scale, seed):
    """(1c) BTS is invariant to per-row positive rescaling of predictions.

    Multiplying every predicted vector by the same positive scalar is a
    constant additive shift in log-space (== Prelec's "log-shift") and
    must leave every score unchanged once the row-sum normalisation
    inside bts_scores absorbs it.
    """
    rng = random.Random(seed)
    answers = [rng.randrange(n_options) for _ in range(n)]
    preds = []
    for _ in range(n):
        v = [rng.uniform(0.05, 1.0) for _ in range(n_options)]
        s = sum(v)
        preds.append([x / s for x in v])
    base = bts_scores(answers, preds, n_options=n_options)
    scaled_preds = [[x * scale for x in p] for p in preds]
    shifted = bts_scores(answers, scaled_preds, n_options=n_options)
    for a, b in zip(base, shifted):
        assert a == pytest.approx(b, abs=1e-9)


# =========================================================================
# 2. alpha-MEU  (Ghirardato-Maccheroni-Marinacci 2004)
# =========================================================================

@given(
    alpha=st.floats(min_value=0.0, max_value=1.0),
    rows=prob_matrix(n_min=1, n_max=6, m_min=2, m_max=4),
)
@settings(max_examples=200, deadline=None)
def test_alpha_meu_p_alpha_in_bracket(alpha, rows):
    """(2a) For every alpha in [0,1], P_alpha lies in [p_min, p_max]."""
    out = alpha_meu_aggregate("q", rows, alpha=alpha, tau=1.0, true_index=0)
    assert out.p_min - 1e-12 <= out.p_alpha <= out.p_max + 1e-12


@given(rows=prob_matrix(n_min=1, n_max=6, m_min=2, m_max=4))
@settings(max_examples=200, deadline=None)
def test_alpha_meu_endpoints_recover_min_and_max(rows):
    """(2b) alpha=1 => P_alpha == p_min ; alpha=0 => P_alpha == p_max."""
    maxmin = alpha_meu_aggregate("q", rows, alpha=1.0, tau=1.0, true_index=0)
    maxmax = alpha_meu_aggregate("q", rows, alpha=0.0, tau=1.0, true_index=0)
    assert maxmin.p_alpha == pytest.approx(maxmin.p_min)
    assert maxmax.p_alpha == pytest.approx(maxmax.p_max)


@given(rows=prob_matrix(n_min=1, n_max=6, m_min=2, m_max=4))
@settings(max_examples=200, deadline=None)
def test_ambiguity_index_nonneg_and_zero_iff_unanimous(rows):
    """(2c) A >= 0; A == 0 iff every debater reports the same q on true_index."""
    out = alpha_meu_aggregate("q", rows, alpha=0.5, tau=1.0, true_index=0)
    assert out.ambiguity_index >= -1e-12
    # Recompute q after the same row-normalisation alpha_meu_aggregate uses,
    # so the equivalence holds even when input rows aren't perfectly normalised.
    qs = [r[0] / sum(r) for r in rows]
    unanimous = max(qs) - min(qs) < 1e-12
    assert (out.ambiguity_index < 1e-12) == unanimous


# =========================================================================
# 3. Garicano (2000)
# =========================================================================

def _truth_judge(reports):
    """Judge that always returns the closure-injected ground truth."""
    return _truth_judge.gt   # type: ignore[attr-defined]


@given(
    n_problems=st.integers(min_value=1, max_value=8),
    n_debaters=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200, deadline=None)
def test_garicano_tau_zero_never_escalates(n_problems, n_debaters, seed):
    """(3a) tau == 0 => worker layer always decides; escalation_rate == 0."""
    rng = random.Random(seed)
    problems = []
    for _ in range(n_problems):
        gt = rng.randrange(2)
        reports = [
            DebaterReport(answer=rng.randrange(2), confidence=rng.uniform(0.0, 1.0))
            for _ in range(n_debaters)
        ]
        problems.append((reports, gt))
    _truth_judge.gt = 0  # unused
    out = garicano_throughput(problems, _truth_judge, worker_threshold=0.0,
                              referral_cost=0.1)
    assert out["escalation_rate"] == 0.0


@given(
    n_problems=st.integers(min_value=1, max_value=8),
    n_debaters=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200, deadline=None)
def test_garicano_tau_above_max_conf_always_escalates(n_problems, n_debaters, seed):
    """(3b) tau strictly above every reported confidence => always escalate.

    Strictly stronger than 'tau==1': we sample confidences in [0, 0.9] and
    set tau to 0.95 so no debater can clear it under any draw.
    """
    rng = random.Random(seed)
    problems = []
    for _ in range(n_problems):
        gt = rng.randrange(2)
        reports = [
            DebaterReport(answer=rng.randrange(2), confidence=rng.uniform(0.0, 0.9))
            for _ in range(n_debaters)
        ]
        problems.append((reports, gt))
    _truth_judge.gt = 1
    # Judge always says 1 -- accuracy depends on gt but escalation must be 100%.
    out = garicano_throughput(problems, _truth_judge, worker_threshold=0.95,
                              referral_cost=0.1)
    assert out["escalation_rate"] == 1.0


@given(
    n_problems=st.integers(min_value=2, max_value=8),
    n_debaters=st.integers(min_value=1, max_value=4),
    tau=st.floats(min_value=0.0, max_value=1.0),
    h_lo=st.floats(min_value=0.0, max_value=0.4),
    h_hi=st.floats(min_value=0.5, max_value=2.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200, deadline=None)
def test_garicano_net_utility_monotone_decreasing_in_h(
    n_problems, n_debaters, tau, h_lo, h_hi, seed
):
    """(3c) For a fixed strategy, net_utility(h) is non-increasing in h.

    accuracy and escalation_rate depend only on (reports, judge, tau), so
    net_utility = accuracy - h * escalation_rate is linear-decreasing in h.
    """
    rng = random.Random(seed)
    problems = []
    for _ in range(n_problems):
        gt = rng.randrange(2)
        reports = [
            DebaterReport(answer=rng.randrange(2), confidence=rng.uniform(0.0, 1.0))
            for _ in range(n_debaters)
        ]
        problems.append((reports, gt))
    _truth_judge.gt = 1
    lo = garicano_throughput(problems, _truth_judge, tau, referral_cost=h_lo)
    hi = garicano_throughput(problems, _truth_judge, tau, referral_cost=h_hi)
    # Same strategy => same accuracy and escalation_rate.
    assert lo["accuracy"] == hi["accuracy"]
    assert lo["escalation_rate"] == hi["escalation_rate"]
    assert lo["net_utility"] >= hi["net_utility"] - 1e-12


# =========================================================================
# 4. Majority vote -- Condorcet jury monotonicity
# =========================================================================

@given(
    p=st.floats(min_value=0.55, max_value=0.95),
    base_seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
def test_majority_vote_accuracy_monotone_in_n(p, base_seed):
    """(4) Condorcet jury: with i.i.d. voters at p>0.5, empirical accuracy
    of majority rule is non-decreasing in N (within Monte-Carlo tolerance).

    Strategy: at fixed p, simulate T=2000 trials at each N in {3,5,9,21}
    (odd to avoid ties), check accuracy(3) <= accuracy(21) up to 3-sigma.
    Pairwise monotonicity at small N can fail by sampling noise; the
    Condorcet theorem is asymptotic, so we test the asymptotic endpoint.
    """
    T = 2000
    Ns = [3, 5, 9, 21]
    accs = []
    rng = random.Random(base_seed)
    for N in Ns:
        correct = 0
        for _ in range(T):
            votes = [1 if rng.random() < p else 0 for _ in range(N)]
            outs = [DebaterOutput(answer=v, p_true=float(v)) for v in votes]
            dec = majority_aggregate("q", outs)
            if dec.answer == 1:        # ground truth is always 1
                correct += 1
        accs.append(correct / T)
    # 3-sigma slack on each accuracy estimate (Bernoulli SE = sqrt(p(1-p)/T)).
    se = (0.25 / T) ** 0.5
    assert accs[-1] >= accs[0] - 3 * se, f"accuracy(N=21)={accs[-1]} not >= accuracy(N=3)={accs[0]}"
    # Final accuracy should also exceed the per-voter p (jury improves on individual).
    assert accs[-1] >= p - 3 * se
