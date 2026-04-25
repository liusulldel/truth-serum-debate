"""Tests for alpha-MEU debate aggregation.

Five mock cases including the headline disagreement case where BTS averages
to 0.5 but alpha-MEU correctly abstains.
"""
from __future__ import annotations

import pytest

from experiments.ambig_alpha_meu.alpha_meu import (
    alpha_meu_aggregate,
    bts_style_mean_decision,
)


def test_consensus_true_decides_true():
    """All debaters confident in TRUE -> low ambiguity, decide TRUE."""
    out = alpha_meu_aggregate(
        "Is water wet?",
        [[0.05, 0.95], [0.10, 0.90], [0.08, 0.92]],
        alpha=1.0,
        tau=0.4,
    )
    assert out.decision == "TRUE"
    assert out.ambiguity_index < 0.1
    assert out.p_alpha == pytest.approx(0.90, abs=1e-9)  # min under maxmin


def test_consensus_false_decides_false():
    """All debaters confident in FALSE -> low ambiguity, decide FALSE."""
    out = alpha_meu_aggregate(
        "Is 2+2=5?",
        [[0.92, 0.08], [0.95, 0.05], [0.88, 0.12]],
        alpha=1.0,
        tau=0.4,
    )
    assert out.decision == "FALSE"
    assert out.ambiguity_index < 0.1
    assert out.p_alpha < 0.5


def test_strong_disagreement_triggers_abstain():
    """Headline case: A=0.9, B=0.1. BTS averages to 0.5 (decision-flip
    sensitive); alpha-MEU sees ambiguity 0.8 > tau and abstains.
    """
    dists = [[0.1, 0.9], [0.9, 0.1]]
    out = alpha_meu_aggregate("Disputed claim", dists, alpha=1.0, tau=0.4)
    assert out.decision == "ABSTAIN"
    assert out.ambiguity_index == pytest.approx(0.8, abs=1e-9)
    # BTS baseline would just average to 0.5 -- arbitrary on threshold
    bts_p, _ = bts_style_mean_decision(dists)
    assert bts_p == pytest.approx(0.5, abs=1e-9)


def test_alpha_zero_recovers_maxmax():
    """alpha=0: P_alpha = max_r q_r (optimist / maxmax)."""
    out = alpha_meu_aggregate(
        "q",
        [[0.7, 0.3], [0.4, 0.6]],
        alpha=0.0,
        tau=1.0,  # disable abstention
    )
    assert out.p_alpha == pytest.approx(0.6, abs=1e-9)


def test_alpha_one_recovers_gilboa_schmeidler_maxmin():
    """alpha=1: P_alpha = min_r q_r (pure Gilboa-Schmeidler MEU)."""
    out = alpha_meu_aggregate(
        "q",
        [[0.7, 0.3], [0.4, 0.6]],
        alpha=1.0,
        tau=1.0,
    )
    assert out.p_alpha == pytest.approx(0.3, abs=1e-9)


def test_hurwicz_half_is_midpoint():
    """alpha=0.5 (Hurwicz): midpoint of min and max."""
    out = alpha_meu_aggregate(
        "q",
        [[0.7, 0.3], [0.4, 0.6]],
        alpha=0.5,
        tau=1.0,
    )
    assert out.p_alpha == pytest.approx(0.45, abs=1e-9)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        alpha_meu_aggregate("q", [[0.5, 0.5]], alpha=1.5)


def test_invalid_distribution_raises():
    with pytest.raises(ValueError):
        alpha_meu_aggregate("q", [[0.0, 0.0]], alpha=1.0)


def test_normalisation():
    """Unnormalised inputs should be re-normalised, not rejected."""
    out = alpha_meu_aggregate("q", [[1.0, 9.0], [2.0, 8.0]], alpha=1.0, tau=1.0)
    # row 1: [0.1, 0.9], row 2: [0.2, 0.8] -> q = [0.9, 0.8] -> min 0.8
    assert out.p_alpha == pytest.approx(0.8, abs=1e-9)
