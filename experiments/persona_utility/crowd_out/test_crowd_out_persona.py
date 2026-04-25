"""Tests verifying the mock reproduces the Gneezy-Rustichini sign.

The mock is calibrated, not learned, so these tests act as guardrails:
they will fail if anyone re-tunes EFFORT_MEAN in a way that breaks the
"Pay Enough or Don't Pay at All" qualitative prediction.
"""
from __future__ import annotations

import statistics

import pytest

from mock_persona_debater import EFFORT_MEAN, persona_debater
from experiment import run


@pytest.fixture(scope="module")
def results():
    return run()


def _mean_effort(persona: str, n: int = 400) -> float:
    xs = [persona_debater(f"q{i}", persona, question_id=i)["effort"] for i in range(n)]  # type: ignore[arg-type]
    return statistics.fmean(xs)


def test_low_extrinsic_below_intrinsic(results):
    """Gneezy-Rustichini (QJE 2000) headline sign: low pay < no pay."""
    s = results["by_persona"]
    assert s["low_extrinsic"]["effort_mean"] < s["intrinsic"]["effort_mean"], (
        "Crowd-out should make low_extrinsic effort STRICTLY lower than intrinsic"
    )


def test_low_extrinsic_below_stateless(results):
    """Even vs the no-persona baseline, low pay should crowd out."""
    s = results["by_persona"]
    assert s["low_extrinsic"]["effort_mean"] < s["stateless"]["effort_mean"]


def test_high_extrinsic_recovers(results):
    """Sufficient pay should be close to intrinsic - within 5 effort points."""
    s = results["by_persona"]
    gap = s["intrinsic"]["effort_mean"] - s["high_extrinsic"]["effort_mean"]
    assert 0 <= gap <= 0.05, f"high_extrinsic should nearly match intrinsic, gap={gap}"


def test_accuracy_tracks_effort(results):
    """Accuracy ordering should follow effort ordering."""
    s = results["by_persona"]
    assert s["intrinsic"]["accuracy_mean"] > s["low_extrinsic"]["accuracy_mean"]
    assert s["high_extrinsic"]["accuracy_mean"] > s["low_extrinsic"]["accuracy_mean"]


def test_hedges_inverse_to_effort(results):
    """Low-effort personas hedge more (Argyle et al. 2023 style proxy)."""
    s = results["by_persona"]
    assert s["low_extrinsic"]["hedge_mean"] > s["intrinsic"]["hedge_mean"]


def test_contrasts_signed_correctly(results):
    c = results["contrasts"]
    assert c["intrinsic_minus_low_extrinsic_effort"] > 0
    assert c["stateless_minus_low_extrinsic_effort"] > 0
    assert results["crowd_out_replicated"] is True


def test_calibration_anchors_unchanged():
    """Hard-coded means must match the spec - guard against silent re-tuning."""
    assert EFFORT_MEAN["intrinsic"] == 0.85
    assert EFFORT_MEAN["high_extrinsic"] == 0.82
    assert EFFORT_MEAN["low_extrinsic"] == 0.55
    assert EFFORT_MEAN["stateless"] == 0.70
