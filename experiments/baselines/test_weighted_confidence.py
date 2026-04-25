from __future__ import annotations
import pytest
from experiments.baselines import DebaterOutput
from experiments.baselines.weighted_confidence import aggregate


def test_unanimous_high_confidence_true():
    outs = [DebaterOutput(answer=1, p_true=0.9, confidence=1.0) for _ in range(3)]
    out = aggregate("q", outs)
    assert out.answer == 1 and out.p_true > 0.99


def test_loud_uncalibrated_does_not_dominate_two_calibrated():
    outs = [DebaterOutput(answer=0, p_true=0.51, confidence=10.0),
            DebaterOutput(answer=1, p_true=0.9, confidence=1.0),
            DebaterOutput(answer=1, p_true=0.9, confidence=1.0)]
    assert aggregate("q", outs).answer == 1


def test_balanced_returns_half():
    outs = [DebaterOutput(answer=1, p_true=0.7, confidence=1.0),
            DebaterOutput(answer=0, p_true=0.3, confidence=1.0)]
    assert aggregate("q", outs).p_true == pytest.approx(0.5)


def test_no_abstain():
    assert not aggregate("q", [DebaterOutput(answer=1, p_true=0.5, confidence=1.0)]).abstain


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate("q", [])
