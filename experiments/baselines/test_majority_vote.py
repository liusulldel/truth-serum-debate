from __future__ import annotations
import pytest
from experiments.baselines import DebaterOutput
from experiments.baselines.majority_vote import aggregate


def _d(answer: int, p: float = 0.6) -> DebaterOutput:
    return DebaterOutput(answer=answer, p_true=p)


def test_unanimous_true():
    out = aggregate("q", [_d(1), _d(1), _d(1)])
    assert out.answer == 1 and out.p_true == pytest.approx(1.0)


def test_unanimous_false():
    out = aggregate("q", [_d(0), _d(0)])
    assert out.answer == 0 and not out.abstain


def test_split_majority_true():
    out = aggregate("q", [_d(1), _d(1), _d(0)])
    assert out.answer == 1 and out.p_true == pytest.approx(2 / 3)


def test_tie_resolves_to_false():
    out = aggregate("q", [_d(1), _d(0)])
    assert out.answer == 0 and out.p_true == pytest.approx(0.5)


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate("q", [])
