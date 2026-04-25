from __future__ import annotations
import pytest
from experiments.baselines import DebaterOutput
from experiments.baselines.self_moa import aggregate


def test_anchor_dominates_via_shrinkage():
    # High-confidence anchor at p_true=0.9 pulls a 0.4 dissenter up past 0.5
    outs = [DebaterOutput(answer=1, p_true=0.9, confidence=0.95),
            DebaterOutput(answer=0, p_true=0.4, confidence=0.5)]
    out = aggregate("q", outs)
    assert out.answer == 1 and out.p_true > 0.5


def test_unanimous_false():
    outs = [DebaterOutput(answer=0, p_true=0.05, confidence=0.9),
            DebaterOutput(answer=0, p_true=0.10, confidence=0.8)]
    out = aggregate("q", outs)
    assert out.answer == 0 and out.p_true < 0.2


def test_low_confidence_anchor_still_anchor():
    # Even a weak-confidence proposer is the anchor if no one beats it
    outs = [DebaterOutput(answer=1, p_true=0.7, confidence=0.4),
            DebaterOutput(answer=1, p_true=0.7, confidence=0.3)]
    out = aggregate("q", outs)
    assert out.answer == 1 and out.p_true == pytest.approx(0.7)


def test_no_abstain_field():
    assert not aggregate("q", [DebaterOutput(answer=1, p_true=0.9, confidence=0.9)]).abstain


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate("q", [])
