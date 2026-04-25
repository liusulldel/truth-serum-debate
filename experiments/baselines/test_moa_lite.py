from __future__ import annotations
import pytest
from experiments.baselines import DebaterOutput
from experiments.baselines.moa_lite import aggregate


def test_confident_proposers_dominate():
    outs = [DebaterOutput(answer=1, p_true=0.95),
            DebaterOutput(answer=1, p_true=0.95),
            DebaterOutput(answer=0, p_true=0.5)]  # uninformative -> downweighted
    out = aggregate("q", outs)
    assert out.answer == 1 and out.p_true > 0.85


def test_consensus_false():
    outs = [DebaterOutput(answer=0, p_true=0.05), DebaterOutput(answer=0, p_true=0.10)]
    out = aggregate("q", outs)
    assert out.answer == 0 and out.p_true < 0.15


def test_uninformative_only_returns_half():
    outs = [DebaterOutput(answer=1, p_true=0.5), DebaterOutput(answer=0, p_true=0.5)]
    assert aggregate("q", outs).p_true == pytest.approx(0.5)


def test_no_abstain_field():
    assert not aggregate("q", [DebaterOutput(answer=1, p_true=0.9)]).abstain


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate("q", [])
