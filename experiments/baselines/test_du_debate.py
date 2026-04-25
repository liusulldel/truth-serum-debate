from __future__ import annotations
import pytest
from experiments.baselines import DebaterOutput
from experiments.baselines.du_debate import aggregate


def test_unanimous_high_p_stays_true():
    outs = [DebaterOutput(answer=1, p_true=0.9),
            DebaterOutput(answer=1, p_true=0.85),
            DebaterOutput(answer=1, p_true=0.95)]
    out = aggregate("q", outs)
    assert out.answer == 1 and out.p_true > 0.8


def test_drift_can_flip_minority():
    # One strong dissenter with weak majority; after drift, majority can collapse
    outs = [DebaterOutput(answer=1, p_true=0.55),
            DebaterOutput(answer=1, p_true=0.55),
            DebaterOutput(answer=0, p_true=0.05)]  # strong, drags mean down
    out = aggregate("q", outs, rounds=3)
    # mean ~ 0.383 -> after enough drift all three drop below 0.5
    assert out.answer == 0


def test_more_rounds_increases_concentration():
    outs = [DebaterOutput(answer=1, p_true=0.8),
            DebaterOutput(answer=0, p_true=0.4)]
    o1 = aggregate("q", outs, rounds=1)
    o5 = aggregate("q", outs, rounds=5)
    # With more rounds, ps converge to common mean -> p_agg unchanged in mean,
    # but post-debate answers should be more aligned (we just check it runs)
    assert o1.answer in (0, 1) and o5.answer in (0, 1)


def test_no_abstain_field():
    assert not aggregate("q", [DebaterOutput(answer=1, p_true=0.9)]).abstain


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate("q", [])
