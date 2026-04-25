from __future__ import annotations
import pytest
from experiments.baselines import DebaterOutput
from experiments.baselines.vanilla_debate import aggregate


def test_more_rebuttable_true_wins():
    outs = [DebaterOutput(answer=1, p_true=0.6, rebuttal_strength=0.9),
            DebaterOutput(answer=0, p_true=0.4, rebuttal_strength=0.4)]
    out = aggregate("q", outs)
    assert out.answer == 1 and out.p_true > 0.5


def test_more_rebuttable_false_wins():
    outs = [DebaterOutput(answer=1, p_true=0.6, rebuttal_strength=0.2),
            DebaterOutput(answer=0, p_true=0.4, rebuttal_strength=0.8)]
    out = aggregate("q", outs)
    assert out.answer == 0 and out.p_true < 0.5


def test_tie_falls_back_to_plurality():
    outs = [DebaterOutput(answer=1, p_true=0.6, rebuttal_strength=0.5),
            DebaterOutput(answer=1, p_true=0.6, rebuttal_strength=0.5),
            DebaterOutput(answer=0, p_true=0.4, rebuttal_strength=0.5)]
    assert aggregate("q", outs).answer == 1


def test_requires_two_debaters():
    with pytest.raises(ValueError):
        aggregate("q", [DebaterOutput(answer=1, p_true=0.6, rebuttal_strength=0.5)])
