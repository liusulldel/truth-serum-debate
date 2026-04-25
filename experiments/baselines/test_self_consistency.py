from __future__ import annotations
import pytest
from experiments.baselines import DebaterOutput
from experiments.baselines.self_consistency import aggregate


def test_majority_within_then_across():
    outs = [DebaterOutput(answer=1, p_true=0.8, samples=(1, 1, 1, 0, 1)),
            DebaterOutput(answer=1, p_true=0.7, samples=(1, 0, 1, 1, 1)),
            DebaterOutput(answer=0, p_true=0.4, samples=(0, 0, 0, 1, 0))]
    assert aggregate("q", outs).answer == 1


def test_falls_back_to_answer_when_no_samples():
    outs = [DebaterOutput(answer=0, p_true=0.2),
            DebaterOutput(answer=0, p_true=0.3),
            DebaterOutput(answer=1, p_true=0.6)]
    assert aggregate("q", outs).answer == 0


def test_p_true_is_mean_of_per_debater_p1():
    outs = [DebaterOutput(answer=1, p_true=0.0, samples=(1, 1, 1, 1)),
            DebaterOutput(answer=0, p_true=0.0, samples=(0, 0, 0, 0))]
    assert aggregate("q", outs).p_true == pytest.approx(0.5)


def test_empty_raises():
    with pytest.raises(ValueError):
        aggregate("q", [])
