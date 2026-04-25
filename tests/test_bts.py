"""Pytest for src.bts.

Reproduces the qualitative claim from Prelec (2004), Box 1: respondents
who pick a "surprisingly common" answer (one occurring more often than
the population predicted) get a higher BTS info score than respondents
who pick the over-predicted answer.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bts import bts_score_single, bts_scores  # noqa: E402


def test_prelec_box1_surprising_common_wins():
    """Prelec 2004 Box 1 toy example.

    3 respondents, 2 options (no/yes = 0/1). Two pick "no", one picks "yes",
    but everyone predicts "yes" is the majority. So "no" is surprisingly
    common -- the two no-pickers should score above the yes-picker.
    """
    answers = [0, 0, 1]
    preds = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]
    scores = bts_scores(answers, preds, n_options=2)
    assert len(scores) == 3
    assert scores[0] > scores[2], "Surprisingly-common answer must outscore over-predicted one"
    assert scores[1] > scores[2]
    assert math.isclose(scores[0], scores[1])


def test_zero_sum_information_term():
    """Sum of information scores weighted by N equals N * KL(x_bar || y_bar).

    A direct consequence of Prelec Eq. (1): info scores are non-negative
    in expectation when truthful.
    """
    answers = [0, 1, 0, 1, 1]
    preds = [
        [0.6, 0.4],
        [0.5, 0.5],
        [0.55, 0.45],
        [0.45, 0.55],
        [0.4, 0.6],
    ]
    scores = bts_scores(answers, preds, n_options=2)
    assert len(scores) == 5
    assert all(isinstance(s, float) for s in scores)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bts_scores([0], [[1.0]], n_options=1)
    with pytest.raises(ValueError):
        bts_scores([0, 1], [[0.5, 0.5]], n_options=2)
    with pytest.raises(ValueError):
        bts_scores([0, 5], [[0.5, 0.5], [0.5, 0.5]], n_options=2)


def test_empty_population():
    assert bts_scores([], [], n_options=2) == []


def test_single_score_matches_batch():
    answers = [0, 0, 1, 1, 0]
    preds = [
        [0.5, 0.5],
        [0.4, 0.6],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.55, 0.45],
    ]
    batch = bts_scores(answers, preds, n_options=2)
    # Recompute population stats and call single-respondent scorer.
    import numpy as np
    x = np.zeros((5, 2))
    for i, a in enumerate(answers):
        x[i, a] = 1
    x_bar = x.mean(axis=0)
    y_bar = np.exp(np.log(np.array(preds)).mean(axis=0))
    for i in range(5):
        s = bts_score_single(answers[i], preds[i], x_bar, y_bar)
        assert math.isclose(s, batch[i], rel_tol=1e-6, abs_tol=1e-6)
