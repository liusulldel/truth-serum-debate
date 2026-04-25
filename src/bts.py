"""Bayesian Truth Serum (BTS) scoring.

Implements the scoring rule from:
    Prelec, D. (2004). "A Bayesian Truth Serum for Subjective Data."
    Science 306(5695): 462-466.

Each respondent supplies:
  1. ``own_answer``: the option (0..m-1) they personally endorse.
  2. ``predicted_answer_distribution``: their predicted *empirical*
     frequency over the m options across the rest of the population.

For each option k, define
  x_bar_k = average of indicator (own_answer == k) across respondents
  y_bar_k = geometric mean of predicted_distribution[k] across respondents

A respondent's BTS score is the sum of two terms (information + prediction):

  info_score(r) = sum_k [ x_r,k * log( x_bar_k / y_bar_k ) ]
  pred_score(r) = sum_k [ x_bar_k * log( y_r,k / x_bar_k ) ]
  bts(r)        = info_score(r) + pred_score(r)

The information score is positive for "surprisingly common" answers --
answers that occur more often than the population predicted -- which is
the property that incentivises truthful reporting. See Prelec (2004) Eq. (1).

Doctest uses the toy example in Prelec (2004), Box 1: a 3-respondent,
2-option survey where the answer "yes" is surprisingly common.

>>> answers = [0, 0, 1]
>>> preds = [[0.5, 0.5], [0.4, 0.6], [0.4, 0.6]]
>>> scores = bts_scores(answers, preds, n_options=2)
>>> # Respondent 2 ("no") was the only one to underestimate "yes" share
>>> # but also picked the unsurprising answer -- check ordering
>>> scores[0] > scores[2]
True
>>> scores[1] > scores[2]
True
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_EPS = 1e-9


def _clip_distribution(dist: Sequence[float]) -> np.ndarray:
    """Renormalise + clip a distribution to (eps, 1-eps) to keep logs finite."""
    arr = np.asarray(dist, dtype=float)
    arr = np.clip(arr, _EPS, None)
    arr = arr / arr.sum()
    arr = np.clip(arr, _EPS, 1.0)
    return arr


def bts_scores(
    own_answers: Sequence[int],
    predicted_distributions: Sequence[Sequence[float]],
    n_options: int,
) -> list[float]:
    """Compute Bayesian Truth Serum scores for a population of respondents.

    Args:
        own_answers: Length-N list of integer answer indices in [0, n_options).
        predicted_distributions: Length-N list, each a length-``n_options``
            probability vector summing to 1, giving each respondent's
            prediction over the empirical distribution of others' answers.
        n_options: Number of discrete answer options ``m``.

    Returns:
        Length-N list of float BTS scores. Higher = more likely truthful by
        Prelec's rule.

    Raises:
        ValueError: if input shapes disagree or n_options < 2.
    """
    if n_options < 2:
        raise ValueError("BTS requires at least 2 answer options.")
    n = len(own_answers)
    if n != len(predicted_distributions):
        raise ValueError("own_answers and predicted_distributions length mismatch.")
    if n == 0:
        return []

    # Indicator matrix: shape (N, m), x[r, k] = 1 iff respondent r picked k.
    x = np.zeros((n, n_options), dtype=float)
    for r, ans in enumerate(own_answers):
        if not (0 <= ans < n_options):
            raise ValueError(f"answer {ans} out of range [0,{n_options})")
        x[r, ans] = 1.0

    # Predictions matrix: shape (N, m).
    y = np.stack([_clip_distribution(p) for p in predicted_distributions], axis=0)
    if y.shape != (n, n_options):
        raise ValueError(
            f"predicted_distributions must each have length {n_options};"
            f" got shape {y.shape}"
        )

    # Empirical distribution x_bar (uniform mean of indicators).
    x_bar = x.mean(axis=0)
    x_bar = np.clip(x_bar, _EPS, 1.0)

    # Geometric mean of predicted distributions y_bar.
    log_y = np.log(y)
    y_bar = np.exp(log_y.mean(axis=0))
    y_bar = np.clip(y_bar, _EPS, 1.0)

    # Information score: sum_k x_r,k * log(x_bar_k / y_bar_k).
    info = x @ np.log(x_bar / y_bar)

    # Prediction score: sum_k x_bar_k * log(y_r,k / x_bar_k).
    pred = (np.log(y) - np.log(x_bar)[None, :]) @ x_bar

    return (info + pred).tolist()


def bts_score_single(
    own_answer: int,
    predicted_distribution: Sequence[float],
    population_x_bar: Sequence[float],
    population_y_bar: Sequence[float],
) -> float:
    """Score a single respondent given pre-computed population statistics.

    Useful for the eval harness where the population is built incrementally.

    Args:
        own_answer: This respondent's chosen option index.
        predicted_distribution: This respondent's predicted distribution.
        population_x_bar: Empirical answer frequencies across the population.
        population_y_bar: Geometric mean of predicted distributions.

    Returns:
        BTS score (info + prediction) as a float.
    """
    x_bar = np.clip(np.asarray(population_x_bar, float), _EPS, 1.0)
    y_bar = np.clip(np.asarray(population_y_bar, float), _EPS, 1.0)
    y = _clip_distribution(predicted_distribution)
    info = math.log(x_bar[own_answer] / y_bar[own_answer])
    pred = float(np.sum(x_bar * (np.log(y) - np.log(x_bar))))
    return info + pred


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
