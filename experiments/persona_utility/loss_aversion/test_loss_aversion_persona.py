"""Unit tests verifying mock parameters match published anchors.

Anchors:
    Horton (2023, NBER w31122): persona-prompted LLMs reproduce direction but
        not full magnitude of behavioral effects.
    Aher, Arriaga & Kalai (2023, ICML, arXiv:2208.10264): silicon subjects
        have reduced between-subject variance.
    Mei, Xie, Yuan & Jackson (2024, PNAS 121(9), doi:10.1073/pnas.2313925121):
        LLM behavioral lambda ~ 1.3-1.5 vs human ~2.25.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiment import make_questions, run_all, run_for_persona
from mock_persona_debater import (
    get_params,
    persona_debater,
    reset_rng,
)


def test_stateless_has_zero_frame_effect_per_THREE_FAILURES():
    """Stateless LLM null: no kink (THREE_FAILURES.md, Failure 1)."""
    p = get_params("stateless")
    assert p["frame_effect"] == 0.0, "stateless must encode the null hypothesis"


def test_loss_averse_persona_lambda_in_Mei2024_band():
    """Mei et al. (2024) PNAS: LLM behavioral lambda ~ 1.3-1.5.

    Our frame_effect of 0.09 corresponds to a kink-magnitude that is
    ~40% (= 0.09 / 0.225) of a calibration where 0.225 maps to lambda=2.25.
    Equivalently, implied_lambda = 2.25 * (frame_effect / 0.225).
    """
    fe = get_params("loss_averse_persona")["frame_effect"]
    implied_lambda = 2.25 * (fe / 0.225)
    assert 1.3 <= implied_lambda <= 1.5, (
        f"implied lambda {implied_lambda:.2f} outside Mei (2024) band [1.3, 1.5]"
    )


def test_persona_variance_below_stateless_per_Aher2023():
    """Aher et al. (2023): persona-prompted variance < stateless variance.

    Implementation: each persona's sigma must be <= stateless sigma.
    """
    stateless_sigma = get_params("stateless")["sigma"]
    for persona in ("loss_averse_persona", "gain_seeking_persona", "neutral_persona"):
        sigma = get_params(persona)["sigma"]
        assert sigma <= stateless_sigma, (
            f"{persona} sigma {sigma} should be <= stateless sigma {stateless_sigma}"
        )


def test_loss_averse_persona_t_dominates_stateless_t():
    """End-to-end: matched-frame paired-t for loss_averse_persona should be
    well-separated from stateless persona t (which should be near zero).

    Per Horton (2023) + Mei (2024): persona-prompted effect is *attenuated
    but detectable*, stateless is null."""
    results = {r.persona: r for r in run_all()}
    t_stateless = results["stateless"].paired_t
    t_loss_persona = results["loss_averse_persona"].paired_t
    assert abs(t_stateless) < 3.0, f"stateless t should be near zero, got {t_stateless:.2f}"
    assert t_loss_persona > 5.0, (
        f"loss_averse_persona t should be highly significant, got {t_loss_persona:.2f}"
    )
    assert t_loss_persona > 2 * abs(t_stateless), (
        "loss-averse persona effect must dominate stateless surface noise"
    )


def test_gain_seeking_persona_has_opposite_sign_kink():
    """Gain-seeking persona: shifts mass *toward* confidence under loss frame
    (anti-loss-averse). Sign flip required for construct validity."""
    fe_loss = get_params("loss_averse_persona")["frame_effect"]
    fe_gain = get_params("gain_seeking_persona")["frame_effect"]
    assert fe_loss > 0 and fe_gain < 0, (
        "construct check: loss_averse positive, gain_seeking negative"
    )


def test_persona_debater_returns_valid_distribution():
    """Output must be a proper distribution over {true, false}."""
    reset_rng(42)
    q = {"qid": 0, "base_difficulty": 0.4, "true_answer": True}
    for persona in ("stateless", "loss_averse_persona", "gain_seeking_persona", "neutral_persona"):
        for framing in ("loss_frame", "gain_frame"):
            resp = persona_debater(q, persona, framing)
            assert set(resp.keys()) == {"true", "false"}
            assert 0.0 <= resp["true"] <= 1.0
            assert 0.0 <= resp["false"] <= 1.0
            assert abs(resp["true"] + resp["false"] - 1.0) < 1e-9


def test_unknown_persona_raises():
    q = {"qid": 0, "base_difficulty": 0.5, "true_answer": True}
    with pytest.raises(ValueError):
        persona_debater(q, "homo_economicus", "loss_frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        persona_debater(q, "stateless", "no_frame")  # type: ignore[arg-type]
