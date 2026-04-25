"""Tests for the tournament/status persona-utility experiment.

Covers:
    1. determinism under fixed seed
    2. mean-effort lift (B-S-W signature: tournament_ahead beats stateless)
    3. Lazear-Rosen Prop. 4 risk-shifting (var(behind) > var(ahead))
    4. behind contestant exhibits action-space variance (Hail-Mary flips reach
       answers a strict thresholder would never pick)
    5. rank inference helper resolves to the right tertile
"""
from __future__ import annotations

import numpy as np
import pytest

from experiment import (
    PERSONAS,
    effort_lift_signature,
    make_questions,
    risk_shifting_signature,
    run_all,
    run_for_persona,
)
from mock_persona_debater import (
    _infer_persona_from_rank,
    persona_debater,
    reset_rng,
)


def test_determinism_under_seed():
    """Same seed -> identical results across runs."""
    a = run_all()
    b = run_all()
    for ra, rb in zip(a, b):
        assert ra.persona == rb.persona
        assert ra.accuracy_mean == pytest.approx(rb.accuracy_mean, abs=1e-12)
        assert ra.p_correct_var == pytest.approx(rb.p_correct_var, abs=1e-12)


def test_effort_lift_bull_schotter_weigelt():
    """tournament_ahead and tournament_middle should beat stateless on accuracy.

    This is the Bull-Schotter-Weigelt 1987 mean-effort lift, attenuated to
    LLM scale (~0.6x of human magnitudes).
    """
    results = run_all()
    es = effort_lift_signature(results)
    assert es["ahead_lift"] > 0.0, (
        f"Expected positive ahead-lift; got {es['ahead_lift']:.3f}"
    )
    assert es["middle_lift"] > 0.0, (
        f"Expected positive middle-lift; got {es['middle_lift']:.3f}"
    )
    # Lift magnitudes consistent with calibration (delta_acc=0.07-0.09).
    assert es["ahead_lift"] >= es["middle_lift"] - 0.02


def test_lazear_rosen_prop4_risk_shifting():
    """Variance of p_correct should be HIGHER for behind than for ahead.

    This is the central testable prediction of Lazear & Rosen (1981) Prop. 4:
    contestants behind in a tournament optimally raise the variance of their
    action because only a tail outcome can change their rank.
    """
    results = run_all()
    rs = risk_shifting_signature(results)
    assert rs["signature_present"], (
        f"Expected var(behind) > var(ahead); got "
        f"{rs['var_p_correct_behind']:.4f} vs {rs['var_p_correct_ahead']:.4f}"
    )
    # Ratio should be substantial (calibrated ~6x: sigma 0.115 vs 0.045 -> var 13.2e-3 vs 2.0e-3)
    assert rs["ratio_behind_over_ahead"] > 2.0, (
        f"Variance ratio behind/ahead = {rs['ratio_behind_over_ahead']:.2f}; "
        "expected >2.0 if risk-shifting is operative."
    )


def test_behind_takes_more_action_space_risk():
    """Behind contestants should sometimes flip to the LOW-prob side.

    Operationalized as: across 200 questions, the number of items where the
    model's answer disagrees with its own argmax(p_correct, 1-p_correct) is
    strictly larger for tournament_behind than for tournament_ahead.
    This is the Hail-Mary channel of Lazear-Rosen Prop. 4 transposed into
    discrete-action space.
    """
    questions = make_questions()
    flips = {}
    for persona in ("tournament_ahead", "tournament_behind"):
        reset_rng(seed=42)
        rank = 3 if persona == "tournament_ahead" else 17
        n_flip = 0
        for q in questions:
            r = persona_debater(q, persona, current_rank=rank, n_competitors=20)
            internal_pick_correct = r["p_correct"] >= 0.5
            picked_correct = (r["answer"] == q["true_answer"])
            if internal_pick_correct != picked_correct:
                n_flip += 1
        flips[persona] = n_flip
    assert flips["tournament_behind"] > flips["tournament_ahead"], (
        f"Hail-Mary flips: behind={flips['tournament_behind']}, "
        f"ahead={flips['tournament_ahead']}; behind should be higher."
    )
    # ahead persona is calibrated to flip_prob=0
    assert flips["tournament_ahead"] == 0


def test_rank_inference_tertiles():
    """_infer_persona_from_rank: top third->ahead, bottom third->behind, else middle."""
    assert _infer_persona_from_rank(1, 20) == "tournament_ahead"
    assert _infer_persona_from_rank(6, 20) == "tournament_ahead"   # 6 <= 20/3 ~ 6.67
    assert _infer_persona_from_rank(10, 20) == "tournament_middle"
    assert _infer_persona_from_rank(14, 20) == "tournament_behind"  # >= 13.33
    assert _infer_persona_from_rank(20, 20) == "tournament_behind"
    # degenerate
    assert _infer_persona_from_rank(1, 1) == "tournament_middle"


def test_all_personas_run_without_error():
    """Smoke test: every named persona produces sensible output ranges."""
    questions = make_questions(n=50)
    for persona in PERSONAS:
        reset_rng(seed=42)
        result = run_for_persona(persona, questions)
        assert 0.0 <= result.accuracy_mean <= 1.0
        assert result.p_correct_var >= 0.0
        assert 0.0 <= result.confidence_mean <= 1.0
