from __future__ import annotations
from experiments.baselines import DebaterOutput
from experiments.benchmark import (REGIMES, alpha_meu_adapter,
                                   generate_question_set, run_benchmark)
from experiments.baselines.majority_vote import aggregate as mv_aggregate
from experiments.full_stack import full_stack_aggregate


def test_question_set_reproducible():
    a = generate_question_set(REGIMES["easy"])
    b = generate_question_set(REGIMES["easy"])
    assert [t for t, _ in a] == [t for t, _ in b]
    assert a[0][1][0].p_true == b[0][1][0].p_true


def test_majority_easy_high_accuracy():
    res = run_benchmark(mv_aggregate, generate_question_set(REGIMES["easy"]), n_bootstrap=100)
    assert res["accuracy"] > 0.7
    assert res["ci_low"] <= res["accuracy"] <= res["ci_high"]


def test_full_stack_runs_and_has_all_fields():
    res = run_benchmark(full_stack_aggregate, generate_question_set(REGIMES["hard"]), n_bootstrap=100)
    for k in ["accuracy", "ci_low", "ci_high", "ece", "runtime_ms",
              "calibrated_abstention_precision", "abstention_rate"]:
        assert k in res
    assert res["abstention_rate"] >= 0.0


def test_alpha_meu_adapter_returns_valid_decision():
    res = run_benchmark(alpha_meu_adapter, generate_question_set(REGIMES["medium"])[:20], n_bootstrap=50)
    assert 0.0 <= res["accuracy"] <= 1.0 and 0.0 <= res["ece"] <= 1.0


def test_full_stack_high_route_threshold_forces_abstention():
    outs = [DebaterOutput(answer=1, p_true=0.9, confidence=0.3),
            DebaterOutput(answer=0, p_true=0.1, confidence=0.3)]
    assert full_stack_aggregate("q", outs, tau_route=0.99, tau_ambig=0.4).abstain
