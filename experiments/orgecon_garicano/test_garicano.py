"""Unit tests for the Garicano (2000) hierarchy aggregator.

Cases:
  1. Confident worker keeps decision -> judge NOT called, zero cost.
  2. Unconfident workers -> judge called, referral cost charged.
  3. Threshold = 0 -> worker layer always decides (flat org).
  4. Threshold = 1 -> always escalates (centralised org).
  5. Interior optimum: sweep tau on a synthetic batch where workers
     are reliable when confident and unreliable when not -- verifies
     Garicano's Prop. 2 prediction that net utility is maximised at
     an interior tau* and falls off on both sides.
  6. (BTS comparison, not strictly a unit test): on the same batch,
     compare net-utility of Garicano-aggregator vs. plurality vote
     ("BTS-style flat aggregation").
"""
from __future__ import annotations

from experiments.orgecon_garicano.garicano import (
    DebaterReport,
    garicano_decide,
    garicano_throughput,
)


def _perfect_judge(reports):
    """Mock judge: always returns the correct answer (set externally)."""
    # Tests inject the correct answer via a closure; default is 1.
    return _perfect_judge.next_answer  # type: ignore[attr-defined]


_perfect_judge.next_answer = 1  # type: ignore[attr-defined]


def test_confident_worker_keeps_decision():
    reports = [DebaterReport(answer=0, confidence=0.9),
               DebaterReport(answer=1, confidence=0.4)]
    d = garicano_decide(reports, judge=lambda r: 1,
                        worker_threshold=0.6, referral_cost=0.2)
    assert d.answer == 0
    assert d.used_judge is False
    assert d.realised_cost == 0.0


def test_unconfident_workers_escalate():
    reports = [DebaterReport(answer=0, confidence=0.3),
               DebaterReport(answer=1, confidence=0.4)]
    d = garicano_decide(reports, judge=lambda r: 1,
                        worker_threshold=0.6, referral_cost=0.2)
    assert d.answer == 1
    assert d.used_judge is True
    assert d.realised_cost == 0.2


def test_threshold_zero_is_flat_org():
    reports = [DebaterReport(answer=0, confidence=0.0),
               DebaterReport(answer=1, confidence=0.0)]
    d = garicano_decide(reports, judge=lambda r: 99,
                        worker_threshold=0.0, referral_cost=0.5)
    # tau = 0: every worker clears, no escalation.
    assert d.used_judge is False
    assert d.answer in (0, 1)


def test_threshold_one_always_escalates():
    reports = [DebaterReport(answer=0, confidence=0.99),
               DebaterReport(answer=1, confidence=0.95)]
    d = garicano_decide(reports, judge=lambda r: 7,
                        worker_threshold=1.0001 if False else 1.0,
                        referral_cost=0.5)
    # confidences strictly < 1.0 -> always escalates when tau = 1.0
    # (we use 0.99 to make this clean).
    # Note tau = 1.0 with confidence == 1.0 wouldn't escalate; that
    # boundary is irrelevant for the property we test.
    assert d.used_judge is True
    assert d.answer == 7


def test_interior_optimum_garicano_prop2():
    """Reproduce Garicano (2000) Prop. 2 in miniature.

    Build a synthetic batch where:
      - When max-confidence >= 0.7, the confident worker is correct.
      - When max-confidence <  0.7, the worker is wrong (50/50 noise),
        and the judge is always right.

    Then net utility should rise from tau=0 (workers handle hard
    problems they get wrong), peak near tau ~= 0.7, and fall again as
    tau -> 1 (paying h on problems workers would have nailed).
    """
    # Construct 20 problems: 10 "easy" (confident & correct), 10 "hard".
    problems = []
    for _ in range(10):
        # Easy: top debater confident & correct (gt = 1).
        problems.append((
            [DebaterReport(answer=1, confidence=0.85),
             DebaterReport(answer=0, confidence=0.4)],
            1,
        ))
    for _ in range(10):
        # Hard: top debater unconfident & wrong (gt = 1).
        problems.append((
            [DebaterReport(answer=0, confidence=0.55),
             DebaterReport(answer=0, confidence=0.5)],
            1,
        ))

    def judge(r):
        return 1  # perfect judge

    h = 0.3  # high referral cost -> sharpens Garicano's interior optimum

    # Sweep tau and find the optimum.
    sweep = {}
    for tau_int in range(0, 11):
        tau = tau_int / 10.0
        out = garicano_throughput(problems, judge, tau, h)
        sweep[tau] = out["net_utility"]

    # Interior optimum: best tau is strictly between 0 and 1.
    best_tau = max(sweep, key=sweep.get)
    assert 0.0 < best_tau < 1.0, f"Expected interior optimum, got {best_tau} (sweep={sweep})"

    # And it should beat both extremes by a non-trivial margin.
    assert sweep[best_tau] > sweep[0.0] + 0.05
    assert sweep[best_tau] > sweep[1.0] + 0.05


def test_compare_against_bts_flat_aggregation():
    """Garicano-aggregator vs. flat plurality vote on the same batch.

    BTS in this repo is a *scoring rule* for population reports; the
    natural ablation against an aggregator is "flat plurality vote with
    no escalation." We just check that on the same batch the Garicano
    aggregator achieves higher net utility (accuracy minus referral
    cost) than flat plurality, when an informative confidence signal
    exists. This is the empirical comparison the REPORT.md elaborates.
    """
    problems = []
    # 5 easy (confident, correct), 5 hard (unconfident, debaters wrong).
    for _ in range(5):
        problems.append((
            [DebaterReport(answer=1, confidence=0.9),
             DebaterReport(answer=0, confidence=0.3)],
            1,
        ))
    for _ in range(5):
        problems.append((
            [DebaterReport(answer=0, confidence=0.4),
             DebaterReport(answer=0, confidence=0.4)],
            1,
        ))

    def judge(r):
        return 1

    h = 0.1

    # Flat plurality (BTS-style ablation: no escalation, take majority).
    def plurality(reports):
        votes: dict[int, int] = {}
        for r in reports:
            votes[r.answer] = votes.get(r.answer, 0) + 1
        return max(votes, key=votes.get)
    flat_correct = sum(1 for rep, gt in problems if plurality(rep) == gt)
    flat_net = flat_correct / len(problems)  # zero referral cost

    g = garicano_throughput(problems, judge, worker_threshold=0.6,
                            referral_cost=h)

    assert g["net_utility"] > flat_net, (
        f"Garicano ({g['net_utility']:.3f}) should beat flat plurality "
        f"({flat_net:.3f}) when an informative confidence signal exists."
    )
