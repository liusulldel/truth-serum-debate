"""Toy comparison: Holmstrom-team contract vs Bayesian Truth Serum.

Scenario where they DISAGREE
----------------------------
Two debaters answer "Is the Riemann hypothesis proven?" (ground truth = false).
- Debater A: answer=False, opponent_prediction(true)=0.1 (well calibrated, hard work).
- Debater B: answer=True,  opponent_prediction(true)=0.9 (lazy, mirrors own view).

BTS: B's "true" is *surprisingly common* relative to predictions (because A
predicted true=0.1, geometric mean of preds for 'true' is sqrt(0.1*0.9)=0.30
< empirical x_bar_true = 0.5). So B's information-score *can be positive*
and B may even win, despite being objectively wrong.

Holmstrom-forcing contract: aggregated verdict picks A (higher confidence),
ground_truth=false matches A, so A is paid the full bonus, B gets 0. Verdict
is correct.

This script prints the side-by-side numbers. Run with:
    python experiments/contract_holmstrom_teams/compare_to_bts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bts import bts_scores
from team_contract import DebaterReport, team_contract_decision


def main() -> None:
    question = "The Riemann hypothesis has been proven."
    ground_truth = 0  # FALSE; it has not been proven (as of 2026-04-25).

    # --- BTS view: 3 respondents, options 0=false, 1=true ---
    # Two confident-but-wrong (B,C say true) plus one truthful (A says false).
    # All three predict the population will mostly say "true" --- because B
    # and C are loud and A is meek. Under BTS this makes "false" the
    # surprisingly UNcommon answer (x_bar_false=1/3, y_bar_false~0.15), and
    # BTS *rewards* respondents who picked the SURPRISINGLY COMMON answer
    # ("true" here). So BTS will score B and C above A --- promoting the
    # majority lie. This is a known BTS failure mode under correlated bias.
    own = [0, 1, 1]
    preds = [
        [0.20, 0.80],  # A predicts most others will say true
        [0.10, 0.90],  # B confidently expects consensus on true
        [0.15, 0.85],  # C similar
    ]
    bts = bts_scores(own, preds, n_options=2)
    names = ["A", "B", "C"]
    bts_winner_idx = max(range(len(bts)), key=lambda i: bts[i])
    bts_winner = names[bts_winner_idx]
    bts_verdict = own[bts_winner_idx]

    # --- Holmstrom team-contract view ---
    # A is the sole truthful debater; B,C are confidently wrong.
    reports = [
        DebaterReport("A", answer=0, confidence=0.90),
        DebaterReport("B", answer=1, confidence=0.55),
        DebaterReport("C", answer=1, confidence=0.55),
    ]
    res = team_contract_decision(
        question, reports,
        contract="holmstrom_forcing", bonus=1.0,
        ground_truth=ground_truth,
    )

    print(f"Question: {question}")
    print(f"Ground truth: {'true' if ground_truth == 1 else 'false'}\n")

    print("--- BTS baseline ---")
    print("  Scores: " + ", ".join(f"{n}={s:+.4f}" for n, s in zip(names, bts)))
    print(f"  Winner: {bts_winner}; selected verdict = {'true' if bts_verdict else 'false'}")
    print(f"  Correct? {bts_verdict == ground_truth}\n")

    print("--- Holmstrom team contract (forcing) ---")
    print(f"  Aggregated verdict: {'true' if res.verdict == 1 else 'false'}")
    print(f"  Confidence: {res.confidence:.3f}")
    print(f"  Correct? {res.correct}")
    for p in res.payoffs:
        print(f"  Payoff[{p.name}]: gross={p.gross:.2f}, net={p.net:.2f}")


if __name__ == "__main__":
    main()
