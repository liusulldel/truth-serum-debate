"""Toy scenario where CJT and BTS disagree -- the *Surprisingly Common*
trap.

Setup: 5 simulated debaters answer a deceptive trivia question whose
correct answer is unintuitive (Prelec 2004's exact "Philadelphia
is the capital of Pennsylvania" template, here adapted: the statement
is FALSE -- the capital is Harrisburg).

- 4/5 debaters are confidently wrong (vote True). They all *also*
  predict that "most others will say True" because the question
  pattern-matches a popular misconception.
- 1/5 debater is right (votes False). She predicts that "most others
  will incorrectly say True" -- i.e. her own minority answer is
  *surprisingly uncommon* given the population's predictions.

Wait: re-read. The Prelec mechanic rewards answers that are MORE
common than predicted. Here:
  empirical share of False = 1/5 = 0.20
  predicted share of False = ~ 0.10 (everyone thinks False is rare)
  -> False is "surprisingly common" (0.20 > 0.10) -> BTS rewards False.

CJT: simple majority -> picks True (the wrong answer).
BTS: highest-scoring respondent gets the right answer (False).

This is the canonical scenario where social-choice independence
assumptions actively hurt and meta-prediction (BTS) wins.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from cjt import aggregate                        # noqa: E402
from src.bts import bts_scores                   # noqa: E402


def main() -> None:
    question = "The capital of Pennsylvania is Philadelphia."  # FALSE
    # Encode False=0, True=1.
    own_answers = [1, 1, 1, 1, 0]               # 4 wrong, 1 right
    # Each respondent's predicted distribution over [False, True] for the
    # rest of the population. The wrong-but-confident majority all expect
    # the "True" answer to dominate; the lone correct respondent also
    # predicts that most peers will say True (she knows the misconception
    # is widespread).
    predicted = [
        [0.10, 0.90],
        [0.10, 0.90],
        [0.15, 0.85],
        [0.10, 0.90],
        [0.15, 0.85],
    ]

    cjt = aggregate(question, [bool(a) for a in own_answers])
    bts = bts_scores(own_answers, predicted, n_options=2)
    bts_winner_idx = max(range(len(bts)), key=lambda i: bts[i])
    bts_decision = bool(own_answers[bts_winner_idx])

    print(f"Question: {question}")
    print("Ground truth: False (capital is Harrisburg)")
    print()
    print(f"Votes (True=1, False=0): {own_answers}")
    print(f"Predicted dist [F, T]:   {predicted}")
    print()
    print(
        f"CJT majority: decision={cjt.decision} "
        f"(yes={cjt.yes_votes}/{cjt.n}, share={cjt.yes_share:.2f}) "
        f"-> {'CORRECT' if cjt.decision is False else 'WRONG'}"
    )
    print()
    print("BTS scores per respondent:")
    for i, (a, s) in enumerate(zip(own_answers, bts)):
        marker = " <-- top" if i == bts_winner_idx else ""
        print(f"  r{i}: answer={'T' if a == 1 else 'F'}  bts={s:+.4f}{marker}")
    print()
    print(
        f"BTS top-scorer's answer: {bts_decision} "
        f"-> {'CORRECT' if bts_decision is False else 'WRONG'}"
    )


if __name__ == "__main__":
    main()
