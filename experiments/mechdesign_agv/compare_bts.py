"""Toy scenario where BTS and AGV give meaningfully different decisions/scores.

Scenario: 3 debaters on a controversial proposition.
  - A: confident TRUE (own=0.85), expects others to also say TRUE.
  - B: confident TRUE (own=0.80), expects others to say TRUE.
  - C: weakly FALSE (own=0.40), but EXPECTS that everyone else will say TRUE.

In Prelec's BTS framework, C's answer (FALSE) becomes "surprisingly common"
relative to predictions ONLY IF the empirical share for FALSE exceeds what
others predicted. Here: empirical x_bar(FALSE)=1/3, but A and B predicted
the FALSE share to be tiny (~0.15). So C's FALSE gets a big positive info
score in BTS -- BTS *rewards* the lone dissenter as a likely truth-teller,
because Prelec's rule is designed exactly to surface contrarian-but-correct
private signals (the famous "surprisingly common" effect).

AGV instead aggregates valuations: sum(t_i) = 0.85+0.80+0.40 = 2.05 > 1.5,
so AGV picks TRUE.

Which is "more correct"? It depends on the ground-truth question. If the
proposition is empirically TRUE (say, "vaccines reduce mortality"), AGV's
decision is correct and BTS's high score for the contrarian would mislead
a downstream "trust the BTS-winner" pipeline. If the proposition is one
where the majority is captured by a popular myth (say, "humans only use
10% of their brain"), BTS's contrarian-rewarding behaviour is the right
prior. So they implement DIFFERENT epistemic philosophies, not strictly
better/worse -- this is the honest finding.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agv import DebaterReport, agv_aggregate  # noqa: E402
from src.bts import bts_scores  # noqa: E402


def main() -> None:
    # Convert beliefs into discrete answers (TRUE=0, FALSE=1 — matching Prelec doctest convention)
    # We'll use TRUE=1, FALSE=0 for clarity throughout.
    own_answers = [1, 1, 0]  # A:TRUE, B:TRUE, C:FALSE
    # Predicted distribution over [FALSE, TRUE]:
    predicted = [
        [0.15, 0.85],  # A predicts most others say TRUE
        [0.20, 0.80],  # B predicts most others say TRUE
        [0.10, 0.90],  # C ALSO predicts others will say TRUE (knows opinion is unpopular)
    ]
    bts = bts_scores(own_answers, predicted, n_options=2)
    print("=== BTS scores (higher = more likely truthful per Prelec) ===")
    for name, s in zip("ABC", bts):
        print(f"  {name}: {s:+.4f}")
    bts_winner = "ABC"[max(range(3), key=lambda i: bts[i])]
    print(f"  BTS winner (highest score): {bts_winner}")

    # AGV input: own_belief = P(TRUE), peer_belief_means inferred from predicted[i][TRUE]
    reps = [
        DebaterReport(own_belief=0.85, peer_belief_means=[0.85, 0.85]),
        DebaterReport(own_belief=0.80, peer_belief_means=[0.85, 0.85]),
        DebaterReport(own_belief=0.40, peer_belief_means=[0.90, 0.90]),
    ]
    agv = agv_aggregate("Toy proposition.", reps)
    print()
    print("=== AGV aggregator ===")
    print(f"  decision: {'TRUE' if agv['decision'] == 1 else 'FALSE'}")
    print(f"  aggregated belief E[T]: {agv['aggregated_belief']:.3f}")
    for name, s, t in zip("ABC", agv["per_agent_score"], agv["transfers"]):
        print(f"  {name}: score={s:+.4f}  transfer={t:+.4f}")
    print(f"  sum(transfers) = {sum(agv['transfers']):+.2e}  (should be ~0)")

    print()
    print("=== Disagreement summary ===")
    print(f"  BTS rewards C (lone dissenter) most: winner={bts_winner}")
    print(f"  AGV picks decision = {'TRUE' if agv['decision'] == 1 else 'FALSE'} (sum-of-valuations rule)")
    print("  These answer DIFFERENT questions: BTS asks 'who is truthful?',")
    print("  AGV asks 'what should we collectively decide?'.")


if __name__ == "__main__":
    main()
