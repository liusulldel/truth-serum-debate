"""Head-to-head: alpha-MEU vs BTS-style mean aggregation on a toy disagreement.

Scenario: two debaters on the same question. Debater A says P(true)=0.9,
Debater B says P(true)=0.1. The arithmetic-mean (BTS-style) judge gets 0.5
and must flip a coin; alpha-MEU detects the ambiguity and abstains.

Run: python -m experiments.ambig_alpha_meu.compare_vs_bts
"""
from __future__ import annotations

from experiments.ambig_alpha_meu.alpha_meu import (
    alpha_meu_aggregate,
    bts_style_mean_decision,
)


def main() -> None:
    scenarios = [
        (
            "STRONG DISAGREEMENT (the OOD-style failure mode)",
            [[0.1, 0.9], [0.9, 0.1]],
        ),
        (
            "Mild disagreement",
            [[0.4, 0.6], [0.6, 0.4]],
        ),
        (
            "Consensus TRUE",
            [[0.05, 0.95], [0.10, 0.90], [0.08, 0.92]],
        ),
        (
            "One outlier among three confident-true debaters",
            [[0.1, 0.9], [0.05, 0.95], [0.85, 0.15]],
        ),
    ]

    print(f"{'scenario':50s} | {'BTS_p':>6s} {'BTS':>6s} | "
          f"{'p_min':>6s} {'p_max':>6s} {'A':>5s} {'p_a':>6s} {'MEU':>8s}")
    print("-" * 110)
    for label, dists in scenarios:
        bts_p, bts_dec = bts_style_mean_decision(dists)
        meu = alpha_meu_aggregate(label, dists, alpha=1.0, tau=0.4)
        print(
            f"{label[:50]:50s} | "
            f"{bts_p:6.3f} {bts_dec:>6s} | "
            f"{meu.p_min:6.3f} {meu.p_max:6.3f} "
            f"{meu.ambiguity_index:5.2f} {meu.p_alpha:6.3f} {meu.decision:>8s}"
        )


if __name__ == "__main__":
    main()
