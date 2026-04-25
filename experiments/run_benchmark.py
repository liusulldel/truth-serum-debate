"""Phase-3 driver: runs all aggregators on all 3 regimes; emits JSON + table.
Usage:  python -m experiments.run_benchmark
"""
from __future__ import annotations
import json
import numpy as np
from experiments.baselines.majority_vote import aggregate as majority_vote
from experiments.baselines.weighted_confidence import aggregate as weighted_confidence
from experiments.baselines.self_consistency import aggregate as self_consistency
from experiments.baselines.vanilla_debate import aggregate as vanilla_debate
from experiments.baselines.moa_lite import aggregate as moa_lite
from experiments.baselines.self_moa import aggregate as self_moa
from experiments.baselines.du_debate import aggregate as du_debate
from experiments.benchmark import (REGIMES, alpha_meu_adapter, bts_alpha_meu_hybrid_adapter,
                                   bts_top_score_adapter, garicano_adapter,
                                   generate_question_set, run_benchmark)
from experiments.full_stack import full_stack_aggregate

AGGREGATORS = {
    "majority_vote": majority_vote, "weighted_confidence": weighted_confidence,
    "self_consistency": self_consistency, "vanilla_debate": vanilla_debate,
    "moa_lite": moa_lite, "self_moa": self_moa, "du_debate": du_debate,
    "BTS_top_score": bts_top_score_adapter,
    "alpha_meu": alpha_meu_adapter, "garicano": garicano_adapter,
    "BTS+alpha_meu": bts_alpha_meu_hybrid_adapter, "full_stack": full_stack_aggregate,
}
BASELINES = {"majority_vote", "weighted_confidence", "self_consistency", "vanilla_debate",
             "moa_lite", "self_moa", "du_debate"}


def bootstrap_pvalue(a: list[int], b: list[int], n_iter: int = 2000, seed: int = 7) -> float:
    """One-sided P(mean(A_resampled) <= mean(B_resampled))."""
    rng = np.random.default_rng(seed)
    if not a or not b:
        return float("nan")
    A, B = np.asarray(a, float), np.asarray(b, float)
    wins = sum(A[rng.integers(0, len(A), len(A))].mean() - B[rng.integers(0, len(B), len(B))].mean() <= 0
               for _ in range(n_iter))
    return wins / n_iter


def main() -> dict:
    out: dict = {}
    for regime_name, regime in REGIMES.items():
        qs = generate_question_set(regime)
        results, per_q = {}, {}
        for name, agg in AGGREGATORS.items():
            results[name] = run_benchmark(agg, qs, n_bootstrap=1000)
            per_q[name] = [1 if agg(f"q{i}", dos).answer == truth else 0
                           for i, (truth, dos) in enumerate(qs) if not agg(f"q{i}", dos).abstain]
        winner = max(results, key=lambda k: results[k]["accuracy"])
        best_base = max(BASELINES, key=lambda k: results[k]["accuracy"])
        results["__meta__"] = {
            "winner": winner, "best_baseline": best_base,
            "winner_minus_best_baseline": results[winner]["accuracy"] - results[best_base]["accuracy"],
            "bootstrap_pvalue_one_sided": bootstrap_pvalue(per_q[winner], per_q[best_base]),
        }
        out[regime_name] = results
    return out


if __name__ == "__main__":
    results = main()
    for regime, res in results.items():
        meta = res["__meta__"]
        print(f"\n=== Regime: {regime} ===")
        print(f"{'aggregator':<22} {'acc':>7} {'ci_lo':>7} {'ci_hi':>7} {'abst%':>7} {'cap':>7} {'ece':>7} {'ms':>8}")
        print("-" * 80)
        for name in AGGREGATORS:
            r = res[name]
            cap = f"{r['calibrated_abstention_precision']:.3f}" if r['calibrated_abstention_precision'] == r['calibrated_abstention_precision'] else "  n/a"
            print(f"{name:<22} {r['accuracy']:>7.3f} {r['ci_low']:>7.3f} {r['ci_high']:>7.3f} "
                  f"{r['abstention_rate']*100:>6.1f}% {cap:>7} {r['ece']:>7.3f} {r['runtime_ms']:>8.1f}")
        print(f"WINNER: {meta['winner']} ({meta['winner_minus_best_baseline']:+.3f} vs "
              f"baseline {meta['best_baseline']}, bootstrap p={meta['bootstrap_pvalue_one_sided']:.3f})")
    with open("experiments/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nWrote experiments/benchmark_results.json")
