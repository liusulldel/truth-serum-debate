"""Head-to-head benchmark harness. ``run_benchmark`` scores any
``(question, debater_outputs) -> Decision`` aggregator on the three regimes.
RNG: ``numpy.random.default_rng(seed=42)``.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import numpy as np
from experiments.ambig_alpha_meu.alpha_meu import alpha_meu_aggregate
from experiments.baselines import DebaterOutput, Decision


@dataclass(frozen=True)
class Regime:
    name: str
    p_correct: float
    rho: float
    sigma_conf: float
    n_questions: int = 200
    n_debaters: int = 3
    k_samples: int = 5


REGIMES = {
    "easy":   Regime("easy",   p_correct=0.85, rho=0.10, sigma_conf=0.10),
    "medium": Regime("medium", p_correct=0.70, rho=0.30, sigma_conf=0.20),
    "hard":   Regime("hard",   p_correct=0.55, rho=0.50, sigma_conf=0.30),
}


def _correlated_bernoulli(rng: np.random.Generator, n: int, p: float, rho: float) -> np.ndarray:
    """Bernoulli(p) with pairwise corr `rho` via Gaussian copula on shared latent."""
    from scipy.stats import norm
    z = np.sqrt(max(rho, 0.0)) * rng.standard_normal() + np.sqrt(max(1 - rho, 0.0)) * rng.standard_normal(n)
    return (z < norm.ppf(p)).astype(int)


def generate_question(rng: np.random.Generator, regime: Regime) -> tuple[int, list[DebaterOutput]]:
    truth = int(rng.integers(0, 2))
    correct_mask = _correlated_bernoulli(rng, regime.n_debaters, regime.p_correct, regime.rho)
    outs: list[DebaterOutput] = []
    for is_correct in correct_mask:
        ans = truth if is_correct else 1 - truth
        base = 0.5 + 0.4 * (1 if ans == 1 else -1)
        p_true = float(np.clip(base + rng.normal(0, regime.sigma_conf), 0.01, 0.99))
        confidence = float(np.clip(2 * abs(p_true - 0.5) + rng.normal(0, regime.sigma_conf), 0.0, 1.0))
        samples = tuple(int(rng.random() < p_true) for _ in range(regime.k_samples))
        rebuttal = float(np.clip(2 * abs(p_true - 0.5) + rng.normal(0, regime.sigma_conf * 0.5), 0.0, 1.0))
        outs.append(DebaterOutput(ans, p_true, confidence, samples, rebuttal))
    return truth, outs


def generate_question_set(regime: Regime, seed: int = 42) -> list[tuple[int, list[DebaterOutput]]]:
    rng = np.random.default_rng(seed)
    return [generate_question(rng, regime) for _ in range(regime.n_questions)]


def _ece(predictions: Sequence[tuple[float, int]], n_bins: int = 10) -> float:
    if not predictions:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(predictions)
    ece = 0.0
    for b in range(n_bins):
        in_bin = [(p, y) for p, y in predictions
                  if bins[b] <= p < bins[b + 1] or (b == n_bins - 1 and p == 1.0)]
        if not in_bin:
            continue
        ap = sum(p for p, _ in in_bin) / len(in_bin)
        ay = sum(y for _, y in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(ap - ay)
    return ece


def _bootstrap_ci(values: list[int], n_bootstrap: int, rng: np.random.Generator,
                  alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, float)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_bootstrap)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


Aggregator = Callable[[str, Sequence[DebaterOutput]], Decision]


def run_benchmark(aggregator: Aggregator,
                  question_set: Iterable[tuple[int, list[DebaterOutput]]],
                  n_bootstrap: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    correct, abstains, abstain_correctness = [], [], []
    pred_for_ece: list[tuple[float, int]] = []
    qs = list(question_set)
    t0 = time.perf_counter()
    for i, (truth, outs) in enumerate(qs):
        d = aggregator(f"q{i}", outs)
        if d.abstain:
            abstains.append(1)
            hard = 1 if d.p_true >= 0.5 else 0
            abstain_correctness.append(1 if hard != truth else 0)
        else:
            abstains.append(0)
            correct.append(1 if d.answer == truth else 0)
            pred_for_ece.append((float(d.p_true), int(truth)))
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    accuracy = (sum(correct) / len(correct)) if correct else 0.0
    ci_lo, ci_hi = _bootstrap_ci(correct, n_bootstrap, rng) if correct else (0.0, 0.0)
    n_abstain = sum(abstains)
    cap = (sum(abstain_correctness) / n_abstain) if n_abstain else float("nan")
    return {"accuracy": accuracy, "ci_low": ci_lo, "ci_high": ci_hi,
            "calibrated_abstention_precision": cap,
            "abstention_rate": n_abstain / len(qs) if qs else 0.0,
            "ece": _ece(pred_for_ece), "runtime_ms": runtime_ms,
            "n_decisions": len(qs), "n_abstain": n_abstain}


def alpha_meu_adapter(question: str, outs: Sequence[DebaterOutput]) -> Decision:
    a = alpha_meu_aggregate(question, [[1 - d.p_true, d.p_true] for d in outs], alpha=1.0, tau=0.4)
    return Decision(question, 1 if a.decision == "TRUE" else 0, a.p_alpha,
                    a.decision == "ABSTAIN", "alpha_meu")


def bts_top_score_adapter(question: str, outs: Sequence[DebaterOutput]) -> Decision:
    best = max(outs, key=lambda d: abs(d.p_true - 0.5))
    return Decision(question, best.answer, best.p_true, False, "bts_top_score")


def garicano_adapter(question: str, outs: Sequence[DebaterOutput], tau_route: float = 0.6) -> Decision:
    best = max(outs, key=lambda d: d.confidence)
    if best.confidence >= tau_route:
        return Decision(question, best.answer, best.p_true, False, "garicano")
    pos = sum(d.confidence * (1 if d.answer == 1 else -1) for d in outs)
    p = max(0.0, min(1.0, 0.5 + 0.5 * (pos / max(sum(d.confidence for d in outs), 1e-9))))
    return Decision(question, 1 if pos >= 0 else 0, p, False, "garicano")


def bts_alpha_meu_hybrid_adapter(question: str, outs: Sequence[DebaterOutput]) -> Decision:
    a = alpha_meu_aggregate(question, [[1 - d.p_true, d.p_true] for d in outs], alpha=0.7, tau=0.45)
    return Decision(question, 1 if a.decision == "TRUE" else 0, a.p_alpha,
                    a.decision == "ABSTAIN", "bts+alpha_meu")
