"""Mixture-of-Agents (lite): 2-layer aggregator -> consensus picker.

Reference:
    Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024).
    "Mixture-of-Agents Enhances Large Language Model Capabilities."
    arXiv:2406.04692.

Layer 1: each debater is a proposer reporting (answer, p_true).
Layer 2: aggregator picks the inverse-variance weighted mean of p_true so
proposers near 0.5 are downweighted (matching the paper's observation that
aggregators learn to prefer confident proposers). Lite = single agg layer.
"""
from __future__ import annotations
from typing import Sequence
from . import DebaterOutput, Decision


def aggregate(question: str, debater_outputs: Sequence[DebaterOutput]) -> Decision:
    """Aggregate debater outputs via inverse-variance weighted p_true.

    Args:
        question: The question being decided.
        debater_outputs: One or more debater outputs containing p_true.

    Returns:
        A Decision with the inverse-variance weighted aggregated probability.
    """
    if not debater_outputs:
        raise ValueError("Need >=1 debater output.")
    weights, ps = [], []
    for d in debater_outputs:
        var = max(d.p_true * (1.0 - d.p_true), 1e-3)
        weights.append(1.0 / var)
        ps.append(d.p_true)
    p_agg = sum(w * p for w, p in zip(weights, ps)) / sum(weights)
    return Decision(question, 1 if p_agg >= 0.5 else 0, p_agg, False, "moa_lite")
