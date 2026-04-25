"""Self-MoA: single-model self-ensembling instead of cross-model mixing.

Reference:
    Li, W., Lin, M., Zhong, S., Yan, S., & Yang, X. (2025).
    "Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models
    Beneficial?" arXiv:2502.00674 (ICLR 2025).

Empirical claim of the paper: on quality-sensitive tasks, sampling N completions
from the SINGLE strongest model and ensembling them outperforms the standard
MoA recipe of mixing heterogeneous models. The diversity of MoA proposers
introduces low-quality samples that the aggregator cannot filter out.

Mock implementation: collapse the diversity across debaters by treating them
as N samples from one source. We pick the highest-confidence proposer as the
"best model" anchor, shrink every other p_true toward that anchor's p_true
(the self-ensembling assumption: same generator -> correlated outputs), then
do confidence-weighted majority. This deliberately removes the cross-model
variance that moa_lite exploits.
"""
from __future__ import annotations
from typing import Sequence
from . import DebaterOutput, Decision

SHRINK = 0.5  # pull other proposers halfway to the anchor's p_true


def aggregate(question: str, debater_outputs: Sequence[DebaterOutput]) -> Decision:
    """Self-MoA aggregator: shrink toward the most confident anchor, then weight.

    Args:
        question: The question being decided.
        debater_outputs: One or more debater outputs with confidence scores.

    Returns:
        A Decision aggregated by self-MoA confidence-weighted shrinkage.
    """
    if not debater_outputs:
        raise ValueError("Need >=1 debater output.")
    anchor = max(debater_outputs, key=lambda d: d.confidence)
    p_anchor = anchor.p_true
    weights, ps = [], []
    for d in debater_outputs:
        # Self-ensembling: shrink each p_true toward the strongest model's p_true
        p_eff = (1.0 - SHRINK) * d.p_true + SHRINK * p_anchor
        w = max(d.confidence, 1e-3)
        weights.append(w)
        ps.append(p_eff)
    p_agg = sum(w * p for w, p in zip(weights, ps)) / sum(weights)
    return Decision(question, 1 if p_agg >= 0.5 else 0, p_agg, False, "self_moa")
