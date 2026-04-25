"""Full mechanism stack: Garicano (2000) routing -> alpha-MEU judging.
Pipeline: (1) if max worker confidence >= tau_route -> commit to that answer
(no judge); (2) else escalate to Ghirardato-Maccheroni-Marinacci 2004 alpha-MEU
on BTS-elicited (Prelec 2004) p_true vectors, abstain if ambiguity > tau_ambig.
Composes the four mechanisms already in `experiments/`; no copy-paste.
"""
from __future__ import annotations
from typing import Sequence
from experiments.ambig_alpha_meu.alpha_meu import alpha_meu_aggregate
from experiments.baselines import DebaterOutput, Decision


def full_stack_aggregate(question: str, debater_outputs: Sequence[DebaterOutput], *,
                         alpha: float = 1.0, tau_ambig: float = 0.4,
                         tau_route: float = 0.6, h: float = 0.15) -> Decision:
    if not debater_outputs:
        raise ValueError("Need >=1 debater output.")
    # Step 1: Garicano routing -- shortcut on confident worker.
    best = max(debater_outputs, key=lambda d: d.confidence)
    if best.confidence >= tau_route:
        return Decision(question, best.answer, best.p_true, False, "full_stack")
    # Step 2: alpha-MEU judge with BTS-elicited probabilities.
    dists = [[1.0 - d.p_true, d.p_true] for d in debater_outputs]
    a = alpha_meu_aggregate(question, dists, alpha=alpha, tau=tau_ambig)
    return Decision(question, 1 if a.decision == "TRUE" else 0,
                    a.p_alpha, a.decision == "ABSTAIN", "full_stack")
