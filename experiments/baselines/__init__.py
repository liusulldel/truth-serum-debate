"""Drop-in aggregator baselines for the head-to-head benchmark.

All baselines expose ``aggregate(question, debater_outputs) -> Decision`` so the
benchmark harness can treat them interchangeably.

DebaterOutput fields:
  answer:             hard pick in {0, 1}
  p_true:             calibrated P(answer=1)
  confidence:         self-reported confidence in [0, 1]
  samples:            optional repeated draws of `answer` (for self_consistency)
  rebuttal_strength:  scalar in [0, 1] (for vanilla_debate)
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DebaterOutput:
    answer: int
    p_true: float
    confidence: float = 1.0
    samples: tuple[int, ...] = field(default_factory=tuple)
    rebuttal_strength: float = 0.5


@dataclass(frozen=True)
class Decision:
    """Uniform aggregator output used by the benchmark harness."""
    question: str
    answer: int           # 0 or 1; ignored when abstain=True
    p_true: float         # calibrated probability of TRUE in [0, 1]
    abstain: bool
    aggregator: str
