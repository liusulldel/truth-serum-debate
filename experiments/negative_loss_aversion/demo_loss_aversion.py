"""Negative-result demo: LLM debaters do NOT exhibit human-style loss aversion.

Hypothesis (human-transfer):
    Loss-framed prompts ("you LOSE 1 point per wrong answer") should make a
    debater more conservative / more hedged than mathematically equivalent
    gain-framed prompts ("you GAIN 1 point per right answer"), with effect
    size consistent with Kahneman-Tversky's lambda ~ 2.

Null (LLM-stateless):
    Because the agent has no realized consumption, no reference-point
    endowment, and no cross-call memory, it should treat the two frames as
    near-equivalent surface paraphrases. Any observed difference should be
    small, sign-unstable across paraphrases, and indistinguishable from prompt
    noise.

This file is a SIMULATION (no API calls) demonstrating the experimental
design and the expected null result. The 'mock_debater' models a stateless
LLM as: posterior_confidence = base_skill + epsilon, where epsilon is i.i.d.
Gaussian noise that does NOT depend on frame. We then show that even with
50 questions x 3 paraphrases x 2 frames, the gain-vs-loss difference is
within noise, while a hypothetical human (lambda=2.25 on hedge probability)
would show a clean effect.

Run: python demo_loss_aversion.py
Python 3.12.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Literal

random.seed(20260425)

Frame = Literal["gain", "loss"]

GAIN_PARAPHRASES = [
    "You will EARN +1 point for each correct answer.",
    "Bonus: +1 per accurate response.",
    "Reward structure: +1 credit for every right answer.",
]
LOSS_PARAPHRASES = [
    "You will LOSE 1 point for each wrong answer.",
    "Penalty: -1 per inaccurate response.",
    "Deduction structure: -1 credit for every wrong answer.",
]


@dataclass
class DebateItem:
    qid: int
    base_difficulty: float  # 0..1, higher = harder


def make_calibration_set(n: int = 50) -> list[DebateItem]:
    return [DebateItem(qid=i, base_difficulty=random.random()) for i in range(n)]


def mock_llm_debater(item: DebateItem, frame: Frame, paraphrase_idx: int) -> float:
    """Return a 'hedge score' in [0, 1]; higher = more conservative.

    Stateless LLM model: hedge depends on item difficulty + small prompt noise
    that depends on the *surface form* of the prompt, NOT on whether the
    frame is gain or loss in any economically meaningful way.
    """
    base = 0.3 + 0.5 * item.base_difficulty
    # Surface-form noise: each unique prompt string contributes a tiny offset.
    surface_seed = hash((frame, paraphrase_idx)) % 10_000 / 10_000
    surface_noise = (surface_seed - 0.5) * 0.04  # +/- 2 percentage points
    sampling_noise = random.gauss(0, 0.03)
    return max(0.0, min(1.0, base + surface_noise + sampling_noise))


def mock_human_debater(item: DebateItem, frame: Frame, paraphrase_idx: int) -> float:
    """Human with lambda=2.25 loss aversion: loss frame raises hedge by ~15pp."""
    base = 0.3 + 0.5 * item.base_difficulty
    frame_effect = 0.15 if frame == "loss" else 0.0  # the kink
    sampling_noise = random.gauss(0, 0.03)
    return max(0.0, min(1.0, base + frame_effect + sampling_noise))


def run_experiment(debater_fn, items: list[DebateItem]) -> dict[Frame, list[float]]:
    out: dict[Frame, list[float]] = {"gain": [], "loss": []}
    for item in items:
        for p_idx in range(3):
            out["gain"].append(debater_fn(item, "gain", p_idx))
            out["loss"].append(debater_fn(item, "loss", p_idx))
    return out


def report(label: str, results: dict[Frame, list[float]]) -> None:
    g_mean = statistics.mean(results["gain"])
    l_mean = statistics.mean(results["loss"])
    g_sd = statistics.stdev(results["gain"])
    l_sd = statistics.stdev(results["loss"])
    diff = l_mean - g_mean
    # Welch-style pooled SE
    n = len(results["gain"])
    se = (g_sd**2 / n + l_sd**2 / n) ** 0.5
    t_like = diff / se if se > 0 else float("inf")
    print(f"\n=== {label} ===")
    print(f"  gain frame: mean hedge = {g_mean:.3f}  sd = {g_sd:.3f}  n = {n}")
    print(f"  loss frame: mean hedge = {l_mean:.3f}  sd = {l_sd:.3f}  n = {n}")
    print(f"  loss - gain = {diff:+.3f}  (approx t = {t_like:+.2f})")
    if abs(t_like) < 2.0:
        print("  -> verdict: NO detectable loss-aversion effect (null holds)")
    else:
        print("  -> verdict: significant frame effect (human-style kink present)")


def main() -> None:
    items = make_calibration_set(50)
    print("Loss-aversion transfer test")
    print("Design: 50 questions x 3 paraphrases x 2 frames = 300 obs per frame")
    llm_results = run_experiment(mock_llm_debater, items)
    human_results = run_experiment(mock_human_debater, items)
    report("Mock stateless LLM debater", llm_results)
    report("Mock human debater (lambda=2.25)", human_results)
    print(
        "\nInterpretation: the LLM null is what a real Claude/GPT debater is\n"
        "predicted to produce, because the loss-frame prompt is text without\n"
        "a hedonic referent. To run for real, replace mock_llm_debater with\n"
        "an Anthropic API call and compare against this baseline."
    )


if __name__ == "__main__":
    main()
