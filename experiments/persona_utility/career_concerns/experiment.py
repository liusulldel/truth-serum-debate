"""Experiment: 200 questions x 4 personas. Measure hedging, accuracy, ECE."""
from __future__ import annotations
import json
import random
from pathlib import Path

from mock_persona_debater import (
    PersonaType,
    Question,
    make_questions,
    persona_debater,
)


def ece(probs: list[float], labels: list[int], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if not probs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    n = len(probs)
    e = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        avg_y = sum(y for _, y in b) / len(b)
        e += (len(b) / n) * abs(avg_p - avg_y)
    return e


def run_persona(persona: PersonaType, questions: list[Question], seed: int = 42) -> dict:
    rng = random.Random(seed + hash(persona) % 10_000)
    history = {"answered": 0, "total": len(questions)}
    n_hedge = 0
    correct = 0
    committed_probs: list[float] = []
    committed_labels: list[int] = []
    early_effort: list[float] = []
    late_effort: list[float] = []

    for i, q in enumerate(questions):
        out = persona_debater(q, persona, history, rng)
        history["answered"] = i + 1
        # hedging: |p - 0.5| < 0.1 OR not committed
        is_hedge = (abs(out["p"] - 0.5) < 0.1) or (not out["committed"])
        if is_hedge:
            n_hedge += 1
        else:
            committed_probs.append(out["p"])
            committed_labels.append(q.true_label)
            pred = 1 if out["p"] >= 0.5 else 0
            if pred == q.true_label:
                correct += 1
        if i < len(questions) // 3:
            early_effort.append(out["effort"])
        elif i >= 2 * len(questions) // 3:
            late_effort.append(out["effort"])

    n = len(questions)
    n_committed = n - n_hedge
    acc = (correct / n_committed) if n_committed else 0.0
    return {
        "persona": persona,
        "n": n,
        "hedging_rate": n_hedge / n,
        "n_committed": n_committed,
        "accuracy_on_committed": acc,
        "ece_on_committed": ece(committed_probs, committed_labels),
        "early_mean_effort": sum(early_effort) / len(early_effort) if early_effort else 0,
        "late_mean_effort": sum(late_effort) / len(late_effort) if late_effort else 0,
    }


def main() -> dict:
    questions = make_questions(n=200, seed=42)
    personas: list[PersonaType] = [
        "persistent_identity_high_rep",
        "persistent_identity_low_rep",
        "anonymous_one_shot",
        "stateless",
    ]
    results = {p: run_persona(p, questions) for p in personas}
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    r = main()
    for p, d in r.items():
        print(
            f"{p:38s} hedge={d['hedging_rate']:.3f}  "
            f"acc={d['accuracy_on_committed']:.3f}  "
            f"ece={d['ece_on_committed']:.3f}  "
            f"effort_early={d['early_mean_effort']:.3f}  "
            f"effort_late={d['late_mean_effort']:.3f}"
        )
