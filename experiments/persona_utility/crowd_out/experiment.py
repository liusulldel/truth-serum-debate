"""Run the persona-prompted crowd-out experiment.

200 questions x 4 personas. Reports the Gneezy-Rustichini contrast:
    effort(low_extrinsic) < effort(intrinsic)  ?
and the secondary contrasts vs stateless and high_extrinsic.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from mock_persona_debater import persona_debater

PERSONAS = ("intrinsic", "high_extrinsic", "low_extrinsic", "stateless")
N_QUESTIONS = 200
SEED = 42


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def _sd(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def run() -> dict:
    questions = [f"Reasoning question #{i:03d}" for i in range(N_QUESTIONS)]
    by_persona: dict[str, list[dict]] = {p: [] for p in PERSONAS}

    for qid, q in enumerate(questions):
        for p in PERSONAS:
            by_persona[p].append(persona_debater(q, p, seed=SEED, question_id=qid))  # type: ignore[arg-type]

    summary = {}
    for p, rows in by_persona.items():
        eff = [r["effort"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        verb = [r["verbosity"] for r in rows]
        hedge = [r["hedge_count"] for r in rows]
        summary[p] = {
            "effort_mean": round(_mean(eff), 4),
            "effort_sd": round(_sd(eff), 4),
            "accuracy_mean": round(_mean(acc), 4),
            "verbosity_mean": round(_mean(verb), 2),
            "hedge_mean": round(_mean(hedge), 3),
            "n": len(rows),
        }

    # Headline contrasts (signed: positive = crowd-out replicated).
    contrasts = {
        "intrinsic_minus_low_extrinsic_effort": round(
            summary["intrinsic"]["effort_mean"] - summary["low_extrinsic"]["effort_mean"], 4
        ),
        "stateless_minus_low_extrinsic_effort": round(
            summary["stateless"]["effort_mean"] - summary["low_extrinsic"]["effort_mean"], 4
        ),
        "high_extrinsic_minus_intrinsic_effort": round(
            summary["high_extrinsic"]["effort_mean"] - summary["intrinsic"]["effort_mean"], 4
        ),
        "intrinsic_minus_low_extrinsic_accuracy": round(
            summary["intrinsic"]["accuracy_mean"] - summary["low_extrinsic"]["accuracy_mean"], 4
        ),
    }

    out = {
        "n_questions": N_QUESTIONS,
        "seed": SEED,
        "by_persona": summary,
        "contrasts": contrasts,
        "crowd_out_replicated": contrasts["intrinsic_minus_low_extrinsic_effort"] > 0,
    }

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
