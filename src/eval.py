"""Eval harness for BTS-debate.

Reads ``data/questions.jsonl``, runs ``N`` questions x ``K`` replicates,
writes ``results/raw.csv`` and ``results/summary.json``.

Each row of raw.csv:
    qid, replicate, question, gold, a_answer, b_answer,
    a_pred, b_pred, bts_a, bts_b, winner, winner_correct,
    majority_correct

CLI:
    python -m src.eval --n 20 --k 1 \\
        --debater-model claude-haiku-4-5 \\
        --judge-model claude-sonnet-4-5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd
from anthropic import Anthropic

from .debate import DebateConfig, run_debate


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL questions; each row must have id/question/answer/source."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _iter_jobs(
    questions: list[dict[str, Any]], n: int, k: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    for q in questions[:n]:
        for rep in range(k):
            yield rep, q


def run_eval(
    questions_path: str | Path,
    out_dir: str | Path,
    n: int = 20,
    k: int = 1,
    cfg: Optional[DebateConfig] = None,
    client: Optional[Anthropic] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the eval and write raw.csv + summary.json. Returns the DataFrame."""
    cfg = cfg or DebateConfig()
    client = client or Anthropic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path)
    if not questions:
        raise RuntimeError(f"No questions loaded from {questions_path}")

    rows: list[dict[str, Any]] = []
    for rep, q in _iter_jobs(questions, n, k):
        gold = bool(q["answer"])
        try:
            res = run_debate(q["question"], cfg=cfg, client=client)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[ERR] q={q['id']} rep={rep}: {exc}", file=sys.stderr)
            continue
        a, b, j = res["debater_a"], res["debater_b"], res["judgement"]
        a_ans = bool(a["answer"])
        b_ans = bool(b["answer"])
        winner_label = j["winner"]  # "A" or "B"
        winner_answer = a_ans if winner_label == "A" else b_ans
        majority_answer = a_ans if a_ans == b_ans else None
        row = {
            "qid": q["id"],
            "replicate": rep,
            "question": q["question"],
            "gold": gold,
            "source": q.get("source", ""),
            "a_answer": a_ans,
            "b_answer": b_ans,
            "a_pred": float(a["opponent_prediction"]),
            "b_pred": float(b["opponent_prediction"]),
            "bts_a": float(j["bts_a"]),
            "bts_b": float(j["bts_b"]),
            "winner": winner_label,
            "winner_correct": winner_answer == gold,
            "majority_correct": (majority_answer == gold)
            if majority_answer is not None
            else None,
        }
        rows.append(row)
        if verbose:
            print(
                f"  q={q['id']:<6} rep={rep} a={a_ans} b={b_ans}"
                f" winner={winner_label} -> correct={row['winner_correct']}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    raw_path = out_dir / "raw.csv"
    df.to_csv(raw_path, index=False)

    summary = _summarise(df)
    summary["wall_clock_s"] = time.time() - getattr(run_eval, "_t0", time.time())
    summary["debater_model"] = cfg.debater_model
    summary["judge_model"] = cfg.judge_model
    summary["n_questions"] = n
    summary["k_replicates"] = k
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\nWrote {raw_path} ({len(df)} rows)")
        print(json.dumps(summary, indent=2))
    return df


def _summarise(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n_rows": 0}
    return {
        "n_rows": int(len(df)),
        "winner_accuracy": float(df["winner_correct"].mean()),
        "majority_accuracy": float(
            df["majority_correct"].dropna().mean() if df["majority_correct"].notna().any() else 0.0
        ),
        "agreement_rate": float((df["a_answer"] == df["b_answer"]).mean()),
        "mean_bts_a": float(df["bts_a"].mean()),
        "mean_bts_b": float(df["bts_b"].mean()),
    }


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--questions", default="data/questions.jsonl")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--debater-model", default=None)
    p.add_argument("--judge-model", default=None)
    args = p.parse_args()
    cfg = DebateConfig()
    if args.debater_model:
        cfg.debater_model = args.debater_model
    if args.judge_model:
        cfg.judge_model = args.judge_model
    run_eval._t0 = time.time()  # type: ignore[attr-defined]
    run_eval(
        questions_path=args.questions,
        out_dir=args.out_dir,
        n=args.n,
        k=args.k,
        cfg=cfg,
    )


if __name__ == "__main__":
    _cli()
