"""LangGraph-based 2-debater + 1-judge orchestration with BTS scoring.

Pipeline:
    [Debater A] --\\
                   \\__ [Judge] -> BTS score + winner
    [Debater B] --/

Each debater is asked to provide:
  1. ``answer``: their own true/false call on the question.
  2. ``opponent_prediction``: a probability in [0, 1] that the OTHER
     debater will answer "true". This is the BTS prediction term.

The Judge does no LLM-based winner picking by default -- the winner is
the debater whose BTS score is higher (with ties broken in favour of A).
The judge node is kept as a hook so a human-style adjudicator could be
added later.

CLI:
    python -m src.debate --question "Water boils at 100C at sea level." \\
        --debater-model claude-haiku-4-5 \\
        --judge-model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from anthropic import Anthropic
from langgraph.graph import END, StateGraph

from .bts import bts_scores

DEBATER_SYSTEM = """You are a debater in a binary-answer truth game. \
You will be given a factual statement. Reply ONLY with valid JSON, no \
prose, of the form:

  {"answer": true|false, "opponent_prediction": <float in [0,1]>, \
"reason": "<one short sentence>"}

- "answer" is YOUR best call on whether the statement is true.
- "opponent_prediction" is the probability you assign to the OTHER \
debater answering true. This is NOT your own answer -- it is your model \
of the opponent. Be honest; calibration is rewarded.
- "reason" is at most 20 words.
"""


class DebateState(TypedDict, total=False):
    """LangGraph state passed between nodes."""
    question: str
    debater_a: dict[str, Any]
    debater_b: dict[str, Any]
    judgement: dict[str, Any]
    debater_model: str
    judge_model: str


@dataclass
class DebateConfig:
    debater_model: str = field(
        default_factory=lambda: os.environ.get("DEBATER_MODEL", "claude-haiku-4-5")
    )
    judge_model: str = field(
        default_factory=lambda: os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
    )
    max_tokens: int = 256
    temperature: float = 0.7


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_debater_reply(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a debater reply."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in debater reply: {text!r}")
    return json.loads(m.group(0))


def _call_debater(
    client: Anthropic,
    model: str,
    question: str,
    persona: str,
    cfg: DebateConfig,
) -> dict[str, Any]:
    """Invoke a single debater turn and return parsed JSON."""
    msg = client.messages.create(
        model=model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=DEBATER_SYSTEM + f"\nYou are debater {persona}.",
        messages=[{"role": "user", "content": f"Statement: {question}"}],
    )
    raw = "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    )
    parsed = _parse_debater_reply(raw)
    parsed.setdefault("answer", False)
    parsed.setdefault("opponent_prediction", 0.5)
    parsed["_raw"] = raw
    parsed["_persona"] = persona
    return parsed


def make_graph(client: Anthropic, cfg: DebateConfig):
    """Build the LangGraph StateGraph for one debate round."""

    def node_debater_a(state: DebateState) -> DebateState:
        return {
            "debater_a": _call_debater(
                client, cfg.debater_model, state["question"], "A", cfg
            )
        }

    def node_debater_b(state: DebateState) -> DebateState:
        return {
            "debater_b": _call_debater(
                client, cfg.debater_model, state["question"], "B", cfg
            )
        }

    def node_judge(state: DebateState) -> DebateState:
        a = state["debater_a"]
        b = state["debater_b"]
        # Build a 2-respondent BTS round, options: 0 = false, 1 = true.
        own = [int(bool(a["answer"])), int(bool(b["answer"]))]
        # Convert opponent_prediction (P(other says true)) into a length-2
        # distribution [P(false), P(true)]. NB: each debater is predicting
        # the OTHER debater's answer; that is exactly the BTS prediction
        # over the rest-of-population (here, n_others=1).
        preds = [
            [1.0 - float(a["opponent_prediction"]), float(a["opponent_prediction"])],
            [1.0 - float(b["opponent_prediction"]), float(b["opponent_prediction"])],
        ]
        scores = bts_scores(own, preds, n_options=2)
        winner = "A" if scores[0] >= scores[1] else "B"
        return {
            "judgement": {
                "bts_a": scores[0],
                "bts_b": scores[1],
                "winner": winner,
                "answers": own,
                "preds": preds,
            }
        }

    g = StateGraph(DebateState)
    g.add_node("debater_a", node_debater_a)
    g.add_node("debater_b", node_debater_b)
    g.add_node("judge", node_judge)
    g.set_entry_point("debater_a")
    g.add_edge("debater_a", "debater_b")
    g.add_edge("debater_b", "judge")
    g.add_edge("judge", END)
    return g.compile()


def run_debate(
    question: str,
    cfg: Optional[DebateConfig] = None,
    client: Optional[Anthropic] = None,
) -> dict[str, Any]:
    """Run a single debate round and return final state.

    Args:
        question: A binary-answerable factual statement.
        cfg: DebateConfig; defaults pull from env.
        client: Pre-built Anthropic client; defaults to ``Anthropic()``.

    Returns:
        Final DebateState dict with keys ``debater_a``, ``debater_b``,
        ``judgement``.
    """
    cfg = cfg or DebateConfig()
    client = client or Anthropic()
    graph = make_graph(client, cfg)
    return graph.invoke({"question": question})


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run one BTS debate round.")
    p.add_argument("--question", required=True)
    p.add_argument("--debater-model", default=None)
    p.add_argument("--judge-model", default=None)
    args = p.parse_args()
    cfg = DebateConfig()
    if args.debater_model:
        cfg.debater_model = args.debater_model
    if args.judge_model:
        cfg.judge_model = args.judge_model
    out = run_debate(args.question, cfg=cfg)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _cli()
