# Beyond BTS — Wave 5 experiment synthesis

**Date:** 2026-04-25
**Method:** six parallel mechanism-design / decision-theory / management-theory candidates implemented as standalone LangGraph-compatible aggregators in `experiments/<family>/`. Each has its own pytest battery, its own head-to-head against the BTS baseline in `src/bts.py`, and an honest REPORT.md. Total: 38/38 tests pass + 1 demo experiment.

## Scoreboard

| # | Family | Candidate (citation) | Beat BTS? | Tests | Honest finding |
|---|---|---|---|---|---|
| 1 | Mechanism design | **AGV** (d'Aspremont & Gérard-Varet 1979) | **No (complement)** | 6/6 | Budget-balanced Bayes-Nash truthful aggregation; scores collective decisions, not agent reliability — answers a different question than BTS. Use together, not against. |
| 2 | Contract theory | **Holmström team** (1982) | **No (complement)** | 10/10 | Reproduces 1/n free-rider inefficiency parametrically (n=2,3,5,10); judge-as-budget-breaker restores first-best effort. *Effort-elicitation* contract, not an aggregator. |
| 3 | Social choice | **Condorcet jury theorem** (1785; Boland 1989; Ladha 1992) | **No (instructive null)** | 7/7 | Independence assumption violated by shared training + prompts. With ρ = 0.3, accuracy plateaus at 0.74 regardless of N. Real value: a *correlated-CJT ceiling* against which any richer mechanism (BTS, Surprisingly Popular, debate) must beat to claim it extracts information beyond vote counts. |
| 4 | Ambiguity | **α-MEU** (Ghirardato-Maccheroni-Marinacci 2004) | **Yes** | 9/9 | When debaters split 0.9/0.1, BTS averages to 0.5 (tie-break determines answer); α-MEU sees ambiguity 0.80 and **abstains**. Direct AI-safety property: OOD inputs → debater disagreement → refuse to act. |
| 5 | Org economics | **Garicano** (2000) knowledge hierarchy | **Yes (toy)** | 6/6 | Worker-confidence threshold τ with manager-escalation cost h reproduces Garicano's interior-optimum Prop. 2. Falsifiable comparative static: optimal τ* decreases as h falls. Composable with BTS. |
| 6 | **Negative results** | Loss aversion, reciprocity, intrinsic-motivation crowd-out | **N/A — fails by construction** | 1 demo | Stateless mock LLM t = +0.57 (null) vs mock human λ=2.25 t = +9.40 (clean kink). Mechanisms requiring felt hedonic loss / persistent counterparty memory / pre-existing intrinsic motivation cannot transfer to LLMs by current architecture. |

## The diagnostic

The single most quotable sentence to come out of this wave (Agent 6):

> **"Mechanism design transfers to LLM agents to the extent that the mechanism's incentive-compatibility proof relies only on properties of the agent's posterior, not on properties of the agent's hedonic or relational architecture."**

This sentence simultaneously explains:
- Why BTS, AGV, α-MEU, Garicano, and Condorcet (modulo independence) **transfer** — they touch only posterior probabilities or aggregation rules.
- Why prospect theory, gift exchange, intrinsic-motivation crowd-out **do not transfer** — they need hedonic or relational structure that stateless LLMs lack.
- Gives the committee a falsifiable diagnostic to evaluate any new proposed mechanism *before* running it.

## What the wave produced overall

- **2 wins over the BTS baseline** (α-MEU, Garicano) — both for properties BTS does not target, not because BTS is wrong.
- **3 honest nulls / complements** (AGV, Holmström team, Condorcet) — each illuminates a different dimension BTS does not cover. AGV adds budget balance; Holmström adds an effort-elicitation contract; Condorcet supplies a correlated-jury *ceiling* that any aggregator must beat to claim non-trivial information extraction.
- **3 ruled-out failure cases** (loss aversion, reciprocity, intrinsic motivation) — with a working empirical demo confirming the predicted null on a stateless model.

## The composition that emerges

```
                    +---------------------------------+
                    |  Garicano routing (orgecon_5)   |
                    |  τ* threshold, escalate to      |
                    |  judge with cost h              |
                    +-------------------+-------------+
                                        |
                            +-----------+-----------+
                            |                       |
                +-----------v---------+   +---------v---------+
                |  α-MEU judge (4)    |   |  Worker layer:    |
                |  abstain if A > τ   |   |  BTS scoring (src/|
                |  ambiguity-aware    |   |  bts.py) on each  |
                +-----------+---------+   |  debater report   |
                            |             +---------+---------+
                            |                       |
                +-----------v-----------+           |
                |  Holmström forcing    |           |
                |  contract (2)         |           |
                |  judge as budget-     |           |
                |  breaker on success   |           |
                +-----------------------+           |
                                                    |
                +-----------------------+           |
                |  AGV transfer rule (1)|<----------+
                |  budget-balanced      |
                |  scoring of group     |
                |  decision             |
                +-----------------------+
```

Each layer is independently testable. The repo currently ships only the BTS scoring node (the `src/` baseline). The five experiment subdirs are stand-alone proof-of-concepts that can be composed in. **Headline claim for the application:** none of the five candidates is a *replacement* for BTS — but four of them give the operator an axiomatically grounded knob the current judge node lacks, and the fifth (Condorcet) provides the right baseline ceiling against which to measure the rest.

## Files

```
experiments/
├── SUMMARY.md                                  ← this file
├── mechdesign_agv/
│   ├── agv.py                       (~140 LOC)
│   ├── test_agv.py                  (6/6 PASS)
│   ├── compare_bts.py
│   └── REPORT.md
├── contract_holmstrom_teams/
│   ├── team_contract.py             (~140 LOC)
│   ├── test_team_contract.py        (10/10 PASS)
│   ├── compare_to_bts.py
│   ├── pytest_output.txt
│   ├── comparison_output.txt
│   └── REPORT.md
├── socialchoice_cjt/
│   ├── cjt.py                       (~120 LOC)
│   ├── test_cjt.py                  (7/7 PASS)
│   ├── compare_vs_bts.py
│   └── REPORT.md
├── ambig_alpha_meu/
│   ├── alpha_meu.py                 (~130 LOC)
│   ├── test_alpha_meu.py            (9/9 PASS)
│   ├── compare_vs_bts.py
│   └── REPORT.md
├── orgecon_garicano/
│   ├── garicano.py                  (~120 LOC)
│   ├── test_garicano.py             (6/6 PASS)
│   └── REPORT.md
├── negative_results/
│   └── THREE_FAILURES.md            (~1750 words, 15 citations)
└── negative_loss_aversion/
    ├── demo_loss_aversion.py        (~50ms run, t-stat null confirmed)
    └── README.md
```

**Aggregate:** 38/38 unit tests pass (`pytest experiments/ -q` → `38 passed in 0.24s`); 1 demo experiment confirms the predicted null. No edits outside `experiments/`. No live API calls. All citations verified against canonical sources (JSTOR / DOI / arXiv).

## What this wave changes for the application

1. **SAFETY essay §3** — the α-MEU result gives a concrete second pitch beyond BTS: ambiguity-aware judge with abstention as an axiomatic safety property.
2. **SAFETY essay §3 closing line** — the Agent-6 diagnostic sentence (above) is the load-bearing claim; cite `experiments/negative_results/THREE_FAILURES.md` as the formal argument.
3. **ECON essay §2(a)** — the Garicano + Holmström + AGV stack is a working demonstration of "production engineering as applied mechanism design" with five canonical theorems instantiated as code.
4. **README** — add a `## Experimental extensions` section linking to the five sibling mechanisms with a one-line summary each.
