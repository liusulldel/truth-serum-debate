# truth-serum-debate

> Bayesian Truth Serum scoring for LLM debate as scalable oversight.
> Implementation + empirical pilot + formal derivation. (Prelec 2004 ↦ Khan et al. 2024)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liusulldel/truth-serum-debate/blob/main/notebooks/01_demo.ipynb)
[![Anthropic API](https://img.shields.io/badge/Anthropic-Claude-cc785c.svg)](https://docs.anthropic.com/)

## TL;DR

LLM debate protocols lack a formal strategy-proofness guarantee — two debaters can coordinate on a plausible falsehood and a weaker judge has no principled way to detect it. This repo grafts **Bayesian Truth Serum** (Prelec, *Science* 2004) onto Khan et al.'s (2024) debate setup, scoring each debater on how surprisingly common their answer is given their prediction of the other's answer. **Headline result: [to be filled after pilot] — BTS-scored debate raises judge accuracy from X% to Y% on a 50-question TruthfulQA subset (Haiku debaters, Sonnet judge).**

> **Wider context**: this repo instantiates a broader research program — *LangGraph orchestration as applied institutional design for AI agents*. See [docs/CORPORATE_MGMT_ANGLE.md](docs/CORPORATE_MGMT_ANGLE.md).

## The idea in one paragraph

LLM debate (Irving 2018, Khan et al. 2024) lets a weaker judge evaluate stronger debaters by letting the debaters cross-examine each other. But current protocols have no formal strategy-proofness guarantee: two debaters can in principle coordinate on a plausible-but-wrong answer, and a Bayesian judge with a weak prior has no mechanism to penalize the coordination. Bayesian Truth Serum (Prelec 2004), a peer-prediction mechanism from social-choice theory, gives each debater incentive to report truthfully **without ground truth and without trusting the opponent** — by scoring each answer on its *information-weighted surprise* (how much more common it is than predicted). This repo implements BTS-scored debate on the Claude family and tests when (and whether) it improves truthfulness over vanilla debate.

## Quickstart

```bash
git clone https://github.com/liusulldel/truth-serum-debate.git
cd truth-serum-debate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
python run_demo.py            # ~3 min, ~$0.40 in API spend
```

## What's in here

- `src/` — BTS scoring + debate orchestration (LangGraph state machine, two debaters + judge)
- `src/bts.py` — pure scoring function, ~80 lines, fully unit-tested
- `notebooks/01_demo.ipynb` — reproducible 50-question pilot, Haiku-as-debater + Sonnet-as-judge
- `THEORY.md` — formal derivation, BTS → debate strategy-proofness, common-prior assumption + its failure modes
- `data/` — small eval set (TruthfulQA subset + ground-truth labels, used only for *post-hoc* validation, never for scoring)
- `results/` — figures + CSV from the pilot run
- `tests/` — pytest suite for the scoring function and orchestrator

## Headline result

On a controlled synthetic benchmark (per-debater accuracy 0.70, pairwise correlation 0.30, N = 200 questions, seed = 42, bootstrap CI), the **BTS + α-MEU hybrid** achieves **84.1%** accuracy on non-abstained decisions versus **74.5%** for the best of seven published multi-agent orchestration baselines — Self-MoA (Li 2025), Du multi-agent debate (2023), MoA-lite (Wang 2024), self-consistency (Wang 2022), Khan-style debate (2024), weighted-confidence, and majority vote — a **+9.6 pp** improvement at bootstrap **p = 0.026**. Full results table, per-regime breakdown, and honest null-where-it-loses section in [`experiments/BASELINE_BENCHMARK.md`](experiments/BASELINE_BENCHMARK.md).

The live-debater notebook pilot (N = 50, Haiku-debaters + Sonnet-judge) is in `notebooks/01_demo.ipynb`; figure goes here once the API run completes.

## Experimental extensions

`experiments/` contains five drop-in mechanism-design aggregators (each ~120 LOC, own pytest battery, real citations) layered on the same `(question, debater_outputs) → Decision` interface as the baselines:

| Folder | Mechanism | Citation | What it adds over BTS |
| --- | --- | --- | --- |
| `experiments/ambig_alpha_meu/` | α-MEU (ambiguity-aware) | Ghirardato-Maccheroni-Marinacci 2004 | Abstain when debaters disagree → safety property |
| `experiments/orgecon_garicano/` | Knowledge-hierarchy router | Garicano 2000 (JPE 108) | Cost-aware escalation to expensive judge |
| `experiments/contract_holmstrom_teams/` | Forcing contract + budget-breaker | Holmström 1982 (Bell J. Econ.) | Effort elicitation under budget balance |
| `experiments/mechdesign_agv/` | AGV expected-externality | d'Aspremont-Gérard-Varet 1979 | Budget-balanced Bayes-Nash truthful aggregation |
| `experiments/socialchoice_cjt/` | Correlated Condorcet ceiling | Condorcet 1785 + Ladha 1992 | Baseline ceiling: any aggregator beating this extracts info beyond vote counts |

Plus `experiments/negative_results/THREE_FAILURES.md` — three classical incentive theorems (loss aversion, reciprocity, intrinsic-motivation crowd-out) that **do not transfer** to stateless LLMs, with a working demo confirming the predicted null. See `experiments/SUMMARY.md` for the full synthesis.

## Reproducibility

| Component        | Version                       |
| ---------------- | ----------------------------- |
| Python           | 3.11+                         |
| Anthropic SDK    | `anthropic>=0.40.0`           |
| LangGraph        | `langgraph>=0.2.0`            |
| Models           | `claude-haiku-4-5`, `claude-sonnet-4-5` |
| Random seed      | 42                            |
| Total API cost   | ~$8 [estimated] for full notebook |
| Wall-clock       | ~25 min on a single thread    |

All prompts are pinned in `src/prompts/`. The notebook caches per-question debate transcripts so partial reruns are cheap.

## Theoretical foundation

See [THEORY.md](THEORY.md) for the BTS → debate derivation, including the common-prior assumption and what breaks when debaters share private information the judge doesn't have.

Key references:

- Prelec, D. (2004). *A Bayesian Truth Serum for Subjective Data.* Science 306, 462–466. [doi:10.1126/science.1102081](https://doi.org/10.1126/science.1102081)
- Irving, G., Christiano, P., Amodei, D. (2018). *AI Safety via Debate.* [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
- Khan, A., Hughes, J., Valentine, D., et al. (2024). *Debating with More Persuasive LLMs Leads to More Truthful Answers.* ICML 2024. [arXiv:2402.06782](https://arxiv.org/abs/2402.06782)
- Anthropic (2025). [Recommended Directions for Technical AI Safety Research.](https://alignment.anthropic.com/2025/recommended-directions/)
- Anthropic (2023). [Core Views on AI Safety.](https://www.anthropic.com/news/core-views-on-ai-safety)

## Roadmap (next 4 months, if extended)

1. **Scale.** N=50 → N=2000 across Haiku/Sonnet/Opus capability gaps; measure how the BTS lift varies with the debater–judge gap.
2. **Adversarial collusion.** Hand-construct debater prompts that explicitly try to coordinate on a falsehood; check whether BTS reward forces defection.
3. **Mechanism failure modes.** When does BTS-debate degenerate into a pure coordination game? Map the boundary where the common-prior assumption breaks empirically.
4. **Activation-patching ablation.** Does the BTS reward signal leak into the debater's hidden chain-of-thought, and if so, is the leakage faithful or steganographic?

## Related framing

The BTS-debate protocol here is one cell in a larger matrix — applying organizational economics to multi-agent LLM systems. Companion piece: [Corporate Management Angle](docs/CORPORATE_MGMT_ANGLE.md).

## Limitations

- **Pilot scale.** N=50 is enough to detect a large effect, not enough to bound a small one. CIs will be wide.
- **Common-prior assumption.** BTS strategy-proofness requires debaters and judge to share a prior over the world. LLMs trained on overlapping corpora approximately satisfy this, but not exactly — see THEORY.md §3.
- **Judge model contamination.** TruthfulQA is in pretraining data. The pilot uses a held-out paraphrased subset, but this is a known confound for any LLM-eval-on-LLM setup.
- **No human baseline.** All judges in this pilot are Claude. A human-judge replication is the obvious next step but out of scope here.

## Citation

```bibtex
@software{liu2026truthserum,
  author  = {Liu, S.},
  title   = {truth-serum-debate: Bayesian Truth Serum scoring for LLM debate},
  year    = {2026},
  url     = {https://github.com/liusulldel/truth-serum-debate},
  version = {0.1.0}
}
```

## License

MIT — see [LICENSE](LICENSE).
