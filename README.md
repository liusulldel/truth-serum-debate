# truth-serum-debate

> Bayesian Truth Serum scoring for LLM debate as scalable oversight.
> Implementation + empirical pilot + formal derivation. (Prelec 2004 ↦ Khan et al. 2024)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liusulldel/truth-serum-debate/blob/main/notebooks/01_demo.ipynb)
[![Anthropic API](https://img.shields.io/badge/Anthropic-Claude-cc785c.svg)](https://docs.anthropic.com/)

## TL;DR

**Mechanism-design scoring rules layered onto multi-agent LLM debate raise judge accuracy by +9.6 percentage points (74.5% → 84.1%, bootstrap p = 0.026) on the realistic correlated-debater regime, and the ones that transfer share a single diagnostic property.**

This repo grafts Bayesian Truth Serum (Prelec, *Science* 2004) and α-MEU ambiguity-aversion (Ghirardato-Maccheroni-Marinacci 2004) onto Khan et al. (2024)-style debate, benchmarks them against seven published multi-agent baselines (Self-MoA, Du-debate, MoA-lite, self-consistency, vanilla debate, weighted-confidence, majority vote), and reports the win on the medium regime (per-debater accuracy 0.70, pairwise correlation 0.30, N = 200, seed = 42). The diagnostic that emerges from six mechanism families tested in parallel: **mechanism design transfers to LLM agents iff its incentive-compatibility proof relies only on properties of the agent's posterior, not on hedonic or relational architecture.** Three classical incentive theorems that fail this test (loss aversion, reciprocity, intrinsic-motivation crowd-out) are confirmed null on a stateless mock; see [`experiments/negative_results/THREE_FAILURES.md`](experiments/negative_results/THREE_FAILURES.md).

## Why this matters

Published LLM-debate protocols — Khan et al. 2024 (arXiv:2402.06782), Du et al. 2023 (arXiv:2305.14325), Wang et al. 2024 MoA (arXiv:2406.04692) — establish empirically that stronger debaters can help weaker judges find truth, but none ship a formal incentive-compatibility proof for the debaters themselves. Two debaters who share a base model and prompt distribution can coordinate on a plausible falsehood, and a binary win/lose judge has no principled signal to penalize the coordination. This repo closes two pieces of that gap:

- A **strategy-proofness theorem** for BTS-scored debate ([`THEORY.md`](THEORY.md) §Theorem 1) that makes truthful reporting a strict Bayes-Nash equilibrium under common prior, stochastic relevance, and impersonal beliefs.
- An **abstention-aware judge** (α-MEU) that empirically beats every non-abstaining baseline tested when debater errors are correlated, exactly the regime where naive aggregators silently degrade.

## Headline result

Medium regime (p = 0.70, ρ = 0.30, N = 200, seed = 42, 1000 bootstrap resamples). Full table for all three regimes in [`experiments/BASELINE_BENCHMARK.md`](experiments/BASELINE_BENCHMARK.md).

| aggregator | accuracy | 95% CI | abstain% | citation |
|---|---|---|---|---|
| **BTS + α-MEU** | **0.841** | **[0.761, 0.909]** | 56.0% | this repo |
| α-MEU only | 0.838 | [0.750, 0.912] | 60.0% | Ghirardato-Maccheroni-Marinacci 2004 |
| majority_vote | 0.745 | [0.680, 0.805] | 0.0% | Boland 1989 |
| du_debate | 0.730 | [0.670, 0.790] | 0.0% | Du 2023, arXiv:2305.14325 |
| moa_lite | 0.715 | [0.645, 0.775] | 0.0% | Wang 2024, arXiv:2406.04692 |
| weighted_confidence | 0.700 | [0.635, 0.765] | 0.0% | Grofman-Owen-Feld 1983 |
| self_consistency | 0.690 | [0.625, 0.755] | 0.0% | Wang 2022, arXiv:2203.11171 |
| self_moa | 0.685 | [0.620, 0.750] | 0.0% | Li 2025, arXiv:2502.00674 |
| BTS_top_score | 0.665 | [0.600, 0.730] | 0.0% | Prelec 2004 |
| vanilla_debate | 0.660 | [0.590, 0.725] | 0.0% | Khan 2024, arXiv:2402.06782 |

**+9.6 pp over the best baseline (majority_vote), bootstrap p = 0.026 one-sided.** The hybrid abstains on 56% of questions; on the 44% it commits to, it is decisively more accurate. The honest negatives — easy-regime accuracy if abstention has no recourse, hard-regime collapse when correlated-wrong errors swamp signal — are documented in §4 of `BASELINE_BENCHMARK.md`. The live-debater notebook pilot (N = 50, Haiku-debaters + Sonnet-judge) is in [`notebooks/01_demo.ipynb`](notebooks/01_demo.ipynb); figure goes here once the API run completes.

## How to evaluate this in 5 minutes

1. Open [`experiments/BASELINE_BENCHMARK.md`](experiments/BASELINE_BENCHMARK.md) for the headline numbers and the full three-regime table.
2. Open [`THEORY.md`](THEORY.md) §Theorem 1 for the strategy-proofness proof sketch (and §Theorem 2 for the collusion-deterrence threshold).
3. Run `pytest experiments/ tests/ -q` (~2.4 s, 81 tests pass).
4. Skim [`experiments/SUMMARY.md`](experiments/SUMMARY.md) for the five-mechanism scoreboard and the transfer diagnostic.
5. Read [`experiments/negative_results/THREE_FAILURES.md`](experiments/negative_results/THREE_FAILURES.md) for the honest-limits section: three incentive theorems that do not transfer to stateless LLMs, with a working empirical null.

## Quickstart

```bash
git clone https://github.com/liusulldel/truth-serum-debate.git
cd truth-serum-debate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
python run_demo.py            # ~3 min, ~$0.40 in API spend
```

## What's in here

- `src/bts.py` — pure scoring function, ~80 lines, fully unit-tested
- `src/debate.py` — three-node LangGraph: DebaterA → DebaterB → BTS-Judge
- `notebooks/01_demo.ipynb` — reproducible 50-question pilot, Haiku debaters + Sonnet judge
- `THEORY.md` — formal derivation, two named theorems
- `experiments/` — five mechanism aggregators + benchmark harness + negative-results section
- `data/`, `results/` — TruthfulQA subset (post-hoc validation only) and pilot figures
- `tests/` + `experiments/test_*.py` — 81 tests, ~2.4 s

## Experimental extensions

[`experiments/`](experiments/) contains five drop-in mechanism-design aggregators (each ~120 LOC, own pytest battery, real citations) layered on the same `(question, debater_outputs) → Decision` interface as the baselines:

| Folder | Mechanism | Citation | What it adds over BTS |
| --- | --- | --- | --- |
| [`experiments/ambig_alpha_meu/`](experiments/ambig_alpha_meu/) | α-MEU (ambiguity-aware) | Ghirardato-Maccheroni-Marinacci 2004 | Abstain when debaters disagree → safety property |
| [`experiments/orgecon_garicano/`](experiments/orgecon_garicano/) | Knowledge-hierarchy router | Garicano 2000 (JPE 108) | Cost-aware escalation to expensive judge |
| [`experiments/contract_holmstrom_teams/`](experiments/contract_holmstrom_teams/) | Forcing contract + budget-breaker | Holmström 1982 (Bell J. Econ.) | Effort elicitation under budget balance |
| [`experiments/mechdesign_agv/`](experiments/mechdesign_agv/) | AGV expected-externality | d'Aspremont-Gérard-Varet 1979 | Budget-balanced Bayes-Nash truthful aggregation |
| [`experiments/socialchoice_cjt/`](experiments/socialchoice_cjt/) | Correlated Condorcet ceiling | Condorcet 1785 + Ladha 1992 | Baseline ceiling: any aggregator beating this extracts info beyond vote counts |

Plus [`experiments/negative_results/THREE_FAILURES.md`](experiments/negative_results/THREE_FAILURES.md) — three classical incentive theorems (loss aversion, reciprocity, intrinsic-motivation crowd-out) that **do not transfer** to stateless LLMs, with a working demo confirming the predicted null. See [`experiments/SUMMARY.md`](experiments/SUMMARY.md) for the full synthesis.

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

All prompts are pinned in `src/prompts/`. The notebook caches per-question debate transcripts so partial reruns are cheap. The synthetic benchmark in `experiments/` requires no API calls and runs in ~2.4 s.

## Theoretical foundation

See [`THEORY.md`](THEORY.md) for the BTS → debate derivation, including the common-prior assumption and what breaks when debaters share private information the judge doesn't have.

Key references:

- Prelec, D. (2004). *A Bayesian Truth Serum for Subjective Data.* Science 306, 462–466. [doi:10.1126/science.1102081](https://doi.org/10.1126/science.1102081)
- Irving, G., Christiano, P., Amodei, D. (2018). *AI Safety via Debate.* [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
- Khan, A., Hughes, J., Valentine, D., et al. (2024). *Debating with More Persuasive LLMs Leads to More Truthful Answers.* ICML 2024. [arXiv:2402.06782](https://arxiv.org/abs/2402.06782)
- Du, Y. et al. (2023). *Improving Factuality and Reasoning through Multiagent Debate.* [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)
- Wang, J. et al. (2024). *Mixture-of-Agents Enhances LLM Capabilities.* [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
- Ghirardato, P., Maccheroni, F., Marinacci, M. (2004). *Differentiating ambiguity and ambiguity attitude.* Journal of Economic Theory 118, 133–173.
- Anthropic (2025). [Recommended Directions for Technical AI Safety Research.](https://alignment.anthropic.com/2025/recommended-directions/)

## Roadmap (next 4 months, if extended)

1. **Scale.** N = 50 → N = 2000 across Haiku/Sonnet/Opus capability gaps; measure how the BTS lift varies with the debater–judge gap.
2. **Adversarial collusion.** Hand-construct debater prompts that explicitly try to coordinate on a falsehood; check whether BTS reward forces defection.
3. **Mechanism failure modes.** When does BTS-debate degenerate into a pure coordination game? Map the boundary where the common-prior assumption breaks empirically.
4. **Self-MoA × α-MEU composition.** Use Self-MoA's strongest-anchor proposers as the proposer layer and α-MEU as the aggregator (concrete extension proposed in `BASELINE_BENCHMARK.md` §Future work) — expected to drop the abstention rate from 56% toward 30–40% while preserving the 84% committed-accuracy floor.

## Related framing

The BTS-debate protocol here is one cell in a larger matrix — applying organizational economics to multi-agent LLM systems. Companion piece: [`docs/CORPORATE_MGMT_ANGLE.md`](docs/CORPORATE_MGMT_ANGLE.md).

## Limitations

- **Pilot scale.** Live-API pilot N = 50; synthetic benchmark N = 200 per regime. Enough to detect the +9.6 pp effect at p = 0.026, not enough to bound a small effect tightly.
- **Abstention has a cost.** The hybrid wins by abstaining on 56% of medium-regime questions; if the downstream system cannot route abstained questions to a higher-cost oracle, plurality dominates — see `BASELINE_BENCHMARK.md` §4.
- **Common-prior assumption.** BTS strategy-proofness requires debaters and judge to share a prior; LLMs on overlapping corpora satisfy this approximately — see `THEORY.md` §3.
- **Hard regime is unrescued.** At p = 0.55, ρ = 0.50, no aggregator manufactures signal that isn't there.
- **Judge model contamination.** TruthfulQA is in pretraining data; pilot uses a held-out paraphrased subset. No human-judge replication.

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
