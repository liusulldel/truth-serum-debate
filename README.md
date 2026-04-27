# truth-serum-debate

Bayesian Truth Serum scoring for LLM debate, with a synthetic benchmark and a short theory note.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liusulldel/truth-serum-debate/blob/main/notebooks/01_demo.ipynb)
[![Anthropic API](https://img.shields.io/badge/Anthropic-Claude-cc785c.svg)](https://docs.anthropic.com/)

## What this is

This repo tests whether Bayesian Truth Serum (BTS; Prelec 2004) can help judge multi-agent LLM debates when debaters may make correlated errors.

The main experiment compares BTS-based aggregators against standard baselines: majority vote, vanilla debate, self-consistency, weighted confidence, Du-style debate, MoA-lite, and Self-MoA. The benchmark is synthetic, so the results are best read as controlled comparisons rather than production claims.

## Main result

In the medium correlated-debater regime (`p = 0.70`, `rho = 0.30`, `N = 200`, `seed = 42`), BTS plus alpha-MEU reached 84.1% accuracy versus 74.5% for majority vote, the best non-abstaining baseline in this run. The bootstrap one-sided p-value was 0.026.

The main caveat is abstention: BTS plus alpha-MEU abstained on 56% of questions. That only helps if abstained cases can be routed to another process.

| Aggregator | Accuracy | 95% CI | Abstain % |
| --- | ---: | --- | ---: |
| BTS + alpha-MEU | 0.841 | [0.761, 0.909] | 56.0 |
| alpha-MEU only | 0.838 | [0.750, 0.912] | 60.0 |
| majority_vote | 0.745 | [0.680, 0.805] | 0.0 |
| du_debate | 0.730 | [0.670, 0.790] | 0.0 |
| moa_lite | 0.715 | [0.645, 0.775] | 0.0 |
| weighted_confidence | 0.700 | [0.635, 0.765] | 0.0 |
| self_consistency | 0.690 | [0.625, 0.755] | 0.0 |
| self_moa | 0.685 | [0.620, 0.750] | 0.0 |
| BTS_top_score | 0.665 | [0.600, 0.730] | 0.0 |
| vanilla_debate | 0.660 | [0.590, 0.725] | 0.0 |

See [`experiments/BASELINE_BENCHMARK.md`](experiments/BASELINE_BENCHMARK.md) for all regimes, including the cases where this method is not helpful.

## Start here

1. Read [`experiments/BASELINE_BENCHMARK.md`](experiments/BASELINE_BENCHMARK.md) for the benchmark setup, full tables, and caveats.
2. Read [`THEORY.md`](THEORY.md) for the BTS debate argument and assumptions.
3. Run `pytest experiments/ tests/ -q` to check the synthetic benchmark and unit tests.
4. Skim [`experiments/negative_results/THREE_FAILURES.md`](experiments/negative_results/THREE_FAILURES.md) for mechanisms that did not transfer in this setup.

## Quickstart

```bash
git clone https://github.com/liusulldel/truth-serum-debate.git
cd truth-serum-debate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
python run_demo.py            # runs one small live API debate
```

To run the local tests:

```bash
pytest experiments/ tests/ -q
```

## Repository map

- [`src/bts.py`](src/bts.py): BTS scoring function.
- [`src/debate.py`](src/debate.py): two-debater, one-judge orchestration and CLI defaults.
- [`src/eval.py`](src/eval.py): batch evaluation harness over `data/questions.jsonl`.
- [`run_demo.py`](run_demo.py): one small live API debate round.
- [`notebooks/01_demo.ipynb`](notebooks/01_demo.ipynb): live API pilot notebook.
- [`THEORY.md`](THEORY.md): theory note and assumptions.
- [`experiments/`](experiments/): benchmark harness, baseline comparisons, and negative results.
- [`experiments/BASELINE_BENCHMARK.md`](experiments/BASELINE_BENCHMARK.md): main synthetic benchmark report.
- [`tests/`](tests/) and `experiments/*/test_*.py`: unit and benchmark tests.

## Reproducibility notes

| Component | Value |
| --- | --- |
| Python | 3.11+ |
| Anthropic SDK | `anthropic>=0.40` |
| LangGraph | `langgraph>=0.2` |
| Random seed | 42 |
| Default models | See `src/debate.py` and `src/eval.py` |

The synthetic benchmark in `experiments/` does not require API calls. The live notebook caches per-question debate transcripts, so partial reruns are cheaper.

## Limitations

- The live API pilot is small (`N = 50`), and the synthetic benchmark uses `N = 200` per regime.
- Abstention does most of the work in the main result. Without a fallback for abstained questions, the headline comparison matters less.
- The BTS derivation relies on a common-prior assumption. [`THEORY.md`](THEORY.md) explains where that assumption matters.
- In the hard regime (`p = 0.55`, `rho = 0.50`), the tested aggregators do not recover reliable signal.
- TruthfulQA may appear in model pretraining data. The pilot uses a held-out paraphrased subset, but that is not the same as a human-judge replication.

## References

- Prelec, D. (2004). *A Bayesian Truth Serum for Subjective Data.* Science 306, 462-466. [doi:10.1126/science.1102081](https://doi.org/10.1126/science.1102081)
- Irving, G., Christiano, P., Amodei, D. (2018). *AI Safety via Debate.* [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
- Khan, A., Hughes, J., Valentine, D., et al. (2024). *Debating with More Persuasive LLMs Leads to More Truthful Answers.* ICML 2024. [arXiv:2402.06782](https://arxiv.org/abs/2402.06782)
- Du, Y. et al. (2023). *Improving Factuality and Reasoning through Multiagent Debate.* [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)
- Wang, J. et al. (2024). *Mixture-of-Agents Enhances LLM Capabilities.* [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
- Ghirardato, P., Maccheroni, F., Marinacci, M. (2004). *Differentiating ambiguity and ambiguity attitude.* Journal of Economic Theory 118, 133-173.

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

MIT. See [`LICENSE`](LICENSE).
