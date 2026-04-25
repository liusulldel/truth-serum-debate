# POSITIONING.md

*Where this repo sits in the scalable-oversight and mechanism-design literatures, what it claims, and what it does not.*

---

## §1. The one-paragraph claim

This repo claims that **Bayesian Truth Serum (Prelec 2004) can be grafted onto a Khan-style LLM debate protocol to give debaters a Bayes-Nash incentive to report their true private posterior, that this incentive survives the LLM-as-agent setting under a precisely stated common-prior assumption, and that the resulting scores compose cleanly with an α-MEU (Ghirardato-Maccheroni-Marinacci 2004) abstention-aware aggregator at the judge layer.** It does *not* claim: (a) that BTS-scored debate beats vanilla debate at scale on real benchmarks (the included pilot is N=30 synthetic items in `data/questions.jsonl`); (b) that the common-prior assumption is empirically satisfied by current Claude models (it is not, exactly — see §4); (c) that this is a substitute for Brown-Cohen-Irving doubly-efficient debate or for prover-estimator debate (it is complementary — a mechanism for the *scoring* layer, not the *protocol-complexity* layer); or (d) that the formal IC proof in `THEORY.md` extends to multi-round debate without further assumptions. The contribution is a small, honest, reproducible bridge between two mature literatures plus the negative-result discipline (`experiments/negative_results/`) of saying which classical mechanisms do *not* transfer.

---

## §2. Where this fits in the scalable-oversight literature

### 2a. Single-agent inference-time search

The single-agent inference-time-search line — Chain-of-Thought (Wei et al. 2022, arXiv:2201.11903), Self-Consistency (Wang et al. 2022, arXiv:2203.11171), Tree of Thoughts (Yao et al. 2023, arXiv:2305.10601), Reflexion (Shinn et al. 2023, arXiv:2303.11366) — improves answer quality by spending more inference compute inside one model. These are **orthogonal** to this work in two senses. First, they have no second agent and therefore no game-theoretic structure to incentivize; there is nothing to make strategy-proof. Second, they compose with this work: a BTS-scored debate where each debater internally runs Self-Consistency or ToT is strictly more expressive than the version in `src/debate.py`. The orthogonality is the point: this repo is about the *scoring* of multi-agent outputs, not the *generation* of single-agent ones.

### 2b. Multi-agent debate

The directly-comparable line is multi-agent debate: Irving, Christiano, Amodei (2018, arXiv:1805.00899) for the original AI-safety formulation; Du et al. (2023, arXiv:2305.14325) for the empirical "improving factuality and reasoning via multiagent debate" demonstration; Khan et al. (2024, arXiv:2402.06782) for the "debate helps weak judges supervise strong debaters" result on QuALITY. **What this work adds over Khan 2024**: Khan's debaters are scored only by the binary judge verdict, which leaves coordinated lying as a Nash equilibrium whenever both debaters share a wrong prior. `THEORY.md` shows that adding the Prelec BTS score to Khan's reward gives a strictly proper scoring rule under common knowledge of the prior, so that truthful reporting of the posterior is the unique Bayes-Nash equilibrium in the limit. The contribution is a *formal IC proof for the debater layer*, not a new empirical benchmark.

### 2c. Multi-LLM ensembling

Mixture-of-Agents (Wang et al. 2024, arXiv:2406.04692) and the Self-MoA follow-up (Li et al. 2025, arXiv:2502.00674) aggregate multiple LLM outputs by *concatenation-and-rerank* — the aggregator sees all proposals and writes a synthesized answer. This works well in the **independent-error regime** (different models, different mistakes) and degrades in the **correlated-error regime** (same base model, same systematic blind spot). The α-MEU aggregator in `experiments/ambig_alpha_meu/` is built for the correlated regime: when debater probabilities disagree the ambiguity index *A = p_max − p_min* triggers Bewley-style abstention rather than averaging the disagreement away. This is *exactly* the Self-MoA failure mode (homogeneous models hide systematic error inside agreement) made into an aggregation primitive.

### 2d. Theoretical scalable oversight

The theoretical line — doubly-efficient debate (Brown-Cohen, Irving, Piliouras 2023+), prover-estimator debate (arXiv:2506.13609), market-making (Hubinger 2020), prover-verifier games (Kirchner, Chen, Kim et al. 2024, arXiv:2407.13692) — sets up the *complexity-theoretic* and *protocol-design* foundations. This repo is **complementary, not competing**: the BTS layer is a scoring rule that drops in *underneath* any of these protocols (any time you elicit a probabilistic report from an agent and another agent's prediction of that report, BTS applies). Section 5 below sketches an explicit BTS × prover-estimator composition.

---

## §3. Where this fits in the mechanism-design literature

### 3a. BTS and the peer-prediction family

Prelec (2004, *Science* 306:462-466) is the load-bearing primitive: a scoring rule for binary or categorical reports that uses each respondent's *prediction of the population frequency* of each answer to compute an "information score" rewarding answers that are surprisingly common given predictions. It is strictly proper in expectation when respondents share a common prior over an unobservable signal-generating world-state. Closely-related peer-prediction work — Miller, Resnick, Zeckhauser (2005, *Management Science* 51:1359-1373); Witkowski-Parkes (2012, AAAI); Kong-Schoenebeck (2019, *EC*) on "Information Theoretic Mechanism Design" — generalizes this to non-binary signals and weakens the common-prior assumption to a learnable prior. **What this repo adds**: an empirical implementation against LLM debaters (`src/bts.py`), a composition with abstention-aware aggregation (the α-MEU layer), and an honest accounting (in `experiments/negative_results/THREE_FAILURES.md`) of where the LLM-as-agent assumption breaks the proof.

### 3b. Strategy-proofness classics that do *not* transfer

`experiments/mechdesign_agv/REPORT.md` works through which classical mechanisms transplant into the LLM debate setting and which do not. Briefly: **Gibbard-Satterthwaite** (1973/1975) is a negative result about deterministic social-choice rules and so applies vacuously here (we use randomized scoring); **VCG** (Vickrey 1961, Clarke 1971, Groves 1973) requires an exogenous budget to subsidise truth-telling — there is no auctioneer in an LLM pipeline; **AGV / d'Aspremont-Gerard-Varet** (1979, *J. Public Econ.* 11:25-45, esp. Mas-Colell-Whinston-Green Prop. 23.D.5) does transfer because it is budget-balanced, and is implemented as the mechanism in `experiments/mechdesign_agv/`; **Myerson** (1981) optimal auctions and **Maskin** (1999) implementation theory both presuppose a designer with commitment power that an inference-time pipeline does not have. The discipline of saying *which* classical results transfer and why is, in my view, the most underrated part of the contribution.

### 3c. Decision theory under ambiguity

The aggregator side draws on decision-theory-under-ambiguity: Gilboa-Schmeidler (1989, *J. Math. Econ.* 18:141-153) maxmin expected utility; Ghirardato-Maccheroni-Marinacci (2004, *Econometrica* 72:1849-1892) α-MEU, which nests GS at α=1 and Hurwicz at α=0.5; Klibanoff-Marinacci-Mukerji (2005, *Econometrica* 73:1849-1892) smooth ambiguity; Bewley (2002, *Decisions in Economics and Finance* 25:79-110) inertia / abstention under incomplete preferences. The α-MEU choice is justified in `experiments/ambig_alpha_meu/REPORT.md` on three grounds: (i) it nests two reference models the operator may want to switch between, (ii) it requires only a set of priors as input — exactly what N debater probability vectors give you, with no second-order belief — and (iii) Bewley inertia gives a principled abstention criterion on the ambiguity index *A = p_max − p_min* that lets the judge refuse to commit when the debaters meaningfully disagree.

---

## §4. Honest limits — what this repo cannot do

A hostile reviewer should attack on these three fronts; I attack them first.

- **No live-debater empirical validation at scale.** `data/questions.jsonl` contains 30 hand-written declarative-fact items. The benchmark in `experiments/run_benchmark.py` is synthetic in the sense that the "debater probabilities" are sampled rather than elicited from live Claude calls. The IC proof is therefore validated *theoretically* and the aggregator is validated *behaviourally on synthetic inputs*; neither is validated against live debaters on a held-out reasoning benchmark. This is the single largest gap. §5 is the plan to close it.
- **The BTS common-prior assumption is not exactly satisfied by current LLMs.** Prelec's strict-properness proof assumes respondents share a common prior over the latent world-state generating the signals. Two Claude debaters seeded differently or run with different system prompts induce *correlated but distinct* posteriors; the proof goes through only in the limit where the correlation is perfect. `THEORY.md` §3 acknowledges this and gives the limiting argument; `experiments/negative_results/THREE_FAILURES.md` characterizes the gap empirically. The honest claim is *approximate* incentive compatibility under bounded prior divergence, not exact IC.
- **No formal connection to RLHF reward modeling.** BTS is a *post-hoc* scoring rule applied at inference time to elicited reports; RLHF (Christiano et al. 2017, Ouyang et al. 2022) is a *training-time* signal applied to model parameters. Whether a BTS-style elicitation could serve as a training reward — and whether doing so would inherit the strategy-proofness property at the meta-level — is a separate, harder analysis the repo does not attempt.

---

## §5. What a 12-week Anthropic Fellowship project would add

1. **Live N≈2000 benchmark.** TruthfulQA-mc (Lin et al. 2022) for hallucination, QuALITY (Pang et al. 2022) for long-context reading comprehension matching Khan et al. 2024, plus a hand-curated adversarial set targeting collusive equilibria. Live debater calls through the codex-lb pool, BTS scoring vs. vanilla judge baseline, ablation of the α-MEU layer.
2. **Empirical measurement of BTS common-prior failure modes across Claude versions.** Hold the question fixed, vary debater (Haiku-3.5, Sonnet-4.5, Opus-4.7) and seed; measure prior divergence and the BTS-IC gap as a function of divergence. Produces a calibration curve: "at this measured divergence, BTS scoring is within ε of strictly proper."
3. **Composition with prover-estimator debate** (Brown-Cohen-Irving 2025, arXiv:2506.13609). The estimator's probability output is exactly the BTS input format; this should be a natural drop-in and would inherit doubly-efficient guarantees for the protocol layer while keeping BTS strategy-proofness for the scoring layer.
4. **Paper draft** targeting either NeurIPS 2026 alignment workshop or *AI & Society*.

---

## §6. The single-sentence diagnostic

> **"Mechanism design transfers to LLM agents to the extent that the mechanism's incentive-compatibility proof relies only on properties of the agent's posterior, not on properties of the agent's hedonic or relational architecture."**

This is the load-bearing claim because it determines, *before* any empirical run, which classical results are worth porting. BTS, AGV, and α-MEU all pass the test: their proofs invoke only properties of reported probability distributions (strict propriety, budget balance, multi-prior representation). VCG, Myerson, Maskin all fail it: their proofs invoke properties of agent utility functions, designer commitment, or off-equilibrium punishment that an inference-time LLM pipeline does not instantiate. The diagnostic is what lets a small repo be principled rather than promiscuous about which mechanisms it claims to import — and it is the criterion against which the §5 fellowship work would be evaluated.
