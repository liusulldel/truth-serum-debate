# PROJECT_ASSESSMENT — Red-Team Review of "Incorporated Agents" as a Fellows Project

*Internal red-team document. Written assuming the user submits to Anthropic Fellows tomorrow. Brutal by design.*

---

## §1. The thesis in one paragraph

The project asserts: **LLM multi-agent systems are institutions, and classical mechanism design + decision theory + organizational economics give us strictly better tools for designing them than ad-hoc orchestration.** Concretely, a LangGraph DAG is isomorphic to an org chart (`docs/CORPORATE_MGMT_ANGLE.md` §1), so debate protocols inherit the litigation model (Dewatripont-Tirole 1999), RLHF labelers inherit peer-prediction (Prelec 2004), CAI principle aggregation inherits Arrow (1951), and tool-use safety inherits Holmström-Milgrom (1991). The empirical evidence in this repo is a single synthetic-debater benchmark (`experiments/BASELINE_BENCHMARK.md` §3 medium regime): the BTS + α-MEU hybrid attains 84.1% accuracy on its non-abstained 44% of questions vs. 74.5% for majority-vote, +9.6 pp at bootstrap p = 0.026, with seven published baselines beaten and the win confined to one of three regimes. The theoretical evidence is `THEORY.md` Theorems 1 and 2 — a port of Prelec's BTS to a two-player debate setting plus a "collusion deterrence under capability gap" corollary, both with proof sketches rather than full proofs, plus a transfer diagnostic (`experiments/SUMMARY.md`): "mechanism design transfers to LLM agents to the extent that the IC proof relies only on properties of the agent's posterior, not on its hedonic or relational architecture." Five mechanism modules ship as ~120-LOC drop-ins (`experiments/{ambig_alpha_meu,orgecon_garicano,contract_holmstrom_teams,mechdesign_agv,socialchoice_cjt}/`) with 38/38 passing tests but zero live-LLM validation.

## §2. Steelman (≈260 words)

The framing is genuinely non-redundant. Three points actually land.

**(a) The transfer diagnostic is publishable on its own.** The single sentence in `experiments/SUMMARY.md` line 21 — "mechanism design transfers iff the IC proof depends only on the posterior, not the hedonic architecture" — is the kind of one-line classifier that a scalable-oversight team could *use*. It immediately predicts that gift-exchange RLHF schemes will fail on stateless models, that BTS-style peer prediction will succeed, and that any future proposal can be pre-screened in five minutes. `experiments/negative_results/THREE_FAILURES.md` plus the loss-aversion demo (t = +0.57 null vs t = +9.40 for a mock human) is the rare case of a candidate *publishing* their negative results before the reviewer asks.

**(b) The applicant's stack is unusual.** A PhD in axiomatic decision theory + three production LangGraph pipelines + concept-to-tested-artifact in 5 days is a combination essentially nobody else applying to either track has. Massenkoff/McCrory will recognize the axiomatic identification eyesight (ECON_ESSAY §2b is well-pitched on this). The scalable-oversight team will recognize the engineering velocity.

**(c) BTS-as-debate-judge is a cheap, falsifiable hypothesis worth running.** The Khan et al. (2024) protocol genuinely lacks a strategy-proofness argument; the field knows this; nobody has ported peer-prediction in. Even if BTS-debate fails empirically, a clean falsification on N = 2000 questions with a measured collusion-survival curve is a publishable result. The repo (THEORY.md §6) names exactly the four predictions that would refute it — a posture reviewers reward.

The project has a real claim, a real test, and a real escape route from the test.

## §3. Red team — five attacks (≈820 words)

### Attack 1 — "+9.6 pp on synthetic mock debaters proves nothing."

**Sharpened.** The `BASELINE_BENCHMARK.md` headline result uses a `numpy.random.default_rng(seed=42)` Bernoulli generator with hand-tuned `(p=0.70, ρ=0.30)`, not a Claude API call. There are zero rows of live-debater data in the repo. `notebooks/01_demo.ipynb` is described as "figure goes here once the API run completes" (README §"Headline result"). The α-MEU win is mechanically guaranteed by the abstention rule on synthetic data: it abstains on 56% of questions and reports the easy half. Any reviewer with 30 seconds will see this.

**Repo response.** Honest in §4 of `BASELINE_BENCHMARK.md` lines 90–102 ("where the mechanism stack does NOT win") and the limitation is flagged in `README.md` line 100. But honesty is not evidence. The N = 50 live pilot is *promised*, not delivered.

**Fix in 12 weeks.** Week 1–3: run the N = 2000 Khan-style live-debater eval the README roadmap §1 already names. Stratify by Haiku/Sonnet/Opus capability gap. If BTS+α-MEU does not beat vanilla debate live, report the null cleanly and pivot the paper to "when does peer-prediction transfer survive correlated foundation-model errors?" — itself a contribution.

### Attack 2 — "Old theorems with an LLM glue layer ≠ new theory."

**Sharpened.** Prelec (2004), AGV (1979), Holmström (1982), Garicano (2000), α-MEU (2004) are all 20–47 years old. Theorem 1 in `THEORY.md` is a two-player restatement explicitly attributed to Radanovic-Faltings 2013. Theorem 2 ("collusion deterrence") is a sketch with a "[Proof sketch]" rather than a full proof. The CORPORATE_MGMT_ANGLE document is a *table* mapping LangGraph primitives to org-econ objects, not a theorem.

**Repo response.** Partial. `experiments/SUMMARY.md`'s transfer diagnostic *is* a novel contribution — a falsifiable classifier no one else has stated. But it is one sentence. The rest of the repo is a careful application of known machinery.

**Fix in 12 weeks.** Convert the diagnostic into a formal theorem: state precisely what "depends only on the posterior" means in measure-theoretic terms over the agent's policy class, then prove that this predicate partitions classical mechanism design results into a transferring set and a non-transferring set. That is genuinely new theory and it needs the diagnostic to do the work.

### Attack 3 — "The corporate-management angle is metaphor, not contribution."

**Sharpened.** `CORPORATE_MGMT_ANGLE.md` §3 maps "refusal training ↔ Kydland-Prescott rule-vs-discretion." Holmström did not have stateless context-window agents in mind. The mappings are evocative but neither makes a Holmström prediction sharper, nor lets us solve a LangGraph problem we couldn't solve before. None of the five "open questions" in §6 are answered or even attempted in the repo.

**Repo response.** Weak. `THREE_FAILURES.md` partly answers this by *empirically* showing that prospect theory and gift exchange do *not* transfer — i.e., the metaphor has real limits the document acknowledges. But the dictionary itself is not load-bearing for any deliverable.

**Fix in 12 weeks.** Pick exactly one mapping (most promising: Aghion-Tirole real-vs-formal authority predicting LLM sycophancy / refusal-override behavior) and build a falsifiable empirical test with current Claude models. One mapping with a clean experiment beats a 13-row dictionary.

### Attack 4 — "Anthropic already has scalable oversight. Why this?"

**Sharpened.** Anthropic has Bowman et al. 2022, Khan et al. 2024 (collaboration), prover-verifier games (Kirchner et al. 2024), and Constitutional AI. The scalable-oversight team has internal headcount and roadmap. A peer-prediction layer on top of debate is a small delta, not a research direction.

**Repo response.** SAFETY_ESSAY.md §3 frames BTS-debate as "complementary, not competing" with existing oversight work — explicitly a *judging-head* layer. THEORY.md §5 cites the 2025 Recommendations memo's gap ("evaluated empirically rather than derived from incentive-compatible mechanisms") and slots into it. This is the strongest defense in the repo.

**Fix in 12 weeks.** Talk to scalable-oversight team ahead of submission about whether they consider the IC-derivation gap real. If yes, the project is a clean fit. If they say "we know, we deprioritized it," the framing dies and the user wastes 12 weeks.

### Attack 5 — "Zero published AI papers; mechanism-design background underweighted."

**Sharpened.** The alignment field weights ML publications (NeurIPS/ICML/ICLR), not Econometrica. The applicant's three working papers are in social choice and bounded rationality. Reviewers asked to evaluate "mechanism design for AI" by reflex compare against ML-trained applicants who built X-eval-on-Y-model with N = 50K and submit to ICLR.

**Repo response.** Mostly indirect. The repo *demonstrates* engineering velocity (5 days, 81 tests, 7 baselines reproduced from arXiv) which is the implicit answer. SAFETY_ESSAY.md §5 makes the bridge claim: "the toolkit alignment increasingly needs and currently imports informally."

**Fix in 12 weeks.** Submit the eventual paper to the ICML Mechanism Design for AI workshop or NeurIPS Cooperative AI workshop simultaneously with main-track submission. A workshop accept during the Fellowship is the credential conversion the applicant needs.

## §4. Alternative framings (≈260 words)

**Alternative A — Pure Econ on Massenkoff-McCrory extension** (`application/ECON_ESSAY.md` §3). Nested-logit substitution-elasticity heatmap from Anthropic Economic Index + O\*NET + BLS OEWS, with axiomatic separability tests. *Verdict: this is the better Econ-track pitch.* It is a direct extension of mentor work, requires no AI-safety framing, and uses the applicant's axiomatic toolkit on the *identification* side where it is strongest. The "incorporated agents" framing is **dominated** here — it is a distraction from a clean policy-relevant deliverable.

**Alternative B — Savage P1–P7 axiom violations across Claude versions** (SAFETY_ESSAY.md Pitch B, plus ECON_ESSAY.md §4). 2,000-item revealed-preference battery, axiom-violation rates per model version, optional mech-interp partnership. *Verdict: ambiguous.* Cleaner thesis ("does Claude have coherent preferences?") and directly model-welfare-relevant. The "incorporated agents" framing **partially competes**: Pitch B is one corner of the same axiomatic-decision-theory toolkit, framed as model-welfare rather than scalable-oversight. Submitting both as twin pitches inside one application (which `SAFETY_ESSAY.md` already does) is fine.

**Alternative C — Pure interpretability on circuit-level preference detection.** Lindsey-style circuit tracing for "preference circuits" supporting/violating Independence. *Verdict: dominated by the applicant's revealed comparative advantage.* The applicant has no published interp work and would burn 8 of 12 weeks ramping on the tooling. Avoid.

**Net.** Pitch A (BTS-debate, "incorporated agents") is the *Safety* pitch. Pitch B (Savage axioms) is the *welfare* hedge. The Econ-track essay correctly does not lead with "incorporated agents" — it leads with the Massenkoff-McCrory extension and uses the repo as a velocity proof.

## §5. Final verdict (≈170 words)

**Submit. Do not pivot.** The current essay structure is correct.

- **Econ track:** P(framing lands) ≈ 0.55. Massenkoff-McCrory will read ECON_ESSAY.md as a mechanism-design-trained labor extension, which it is. The "incorporated agents" framing is *deemphasized* in the Econ essay (mentioned only in §2a as an engineering credibility signal) and that is correct. Risk: mentors may want a labor economist, not a decision theorist; the no-admin-microdata gap is real.

- **Safety track:** P(framing lands) ≈ 0.40. The scalable-oversight team may already have considered and rejected peer-prediction layers, in which case Pitch A dies and Pitch B (Savage / model welfare) carries the application. The +9.6 pp synthetic result is the single most exposed claim — Attack 1 is **the most damaging un-answered red-team** point.

**Cheapest hedge:** add a one-paragraph Pitch C to SAFETY_ESSAY.md proposing the **transfer-diagnostic formalization** as a 12-week theory-only deliverable, so that if both BTS-debate (empirical) and Savage-axioms (welfare) are deemed redundant, there is still a pure-theory project the applicant uniquely can do. Cost: 30 minutes. Insurance value: substantial.
