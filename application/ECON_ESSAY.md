# Anthropic Fellows — Economics Track Application

**Applicant:** Liu Su (`liusulldel`)
**Track:** Economics (mentors: Maxim Massenkoff, Peter McCrory)
**Date:** 2026-04-25

---

## 1. Why I am applying

Mechanism design and decision theory are not just normative tools. They are *measurement primitives*: they tell you what an "elasticity," a "preference," or a "substitution rate" can mean before you regress anything on anything. Massenkoff and McCrory's "Labor Market Impacts of AI: A New Measure and Early Evidence" (Anthropic, 2026-03-05) is a sharp example — the paper builds a task-level exposure measure from Claude.ai conversation logs and shows that occupational wage growth tracks AI-augmentation share, not AI-automation share. The next question I want to work on, and the one I am applying to work on with Maxim and Peter, is whether the augmentation-vs-automation split that the paper recovers from text is *consistent* with the substitution behavior firms actually reveal in their hiring and task-allocation decisions over the same window. If the two measures disagree, that disagreement is itself the policy-relevant quantity.

## 2. Where I fit, and where I do not

I should be honest about the gap first. My PhD work under `<PHD_ADVISOR>` at Princeton University (`<THESIS_TITLE>`) is in axiomatic decision theory and mechanism design; my published record is `<3_PUBLICATIONS_OR_WORKING_PAPERS>`. **I have not previously published with administrative labor microdata.** I have not run a CPS extract in anger, and I have not linked LEHD records. That is a real ramp-up cost and I am not going to dress it up.

Three things make the ramp-up tractable, and I think they make me a non-redundant addition to a team that is already strong on reduced-form labor.

*(a) Production engineering as applied mechanism design.* I have built and maintained three LangGraph pipelines in Python — most relevant here is `<SPECIFIC_LANGGRAPH_USE_CASE>` — where the day-to-day work is: specify agent roles, design payoff structures so that local agent behavior aggregates to a desired global property, and instrument it. That is mechanism design with a `pytest` suite. The framing is written up in `docs/CORPORATE_MGMT_ANGLE.md` of the repo linked below: LangGraph as institutional design with a compiler.

*(b) Axiomatic training as identification eyesight.* The thing axiomatic theorists do compulsively is ask, "what behavior on the data side would falsify this representation?" That is the same question an applied economist asks about an identifying assumption — separability, monotonicity, single-crossing — just in a different vocabulary. I expect to be useful in seminar by being the person who notices when a regression specification is implicitly assuming an Anscombe-Aumann mixture that the firm-level decision problem does not actually satisfy.

*(c) Demonstrated ramp-up speed.* In the 48 hours before submitting this application, I shipped a working artifact end-to-end: **github.com/liusulldel/truth-serum-debate** — a Bayesian-Truth-Serum-scored debate protocol as a scalable-oversight mechanism, with `THEORY.md` containing the formal incentive-compatibility derivation, a runnable LangGraph demo, and 81 passing unit tests. The repo ships **five distinct mechanism-design aggregators** (Prelec 2004 BTS, d'Aspremont-Gérard-Varet 1979 AGV, Holmström 1982 team contract, Ghirardato-Maccheroni-Marinacci 2004 α-MEU, Garicano 2000 knowledge hierarchy) each as a ~120-LOC Python module with its own pytest battery, plus a head-to-head benchmark against seven published multi-agent orchestration baselines (Self-MoA Li 2025, Du-debate 2023, MoA-lite Wang 2024, self-consistency Wang 2022, Khan-debate 2024, weighted-confidence, majority vote). The BTS + α-MEU hybrid wins the realistic medium regime by **+9.6 percentage points (84.1% vs 74.5%, bootstrap p = 0.026)**. The point of mentioning this here is not the artifact itself but the velocity: theorem to benchmarked working code in five days.

## 3. Project 1 — Revealed AI-Human Substitutability from Claude.ai Conversation Logs

This is the project I would most want to start with Maxim and Peter in week one. It is a direct extension of the March 2026 paper, not a tangent.

**Question.** Massenkoff and McCrory measure AI exposure from what Claude is *asked to do*. Firms reveal substitution from what they *stop hiring humans for*. Within a 12-month window, how tightly do these two measures co-move at the occupation × task level, and where do they diverge?

**Data.** Anthropic Economic Index public releases (occupational task distributions over Claude.ai conversations); O\*NET 28.x for the task taxonomy crosswalk; BLS OEWS for occupation-level employment and wage panels; if Anthropic-internal admin data on enterprise seat counts is available to Fellows, that anchors the firm side.

**Method.** A nested-logit choice model over (hire human, assign to AI, do not produce) at the task level, with separability constraints on the substitution matrix imposed *axiomatically* (independence of irrelevant tasks within a job-family nest). The separability constraints are testable, not assumed away — the first empirical exercise is whether the data reject them, and for which job families. Where they fail, that failure is the finding.

**Deliverables.**
1. An occupation × 12-month substitution-elasticity heatmap, comparable to Figure 3 of Massenkoff and McCrory but with a *behavioral* rather than text-based denominator.
2. A simple firm-side decision rule — "for an occupation with measured augmentation share α and substitution elasticity σ, the wage-vs-headcount adjustment that minimizes adjustment costs is..." — designed to be quotable in a one-page memo for the Economic Advisory Council (Cowen, Korinek, Horton, List, Tenreyro, Restrepo, Farronato).

**Twelve-week plan.**
- W1–2: data ingest, Economic Index replication, O\*NET crosswalk.
- W3–6: nested-logit estimation, axiomatic separability tests.
- W7–9: robustness — alternative nest structures, placebo occupations with low AI exposure, pre-2024 falsification window.
- W10–12: policy memo, working-paper draft, internal seminar.

The deliverable I am committing to is the memo. The paper is the by-product.

## 4. Project 2 — Auditing Claude as an Economic Agent

A smaller, parallel project that connects directly to the truth-serum-debate work.

I would build a 2,000-item standardized economic-choice battery — Anscombe-Aumann acts, Allais and Ellsberg variants, α-MEU ambiguity items, intertemporal trade-offs — and run it across Claude model versions. The output is a revealed-preference fingerprint per model: implied risk aversion, ambiguity aversion, discount rate, and the subset of axioms (Independence, Sure-Thing, Stationarity) the model satisfies versus violates.

Why this matters for the team's agenda: Korinek and Horton are both publicly interested in LLMs as "synthetic subjects" for economic research. That program needs a validity boundary — *which* economic primitives are stable in Claude and which drift across versions or prompts. The methodological link to the truth-serum-debate repo is direct: BTS is a peer-prediction mechanism for eliciting truthful reports from a *population*; this battery is the individual-level revealed-preference analogue. Both ask, "under what incentive and elicitation structure does the agent's report identify the underlying parameter?"

## 5. Engineering readiness

I do not need mentor time on the stack. Three LangGraph production pipelines in daily use, fluent with the Anthropic SDK including prompt caching and the Batch API, comfortable with `pandas` / `polars` / `statsmodels` / `linearmodels` / `pyfixest` for the econometrics side. The truth-serum-debate repo went from concept to working notebook with a proven theorem in five days, including the `pytest` suite and a CITATION.cff. I would like mentor time on ideas, identification, and which margins of the Massenkoff-McCrory paper are most worth pushing on — not on debugging a `merge_asof`.

## 6. What I want to learn

Three specific things that are gaps, not aspirations:
- Administrative labor microdata cleaning and linkage at production quality — CPS basic monthly files, LEHD QWI, IRS-style record linkage where available.
- Survey and field-experiment design at firm scale, the kind of work John Horton runs.
- The internal Anthropic Economic Index pipeline end-to-end, so that any extension I propose is grounded in what the data actually supports rather than what I wish it did.

## 7. Closing

I am asking to be evaluated on the marginal contribution of bringing axiomatic measurement primitives to a team that is already strong in reduced-form labor and policy translation. The Massenkoff-McCrory paper is the right starting point because it already takes seriously that what we *measure* about AI exposure determines what we can *say* about it. I want to push that one layer further: from text-revealed exposure to behavior-revealed substitution, and from there to a decision rule a firm or a regulator can act on.

---

**Referenced material**
- Massenkoff, M. and McCrory, P. (2026). "Labor Market Impacts of AI: A New Measure and Early Evidence." Anthropic. https://www.anthropic.com/research/labor-market-impacts
- Anthropic Economic Index. https://www.anthropic.com/economic-index
- Anthropic Economic Advisory Council. https://www.anthropic.com/news/introducing-the-anthropic-economic-advisory-council
- Anthropic Fellows — Economics Track JD. https://job-boards.greenhouse.io/anthropic/jobs/5183053008
- Applicant repo: github.com/liusulldel/truth-serum-debate (THEORY.md formal derivation, LangGraph demo, pytest suite, CORPORATE_MGMT_ANGLE.md framing note)
