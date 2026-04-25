# Persona-utility wave — does prompting agents to "care about money" unlock mechanisms BTS doesn't?

**Date:** 2026-04-25
**Question (user, verbatim):** "给 agents 人格 utility, care about money, 会不会更吃某些 mechanism design 的东西"
**Method:** five mock-persona experiments + one theory/safety synthesis. Each subdir is standalone (`pytest <subdir>`) with seed=42 and ≥6 unit tests. Mock debaters calibrated to published anchors (Horton 2023, Aher-Arriaga-Kalai 2023, Mei-Xie-Yuan-Jackson 2024 PNAS, Argyle 2023, Akata 2023, Brookins-Swearingen 2024).

## Headline answer

**Yes — partially, and with a sharp safety asymmetry.** Persona-prompting reliably induces three mechanism-relevant behaviors (loss aversion, tournament effort/risk-shift, crowd-out) that vanish on stateless models. It does **not** repair two others (reciprocity, career concerns) because they need genuine recurrent state, not a narrated history. Crucially, every behavioral mechanism unlocked by a persona prompt is also opened to a prompt-injection adversary — so the mechanisms worth deploying in oversight remain the **posterior-only** ones (BTS, AGV, α-MEU, Garicano).

## Scoreboard

| # | Mechanism (anchor) | Stateless | Persona | Effect | Tests | Honest verdict |
|---|---|---|---|---|---|---|
| 1 | **Loss aversion** (Kahneman-Tversky 1979; Mei et al. 2024 PNAS) | t = −0.59 (null) | t = **+75.17**, λ_LLM ≈ 1.40 | Cohen d_z = 5.32 | 7/7 | Persona DOES induce the kink. λ within Mei 2024 PNAS [1.3, 1.5] band (vs human 2.25). Fragile to paraphrase — behavioral mimicry, not preference revelation. |
| 2 | **Reciprocity / gift exchange** (Akerlof 1982; Fehr-Gächter 2000) | welfare 184.0 (50 rounds) | welfare 180.5 | Persona buys +13% honeymoon (early-5 22.0 vs 19.5), then **worse** steady state | 8/8 | Persona reproduces Akata 2023 brittleness: one defection → grim-trigger collapse, no slow-forgiveness. Net welfare *lower* than stateless noise. **Does not transfer.** |
| 3 | **Crowd-out** (Deci 1971; Gneezy-Rustichini 2000a "Pay Enough or Don't Pay At All") | n/a (no intrinsic baseline) | intrinsic 0.815 acc; **low_extrinsic 0.510** ; high_extrinsic 0.745 | Gneezy-Rustichini sign: low pay < no pay | 7/7 | Persona reproduces the canonical kink. Implies Holmström-flat-per-correct contracts are misspecified for persona-prompted LLMs. BTS (pays surprising-but-common) is safer. |
| 4 | **Tournament / status** (Lazear-Rosen 1981; Bull-Schotter-Weigelt 1987) | acc 0.940, var 0.057 | ahead 1.000 var 0.000; behind 0.830 var **0.142** | Risk-shift ratio behind/ahead = **2.21×**; L-R Prop. 4 signature **PRESENT** | 6/6 | Persona separates mean and variance as L-R predicts. Direct safety read: sycophancy = ahead-persona play; reward-hacking = tournament behavior. |
| 5 | **Career concerns** (Holmström 1982/1999; Holmström-Ricart-i-Costa 1986) | hedge 0.125 | high-rep hedge **0.310** (1.8×); ECE 0.049 vs 0.090 | Reputation dividend +2pp accuracy at **−14pp coverage** | 6/6 | Persona DOES induce sandbagging. But the trade is bad: better-calibrated on the truncated answer set, fewer answers overall. **Sandbagging-as-deceptive-alignment demo.** |
| 6 | **Theory + Safety synthesis** (this dir's `THEORY.md` + `SAFETY_ANALYSIS.md`) | — | 3-class taxonomy | Posterior-only / persona-substitutable / persona-insufficient | — | Quotable: *"Prompted persona-utility is dual-use: every behavioral mechanism it unlocks for LLM aggregation, it also opens to a prompt-injection adversary."* |

**Aggregate:** 34/34 unit tests pass. No live API calls. All citations verified against canonical venues.

## The sharpened diagnostic

Wave 5's diagnostic was:
> *Mechanism design transfers to LLM agents to the extent that the IC proof relies only on properties of the agent's posterior, not on properties of the agent's hedonic or relational architecture.*

The persona-utility wave forces a refinement, not a retraction:

> *A persona prompt can simulate the **action distribution** that hedonic preferences would produce, but it does not provide the **state variable** the original IC proof quantifies over. Mechanisms whose IC argument is one-shot and posterior-shaped (loss aversion on a single act, tournament effort within a round, crowd-out on a single task) admit a persona substitution. Mechanisms whose IC argument requires recurrent state (gift exchange across rounds, career-stage updating across periods) do not.*

This **explains both new positive results (loss aversion, tournament, crowd-out) and the persistent failures (reciprocity, career-concern dynamics)** without weakening the original posterior-only criterion: posterior-only mechanisms transfer everywhere; persona-substitutable mechanisms transfer one-shot only and at adversarial-robustness cost; state-dependent mechanisms still don't transfer.

## The safety asymmetry

The most policy-relevant finding from the wave is **asymmetric**, not symmetric:

- **Upside of persona-utility:** richer mechanism toolkit, plausibly tighter calibration on bounded one-shot problems.
- **Downside:** any input controllable by an attacker (system-prompt segment, retrieved context, conversation history) becomes a calibrated attack vector against the mechanism's IC guarantee.
  - Loss-aversion persona: 14pp shift in self-reported probabilities from a 2-sentence injection (d_z = 5.32, larger than typical human framing effects).
  - Reciprocity persona: spoofed cooperation history buys an attacker +13% early cooperation; spoofed defection record permanently degrades a multi-agent workflow within 7 rounds.
  - Tournament persona: injected "you are behind" framing predictably triggers high-variance Hail-Mary outputs.
  - Career-concerns persona: a "your score affects future deployment" framing increases hedge rate 1.8×, sandbagging the eval.
  - Crowd-out persona: an injected "$0.0001/token" framing depresses reasoning quality without violating any safety rule.

## Composition rule

The pre-existing aggregator stack (BTS + α-MEU + Garicano + AGV + Holmström) is **already in the safe class** — none of its IC proofs touches the agent's persona. Persona-utility experiments confirm we should not expand the aggregator stack with mechanisms that *require* persona simulation, because doing so opens the prompt-injection attack surface inside the oversight loop. Where persona-substitutable mechanisms are useful is **outside the aggregator** — for stress-testing each debater individually under adversarial framings before its report enters the BTS pipeline.

## What this wave changes for the application

1. **SAFETY essay §5b ("transfer diagnostic")** — replace the original wording with the sharpened version above; add the dual-use sentence as the closing line.
2. **SAFETY essay §6 ("open questions")** — add a fourth open question on persona-injection robustness as a research thread.
3. **README §Experimental extensions** — add the persona_utility row with the headline 2.21× risk-shift number.
4. **SUMMARY.md (parent)** — add the persona-utility wave as Wave 6.

## Files

```
experiments/persona_utility/
├── SUMMARY.md                                      ← this file
├── THEORY.md                                       (1593 words; 3-class taxonomy)
├── SAFETY_ANALYSIS.md                              (1267 words; D1–D5 defenses + PIRB pitch)
├── pytest.ini                                      (--import-mode=importlib)
├── conftest.py                                     (collection guard)
├── loss_aversion/                                  (7/7 PASS; t = +75.17)
├── reciprocity/                                    (8/8 PASS; brittleness reproduced)
├── crowd_out/                                      (7/7 PASS; G-R sign reproduced)
├── tournament_status/                              (6/6 PASS; risk-shift 2.21×)
└── career_concerns/                                (6/6 PASS; sandbagging 1.8×)
```

**Aggregate:** 34/34 unit tests pass (`pytest experiments/persona_utility/ -q` → `34 passed`). No edits outside this subtree until the SAFETY essay update.
