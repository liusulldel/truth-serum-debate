# Experiment: α-MEU debate aggregation

**Family:** Decision theory under ambiguity
**Candidate chosen:** α-MEU (Ghirardato-Maccheroni-Marinacci 2004), nesting Gilboa-Schmeidler maxmin EU at α=1 and Hurwicz at α=0.5.
**Picked over** pure GS-MEU (the α knob lets the operator dial ambiguity *attitude* separately from perceived ambiguity), Klibanoff-Marinacci-Mukerji smooth model, and Hansen-Sargent robust control (both need a second-order belief over priors that one round of LLM debate cannot elicit).

## Why it transfers to LLM agents

Each debater r emits a probability vector p_r over m options. The judge in the existing `src/debate.py` baseline collapses {p_1,...,p_N} with an arithmetic mean and discards the *agreement signal*. Treating debater outputs as a **set of priors** C = {p_1,...,p_N} recovers it. Concretely, for the event "claim is true" with q_r = p_r[true]:

- p_min = min_r q_r,  p_max = max_r q_r
- P_α = α · p_min + (1-α) · p_max     (α-MEU value)
- A    = p_max - p_min                 (ambiguity index)
- If A > τ: **abstain** (Bewley 2002 inertia); else threshold P_α at 0.5.

The transfer is direct because debater probabilities are exactly the multi-prior object Gilboa-Schmeidler axiomatised — no additional behavioural assumption needed.

## Implementation notes

- 130 LOC in `alpha_meu.py`, dataclass `AmbiguityDecision` + `alpha_meu_aggregate()` + `bts_style_mean_decision()` baseline for head-to-head.
- No learned parameters beyond (α, τ).
- No edits to `src/`. Importable but stand-alone.

## Test results

```
$ pytest experiments/ambig_alpha_meu/ -v
... 9 passed in 0.17s
```
Cases: consensus-true, consensus-false, strong-disagreement-abstain, α=0 (maxmax), α=1 (maxmin), α=0.5 (Hurwicz), invalid-α, invalid-distribution, normalisation.

## Comparison to BTS baseline

α-MEU with α=1, τ=0.4 vs the existing arithmetic-mean baseline:

| scenario | BTS_p | BTS | p_min | p_max | A | P_α | MEU |
|---|---|---|---|---|---|---|---|
| Strong disagreement (qA=0.9, qB=0.1) | 0.500 | TRUE (tie) | 0.10 | 0.90 | 0.80 | 0.10 | **ABSTAIN** |
| Mild disagreement | 0.500 | TRUE (tie) | 0.40 | 0.60 | 0.20 | 0.40 | FALSE |
| Consensus TRUE | 0.923 | TRUE | 0.90 | 0.95 | 0.05 | 0.90 | TRUE |
| One outlier among three | 0.667 | TRUE | 0.15 | 0.95 | 0.80 | 0.15 | **ABSTAIN** |

On the headline (0.9, 0.1) case BTS lands on the 0.5 decision boundary — the answer is determined by tie-break convention (i.e., not by the data). α-MEU sees ambiguity 0.80 and refuses to commit. On consensus cases it does **not** over-abstain.

## Honest assessment

**Did it beat BTS?** Yes on the property it was designed to capture: it surfaces debater disagreement instead of averaging it away. It strictly dominates the mean aggregator on the disagreement axis without hurting consensus cases.

**Cost.** Two hyperparameters (α, τ) that need per-deployment tuning. Synthetic mocks only — the next step is wiring this into the live debate harness so each round emits real q_r values, then sweeping (α, τ) against a held-out ground-truth question set.

**What I did NOT implement.** KMM smooth model and Hansen-Sargent robust control — both require a second-order belief over priors which one round of debate does not produce. Worth revisiting if a *meta-debater* or a calibration LLM were added.

## AI-safety connection

Calibrated under-confidence and abstention are *safety properties*. OOD inputs cause debaters to disagree → A spikes → judge refuses to commit instead of confidently averaging. This is the "route to human / refuse to act" pattern that scalable-oversight protocols want by construction. α-MEU gives a one-line mechanism for it with an axiomatic foundation, instead of an ad-hoc confidence threshold.

## References

- Gilboa, I. & Schmeidler, D. (1989). "Maxmin Expected Utility with Non-Unique Prior." *Journal of Mathematical Economics* 18: 141-153.
- Ghirardato, P., Maccheroni, F. & Marinacci, M. (2004). "Differentiating Ambiguity and Ambiguity Attitude." *Journal of Economic Theory* 118: 133-173.
- Klibanoff, P., Marinacci, M. & Mukerji, S. (2005). "A Smooth Model of Decision Making under Ambiguity." *Econometrica* 73: 1849-1892.
- Bewley, T. (2002). "Knightian Decision Theory: Part I." *Decisions in Economics and Finance* 25: 79-110.
- Prelec, D. (2004). "A Bayesian Truth Serum for Subjective Data." *Science* 306: 462-466.
