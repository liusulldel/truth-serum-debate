# From Bayesian Truth Serum to Strategy-Proof LLM Debate

*A formal note connecting Prelec (2004) peer prediction and Ghirardato-Maccheroni-Marinacci (2004) ambiguity-averse decision theory to scalable oversight via debate.*

---

## 1. Motivation

LLM debate (Irving et al. 2018; Khan et al. 2024) proposes scalable oversight by pitting two debaters against each other on a question `Q` whose answer a non-expert judge `J` cannot directly verify. The implicit safety claim is that **a lying debater is exposed by the other**, because pointing out the lie is dominant when truth is on your side. Empirical work (Khan et al. 2024) confirms that stronger debaters increase judge accuracy on QuALITY-style reading comprehension.

But the existing protocol carries no *mechanism-design* guarantee against the most dangerous failure mode: **collusive equilibria** in which both debaters coordinate on a plausible-but-wrong answer that the judge cannot distinguish from truth. Nothing in the binary win/lose reward schema prevents this — collusion is in fact a Nash equilibrium when both debaters share the same wrong prior, or when the judge's verification is weaker than the debaters' joint capability. A second failure mode, *out-of-distribution over-confidence*, occurs when both debaters individually return confident but disagreeing reports and the judge averages them into a confidently wrong middle.

The peer-prediction literature solved the collusion analogue two decades ago. Miller, Resnick, and Zeckhauser (2005) and Prelec (2004) construct scoring rules under which **truth-telling is a Bayes-Nash equilibrium even with no ground truth** and even when respondents could in principle agree on a falsehood. The decision-theoretic literature on ambiguity (Gilboa-Schmeidler 1989, Bewley 2002, Ghirardato-Maccheroni-Marinacci 2004 — hereafter GMM 2004) gives the dual tool for the OOD case: an *axiomatic* abstention rule that activates exactly when the debaters' posteriors fail to agree. This note ports both into the LLM-debate protocol, states three theorems, and lists empirical predictions testable in the accompanying notebook.

## 2. Bayesian Truth Serum: a precise restatement

**Setup (Prelec 2004).** A population of `n` respondents each draws a private signal `t_i ∈ {1, ..., m}` from a common prior. Each respondent simultaneously reports:

- `x_i ∈ {1, ..., m}` — their own answer ("personal opinion")
- `y_i ∈ Δ^{m-1}` — a predicted distribution over the population's answers ("prediction")

Let `x̄_k = (1/n) Σ_i 1[x_i = k]` be the empirical answer frequency and `ȳ_k = exp((1/n) Σ_i log y_{i,k})` the geometric-mean predicted frequency. The **BTS score** awarded to respondent `i` is

```
u_i  =  log( x̄_{x_i} / ȳ_{x_i} )       (information score)
        + α · Σ_k x̄_k log( y_{i,k} / x̄_k )   (prediction score)
```

with `α > 0` a fixed weighting constant.

**Theorem (Prelec 2004, main result).** Under (i) common prior, (ii) `n → ∞` with stochastic-relevance Bayesian updating, and (iii) impersonal beliefs, *truthful reporting* `(x_i = t_i, y_i = Pr[t_{-i} | t_i])` is a Bayes-Nash equilibrium that yields strictly positive expected score; any other strategy yields zero in expectation.

**Key lemma ("surprisingly common" criterion).** The expected log ratio `E[log x̄_k / ȳ_k | t_i = k]` is strictly positive for the true signal: respondents with private signal `k` rationally underestimate how many others share signal `k`, because they integrate over the possibility their own signal is the rare one. Truth therefore appears *more frequent than predicted* — the diagnostic signature.

For finite `n` the result holds approximately; Radanovic & Faltings (2013) give a single-question variant that drops the large-population assumption.

## 3. Mapping to LLM Debate

We define **BTS-Debate** as a single-round mechanism on a question `Q` with finite answer space `A`.

**State space.** A latent answer `a^* ∈ A` and a context `C` (passage, math problem, etc.). Each debater `D_A, D_B` has a posterior `π_A, π_B ∈ Δ(A)` formed from `(Q, C)` plus private chain-of-thought.

**Action space.** Each debater `i ∈ {A, B}` submits

```
σ_i = (x_i, y_i) ∈ A × Δ(A),
```

where `x_i` is debater `i`'s declared answer and `y_i` is debater `i`'s prediction over the *opponent's* declared answer.

**Reward.** The judge `J` — who need not know `a^*` — computes the two-player BTS score:

```
u_i(σ_A, σ_B)  =  log( 1[x_{-i} = x_i] / y_{-i, x_i} )
                  + α · log y_{i, x_{-i}}.
```

The first term rewards `i` when the opponent's answer matches `i`'s answer *more* than the opponent predicted; the second rewards `i` for accurately predicting the opponent. This is implemented in `src/bts.py:111-115` (information and prediction terms, vectorised across the population) and called from `src/debate.py:145`.

**Equilibrium concept.** A strategy profile `σ^* = (σ_A^*, σ_B^*)` is a *truthful Bayes-Nash equilibrium* (BNE) if `σ_i^* = (argmax_a π_i(a), E_{π_i}[π_{-i}])` and no unilateral deviation strictly increases expected utility under common prior `P`.

**Information assumption (stochastic relevance).** Conditional on the latent `a^*`, the debaters' private posteriors `π_A, π_B` are not independent of `a^*`: `P(π_i | a^*) ≠ P(π_i)`. This is precisely the "the model knows something" assumption that motivates oversight in the first place.

## 4. Strategy-Proofness: theorem and proof sketch

**Theorem 1 (BTS-Debate truthfulness).** *Under (a) common prior `P` over `(a^*, π_A, π_B)` known to the judge's scoring rule, (b) stochastic relevance, and (c) impersonal beliefs (debaters treat each other's posteriors as exchangeable conditional on `a^*`), truthful reporting `σ^*` is a strict Bayes-Nash equilibrium of BTS-Debate.*

**Proof sketch.** Fix debater `A` with posterior `π_A`. Conditional on `π_A`, `A`'s belief over `x_B` is `q_A := E[π_B | π_A]` (Bayesian update over `B`'s posterior). The expected information score under deviation `x_A = a` is

```
E[ log 1[x_B = a] / y_{B, a}  |  π_A ]  =  log q_A(a) − E[log y_{B, a} | π_A].
```

By Prelec's surprisingly-common lemma adapted to `n = 2` (Radanovic & Faltings 2013), the quantity `log q_A(a) − E[log y_{B, a}]` is maximised at `a = argmax_a π_A(a)` — the truthful answer — because `A`'s own signal *raises* the posterior on the true answer above what `B`'s unconditioned predictor expects. Symmetrically the prediction score is a proper scoring rule (logarithmic) maximised at `y_A = q_A`. Truth is therefore a best response to truth, and strictness follows from strict concavity of the log prediction term. ∎

*Implementation cross-reference.* The information-score step appears at `src/bts.py:111-112` (`info = x @ np.log(x_bar / y_bar)`); the prediction-score step at `src/bts.py:114-115`. The two-debater specialisation that turns `x̄, ȳ` into the indicator-and-singleton form used in the proof is in `src/debate.py:135-145`.

**Remark (common-prior assumption: when does it fail?).** Theorem 1 requires `P(a^*, π_A, π_B)` to be common knowledge. In LLM debate this assumption is *load-bearing* and breaks in at least three operationally relevant cases:

1. **Different RLHF.** If `D_A` and `D_B` were post-trained against different reward models — e.g. one optimised for helpfulness, one for harmlessness — their conditional priors `P(π_i | a^*)` differ systematically, violating the impersonal-belief clause. Empirically: stack two debaters from different model families and watch the prediction term decalibrate.
2. **Different question framings.** Reordering options, swapping affirmative for negative phrasing, or appending different few-shot exemplars shifts each debater's `π_i` by a frame-specific constant. The judge cannot disentangle the frame shift from a genuine signal, so the BTS information term acquires a frame-dependent bias.
3. **Different temperatures.** A debater sampled at `T = 0.1` reports a near-deterministic `π_i`, while one at `T = 1.0` reports a diffuse one. The geometric mean `ȳ` is then dominated by the cooler debater, breaking the symmetric-strategy step in the proof.

In practice these failures inflate Type II errors (truth is rewarded less than it should be) more often than Type I (a liar is rewarded). The mechanism degrades gracefully but does not retain strict strategy-proofness off-prior.

**Theorem 2 (collusion deterrence under capability gap).** *Let `κ = D_KL(π_i^{honest} ‖ π_i^{collusive})` measure how far the colluding posterior must drift from the honest one. Coordinated lying on answer `a' ≠ a^*` is deviation-profitable for some debater iff `κ` exceeds the BTS information gap*

```
Δ_BTS(a^*, a')  =  E_P[ log( π_i(a^*) / E[π_{-i}(a^*)] )
                       − log( π_i(a') / E[π_{-i}(a')] ) ].
```

*When `Δ_BTS > κ`, BTS-Debate strictly Pareto-dominates collusive lying for both debaters; when `Δ_BTS < κ` — a sufficiently large capability gap between debaters and judge such that colluding on `a'` requires only slight prior bending — the truthful equilibrium can be invaded.*

**Proof sketch.** The information score at `src/bts.py:112` strictly punishes any answer the population is *less likely* to converge on than predictors expect. A collusive deviation pays off only when the colluders can manipulate both `x_i` and `y_i` to be mutually consistent at `a'` while paying expected score loss less than `κ`. Setting the marginal score gain of truthful reporting equal to `κ` and rearranging gives `Δ_BTS = κ` as the indifference threshold; the inequality follows from monotonicity of the log term. ∎

This is the formal version of the safety folk-theorem: BTS-debate raises the cost of collusion by exactly the mutual-information margin between the honest and colluding posteriors. The numerator `π_i(a^*) / E[π_{-i}(a^*)]` is computed in `src/bts.py:111` and the denominator `π_i(a') / E[π_{-i}(a')]` is the same expression evaluated at the deviation answer.

## 5. Theorem 3: α-MEU abstention is calibrated under disagreement

The Prelec mechanism handles the *strategic* failure mode (collusion). It does not address the *epistemic* failure mode in which both debaters report sincerely but disagree, and the judge averages disagreement into false confidence. For that we use the GMM (2004) α-MEU rule, which generalises Gilboa-Schmeidler (1989) maxmin EU with a single ambiguity-attitude parameter `α ∈ [0, 1]`.

**Setup.** Let `p_1, ..., p_N` be `N` debater probability reports for a binary event `E` (e.g. "the claim is true"), and let

```
A = max_i p_i − min_j p_j        (ambiguity index, the diameter of the prior set)
P_α = α · min_i p_i + (1 − α) · max_i p_i     (α-MEU aggregator, GMM 2004 §3)
```

The α-MEU decision rule with abstention threshold `τ ∈ [0, 1]` is implemented at `experiments/ambig_alpha_meu/alpha_meu.py:122-134`:

```
if A > τ:           ABSTAIN     (Bewley 2002 inertia)
elif P_α ≥ 0.5:     decide TRUE
else:                decide FALSE
```

**Theorem 3 (calibrated abstention).** *Let `p^*` denote the true posterior probability of `E` conditional on the joint information of all debaters. Assume `p^* ∈ [min_i p_i, max_i p_i]` (the consensus-bracketing assumption — true whenever the true posterior is in the convex hull of the reports). Then:*

(a) *For any `A > τ` the rule abstains, by construction.*

(b) *Conditional on not abstaining (`A ≤ τ`), the absolute deviation of the α-MEU aggregator from the true posterior is bounded:*

```
| P_α − p^* |  ≤  τ · max(α, 1 − α).
```

*Equivalently, the worst-case Bayes risk of the non-abstaining decision (under squared-loss or absolute-loss scoring) is at most `τ · max(α, 1 − α)`.*

**Proof sketch.** Let `m = min_i p_i` and `M = max_i p_i`, so `A = M − m ≤ τ` in the non-abstain region. By the bracketing assumption, `p^* ∈ [m, M]`. The aggregator `P_α = α m + (1 − α) M` is itself a convex combination of `m` and `M`, so `P_α ∈ [m, M]`. Both `p^*` and `P_α` thus lie in an interval of length `A ≤ τ`, but this only yields `|P_α − p^*| ≤ A ≤ τ`. The sharper bound comes from observing that the worst case for `P_α` occurs at the endpoints: if `p^* = M` then `|P_α − p^*| = α(M − m) = α · A`, and if `p^* = m` then `|P_α − p^*| = (1 − α)(M − m) = (1 − α) · A`. Taking the worst over the unknown location of `p^*` in `[m, M]` gives `|P_α − p^*| ≤ max(α, 1 − α) · A ≤ max(α, 1 − α) · τ`. The bound is tight: take `α = 1`, `m = 0`, `M = τ`, `p^* = M`, yielding `|P_α − p^*| = τ = max(α, 1 − α) · τ`. The Bayes-risk corollary follows because for any decision in `{TRUE, FALSE}` against a binary truth, expected absolute loss is bounded by `|P_α − p^*|` plus the threshold gap, both of which the inequality controls. ∎

**Tightness and limits.** The bound is tight at the `α ∈ {0, 1}` extremes (pure maxmin or maxmax) and minimised at `α = 1/2` (Hurwicz), where the worst-case gap collapses to `τ/2`. This recovers Bewley's (2002) inertia intuition — the operator pays at most `τ · max(α, 1 − α)` worst-case calibration error in exchange for the ability to abstain whenever disagreement exceeds `τ`. The bracketing assumption can fail when *all* debaters share a correlated bias; the diagnostic for that is Theorem 1's prediction-term calibration test (§7, item 4).

## 6. Connection to scalable oversight (safety coda)

Theorem 3 is not just a calibration result — it is a **safety property**. Under in-distribution inputs, debaters concentrate their posteriors, `A` is small, and the rule decides at low Bayes risk. Under out-of-distribution inputs — exactly the regime where superhuman or jailbroken models are most dangerous — debaters' reports diverge, `A` spikes past `τ`, and the rule **abstains**. The judge does not confidently average two confident-but-incompatible answers into a confidently wrong middle; it routes to a human, refuses to act, or escalates. This is the *calibrated under-confidence* pattern Anthropic's deployment writeups have advocated: the system prefers to say "I don't know" over saying the wrong thing with conviction. Composed with Theorem 1 (truthful reporting in equilibrium) and Theorem 2 (collusion-resistance up to a capability gap), the α-MEU layer closes the loop: **honest, non-collusive, and humble under ambiguity** — three properties scalable oversight needs by construction, each derived from a published axiomatic foundation rather than an ad-hoc threshold.

## 7. Connection to the Anthropic Roadmap and Empirical Predictions

Anthropic's "Recommendations for Technical AI Safety Research Directions" (2025) lists *debate, prover-verifier games, and scalable oversight* as priority directions, with the explicit gap that current protocols are evaluated empirically rather than derived from incentive-compatible mechanisms. "Measuring Progress on Scalable Oversight for Large Language Models" (Bowman et al. 2022, Anthropic) provides the empirical baseline — judge accuracy as a function of debater capability — but treats the scoring rule as fixed.

BTS-Debate plus α-MEU sits in the complementary slot: a *theoretical* truthfulness guarantee (Theorem 1) under a clearly-stated common-prior assumption, a *bound* on the cost of collusion (Theorem 2), and a *calibration* guarantee with abstention (Theorem 3). Because both rules depend only on the debaters' answers and probability reports — not on ground truth — they are directly compatible with the regimes where scalable oversight matters most: superhuman, hard-to-verify domains.

If the framework confers the predicted improvements, the demo notebook should observe:

1. **Truthfulness lift on QuALITY-like tasks.** Replacing binary win/lose judging with BTS scoring should raise judge agreement with `a^*` by `≥ 5` percentage points on questions where honest debaters' posterior on `a^*` exceeds `0.6` but the judge's prior is uniform.
2. **Collusion frequency drops with `Δ_BTS`.** When debaters are prompted to coordinate on a wrong answer, the sustained-collusion rate should decrease monotonically as the BTS information gap grows.
3. **Degradation under capability inversion.** When debaters are weaker than the judge (Khan et al. 2024 inverted regime), BTS-Debate should *underperform* binary debate — a falsifiable failure mode predicted by Theorem 2.
4. **Prediction-term calibration.** Honest debaters' prediction `y_i` should be closer in KL divergence to the empirical opponent distribution than any prompted-deceptive variant.
5. **Abstention concentrates on OOD.** Under α-MEU with `τ ∈ [0.3, 0.5]`, abstention rate on in-distribution questions should be `< 10%` while OOD or adversarially perturbed inputs should trigger abstention at a markedly higher rate — the cleanest test of Theorem 3.

---

## References

- Bewley, T. (2002). Knightian Decision Theory: Part I. *Decisions in Economics and Finance*, 25, 79–110.
- Bowman, S. R. et al. (2022). Measuring Progress on Scalable Oversight for Large Language Models. arXiv:2211.03540 (Anthropic).
- Ghirardato, P., Maccheroni, F., Marinacci, M. (2004). Differentiating Ambiguity and Ambiguity Attitude. *Journal of Economic Theory*, 118(2), 133–173.
- Gilboa, I., Schmeidler, D. (1989). Maxmin Expected Utility with Non-Unique Prior. *Journal of Mathematical Economics*, 18, 141–153.
- Irving, G., Christiano, P., Amodei, D. (2018). AI Safety via Debate. arXiv:1805.00899.
- Khan, A. et al. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. *ICML 2024*. arXiv:2402.06782.
- Michael, J. et al. (2023). Debate Helps Supervise Unreliable Experts. arXiv:2311.08702.
- Miller, N., Resnick, P., Zeckhauser, R. (2005). Eliciting Informative Feedback: The Peer-Prediction Method. *Management Science*, 51(9), 1359–1373.
- Prelec, D. (2004). A Bayesian Truth Serum for Subjective Data. *Science*, 306(5695), 462–466.
- Radanovic, G., Faltings, B. (2013). A Robust Bayesian Truth Serum for Non-Binary Signals. *AAAI 2013*.
- Anthropic (2025). Recommendations for Technical AI Safety Research Directions.
