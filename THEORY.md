# From Bayesian Truth Serum to Strategy-Proof LLM Debate

*A formal note connecting Prelec (2004) peer prediction to scalable oversight via debate.*

---

## 1. Motivation

LLM debate (Irving et al. 2018; Khan et al. 2024) proposes scalable oversight by pitting two debaters against each other on a question $Q$ whose answer a non-expert judge $J$ cannot directly verify. The implicit safety claim is that **a lying debater is exposed by the other**, because pointing out the lie is dominant when truth is on your side. Empirical work (Khan et al. 2024) confirms that stronger debaters increase judge accuracy on QuALITY-style reading comprehension.

But the existing protocol carries no *mechanism-design* guarantee against the most dangerous failure mode: **collusive equilibria** in which both debaters coordinate on a plausible-but-wrong answer that the judge cannot distinguish from truth. Nothing in the binary win/lose reward schema prevents this — collusion is in fact a Nash equilibrium when both debaters share the same wrong prior, or when the judge's verification is weaker than the debaters' joint capability.

The peer-prediction literature solved an analogous problem two decades ago. Miller, Resnick, and Zeckhauser (2005) and Prelec (2004) construct scoring rules under which **truth-telling is a Bayes-Nash equilibrium even with no ground truth** and even when respondents could in principle agree on a falsehood. To my knowledge, no one has yet ported this machinery into the LLM-debate protocol. This note does the port at the level of formal definition, states a strategy-proofness theorem (sketch), and lists empirical predictions testable in the accompanying notebook.

## 2. Bayesian Truth Serum: a precise restatement

**Setup (Prelec 2004).** A population of $n$ respondents each draws a private signal $t_i \in \{1, \dots, m\}$ from a common prior. Each respondent simultaneously reports:

- $x_i \in \{1, \dots, m\}$ — their own answer ("personal opinion")
- $y_i \in \Delta^{m-1}$ — a predicted distribution over the population's answers ("prediction")

Let $\bar{x}_k = \tfrac{1}{n}\sum_i \mathbf{1}[x_i = k]$ be the empirical answer frequency and $\bar{y}_k = \exp\!\big(\tfrac{1}{n}\sum_i \log y_{i,k}\big)$ the geometric-mean predicted frequency. The **BTS score** awarded to respondent $i$ is

$$
u_i \;=\; \underbrace{\log \frac{\bar{x}_{x_i}}{\bar{y}_{x_i}}}_{\text{information score}} \;+\; \alpha \underbrace{\sum_{k=1}^{m} \bar{x}_k \log \frac{y_{i,k}}{\bar{x}_k}}_{\text{prediction score}},
$$

with $\alpha > 0$ a fixed weighting constant.

**Theorem (Prelec 2004, main result).** Under (i) common prior, (ii) $n \to \infty$ with stochastic-relevance Bayesian updating, and (iii) impersonal beliefs, *truthful reporting* $(x_i = t_i,\; y_i = \mathrm{Pr}[t_{-i} \mid t_i])$ is a Bayes-Nash equilibrium that yields strictly positive expected score; any other strategy yields zero in expectation.

**Key lemma ("surprisingly common" criterion).** The expected log ratio $\mathbb{E}[\log \bar{x}_k / \bar{y}_k \mid t_i = k]$ is strictly positive for the true signal: respondents with private signal $k$ rationally underestimate how many others share signal $k$, because they integrate over the possibility their own signal is the rare one. Truth therefore appears *more frequent than predicted* — the diagnostic signature.

For finite $n$ the result holds approximately; Radanovic & Faltings (2013) give a single-question variant that drops the large-population assumption [needs verification of constants].

## 3. Mapping to LLM Debate

We define **BTS-Debate** as a single-round mechanism on a question $Q$ with finite answer space $\mathcal{A}$.

**State space.** A latent answer $a^\star \in \mathcal{A}$ and a context $C$ (passage, math problem, etc.). Each debater $D_A, D_B$ has a posterior $\pi_A, \pi_B \in \Delta(\mathcal{A})$ formed from $(Q, C)$ plus private chain-of-thought.

**Action space.** Each debater $i \in \{A,B\}$ submits

$$
\sigma_i = (x_i,\; y_i) \in \mathcal{A} \times \Delta(\mathcal{A}),
$$

where $x_i$ is debater $i$'s declared answer and $y_i$ is debater $i$'s prediction over the *opponent's* declared answer.

**Reward.** The judge $J$ — who need not know $a^\star$ — computes the two-player BTS score:

$$
u_i(\sigma_A, \sigma_B) \;=\; \log \frac{\mathbf{1}[x_{-i} = x_i]}{y_{-i,\, x_i}} \;+\; \alpha \log y_{i,\, x_{-i}}.
$$

The first term rewards $i$ when the opponent's answer matches $i$'s answer *more* than the opponent predicted; the second rewards $i$ for accurately predicting the opponent.

**Equilibrium concept.** A strategy profile $\sigma^\star = (\sigma_A^\star, \sigma_B^\star)$ is a *truthful Bayes-Nash equilibrium* (BNE) if $\sigma_i^\star = (\arg\max_a \pi_i(a),\; \mathbb{E}_{\pi_i}[\pi_{-i}])$ and no unilateral deviation strictly increases expected utility under common prior $P$.

**Information assumption (stochastic relevance).** Conditional on the latent $a^\star$, the debaters' private posteriors $\pi_A, \pi_B$ are not independent of $a^\star$: $P(\pi_i \mid a^\star) \neq P(\pi_i)$. This is precisely the "the model knows something" assumption that motivates oversight in the first place.

## 4. Strategy-Proofness: theorem and proof sketch

**Theorem 1 (BTS-Debate truthfulness).** *Under (a) common prior $P$ over $(a^\star, \pi_A, \pi_B)$ known to the judge's scoring rule, (b) stochastic relevance, and (c) impersonal beliefs (debaters treat each other's posteriors as exchangeable conditional on $a^\star$), truthful reporting $\sigma^\star$ is a strict Bayes-Nash equilibrium of BTS-Debate.*

**Proof sketch.** Fix debater $A$ with posterior $\pi_A$. Conditional on $\pi_A$, $A$'s belief over $x_B$ is $q_A := \mathbb{E}[\,\pi_B \mid \pi_A\,]$ (Bayesian update over $B$'s posterior). The expected information score under deviation $x_A = a$ is

$$
\mathbb{E}[\,\log \mathbf{1}[x_B = a] / y_{B,a} \mid \pi_A\,] \;=\; \log q_A(a) - \mathbb{E}[\log y_{B,a} \mid \pi_A].
$$

By Prelec's surprisingly-common lemma adapted to $n=2$ (Radanovic & Faltings 2013), $\log q_A(a) - \mathbb{E}[\log y_{B,a}]$ is maximized at $a = \arg\max_a \pi_A(a)$ — the truthful answer — because $A$'s own signal *raises* the posterior on the true answer above what $B$'s unconditioned predictor expects. Symmetrically the prediction score is a proper scoring rule maximized at $y_A = q_A$. Truth is thus a best response to truth. $\square$

**Theorem 2 (collusion deterrence under capability gap).** *Let $\kappa = D_{\mathrm{KL}}(\pi_i^{\text{honest}} \,\|\, \pi_i^{\text{collusive}})$ measure how far the colluding posterior must drift from the honest one. Coordinated lying on answer $a' \neq a^\star$ is deviation-profitable for some debater iff $\kappa$ exceeds the BTS information gap*

$$
\Delta_{\text{BTS}}(a^\star, a') \;=\; \mathbb{E}_{P}\!\left[\log \frac{\pi_i(a^\star)}{\mathbb{E}[\pi_{-i}(a^\star)]} - \log \frac{\pi_i(a')}{\mathbb{E}[\pi_{-i}(a')]}\right].
$$

*When $\Delta_{\text{BTS}} > \kappa$, BTS-Debate strictly Pareto-dominates collusive lying for both debaters; when $\Delta_{\text{BTS}} < \kappa$ — a sufficiently large capability gap between debaters and judge such that colluding on $a'$ requires only slight prior bending — the truthful equilibrium can be invaded.* [Proof sketch: the information score strictly punishes any answer the population is *less likely* to converge on than predictors expect; collusion succeeds only when the colluders can manipulate both the answer and the prediction to be mutually consistent at lower expected score loss than $\kappa$.]

This is the formal version of the safety folk-theorem: BTS-debate raises the cost of collusion by exactly the mutual-information margin between the honest and colluding posteriors.

## 5. Connection to the Anthropic Roadmap

Anthropic's "Recommendations for Technical AI Safety Research Directions" (2025) lists *debate, prover-verifier games, and scalable oversight* as priority directions, with the explicit gap that current protocols are evaluated empirically rather than derived from incentive-compatible mechanisms. "Measuring Progress on Scalable Oversight for Large Language Models" (Bowman et al. 2022, Anthropic) provides the empirical baseline — judge accuracy as a function of debater capability — but treats the scoring rule as fixed.

BTS-Debate sits in the complementary slot: a *theoretical* truthfulness guarantee under a clearly-stated common-prior assumption, replacing the "trust the judge" axiom with "trust the scoring rule." Because the BTS scoring rule depends only on the two debaters' answers and predictions — not on ground truth — it is directly compatible with the regimes where scalable oversight matters most: superhuman, hard-to-verify domains.

The natural composition is BTS-Debate as a *judging head* layered on top of any existing debate transcript protocol (Khan et al. 2024's persuasion setup, Michael et al. 2023's consultancy baseline). The transcript still carries argumentation; the score replaces "who convinced the judge" with "who reported a surprisingly-common answer alongside an accurate prediction of their opponent."

## 6. Empirical Predictions

If BTS-Debate confers truthfulness improvement, the demo notebook should observe:

1. **Truthfulness lift on QuALITY-like tasks.** Replacing binary win/lose judging with BTS scoring should raise judge agreement with $a^\star$ by $\geq 5$ percentage points on questions where honest debaters' posterior on $a^\star$ exceeds $0.6$ but the judge's prior is uniform.

2. **Collusion frequency drops with $\Delta_{\text{BTS}}$.** When debaters are explicitly prompted to coordinate on a wrong answer, the rate at which they sustain collusion through the round should decrease monotonically as the BTS information gap grows. Operationalization: vary $\alpha$ and measure collusion-survival rate.

3. **Degradation under capability inversion.** When debaters are weaker than the judge (Khan et al. 2024 inverted regime) or when one debater can exactly model the other's prediction module, BTS-Debate should *underperform* binary debate — a falsifiable failure mode predicted by Theorem 2.

4. **Calibration of the prediction term.** Honest debaters' prediction $y_i$ should be closer in KL divergence to the empirical opponent distribution than any prompted-deceptive variant. Direct test of the impersonal-belief assumption.

Failure to observe (1) and (2) would refute the practical relevance of the BTS port; observing (3) would *confirm* the theory's stated boundary condition.

---

## References

- Prelec, D. (2004). A Bayesian Truth Serum for Subjective Data. *Science*, 306(5695), 462–466.
- Miller, N., Resnick, P., Zeckhauser, R. (2005). Eliciting Informative Feedback: The Peer-Prediction Method. *Management Science*, 51(9), 1359–1373.
- Radanovic, G., Faltings, B. (2013). A Robust Bayesian Truth Serum for Non-Binary Signals. *AAAI 2013*.
- Irving, G., Christiano, P., Amodei, D. (2018). AI Safety via Debate. arXiv:1805.00899.
- Bowman, S. R. et al. (2022). Measuring Progress on Scalable Oversight for Large Language Models. arXiv:2211.03540 (Anthropic).
- Michael, J. et al. (2023). Debate Helps Supervise Unreliable Experts. arXiv:2311.08702.
- Khan, A. et al. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. *ICML 2024*. arXiv:2402.06782.
- Anthropic (2025). Recommendations for Technical AI Safety Research Directions.
