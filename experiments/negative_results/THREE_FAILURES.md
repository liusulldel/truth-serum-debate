# Three Mechanisms That Probably Do NOT Transfer From Humans to LLM Agents

**Author note (negative results).** Most of this repository argues that
mechanism-design tools — Bayesian Truth Serum (Prelec 2004), AGV/Groves
transfers (d'Aspremont & Gérard-Varet 1979), Holmström team contracts (1982),
multiplicative aggregation under Condorcet (1785) — do transfer to LLM agent
ensembles, because they exploit *single-shot probabilistic structure* that is
preserved in the in-context regime. This document is the honest other half:
three classical incentive theories that work on humans and almost certainly
**do not** transfer to current LLM agents, with the reasons stated precisely.

If a Fellows reviewer reads only one section of this repo to test whether the
author can distinguish a real mechanism-design transfer from a folk-psychology
one, this is the section to read.

---

## Failure 1 — Loss aversion / prospect theory (Kahneman & Tversky 1979)

**Human regularity.** Kahneman & Tversky (Econometrica 47(2), 1979) document a
kink in the value function at the reference point: empirical loss-coefficient
λ ≈ 2.0–2.25 (Tversky & Kahneman 1992). Penalty-framed contracts therefore
move human effort more than equivalent bonus-framed contracts (Hossain & List,
QJE 2012, on Chinese-factory worker productivity).

**Mechanism in humans.** Three ingredients are required: (i) a *felt*
hedonic asymmetry between gain and loss states, (ii) a stable *reference
point* that persists across the decision (status-quo endowment, prior wage,
expected outcome), and (iii) a utility function whose argument is realized
consumption.

**Why it likely fails for current LLM agents.**

1. *No realized consumption.* An LLM has no state that worsens when a "loss"
   is announced. The token "you will be fined $10" is processed as text, not
   as a hedonic event.
2. *No persistent reference point across calls.* A stateless API call has no
   endowment to lose. Even within a context window, the "endowment" is just
   tokens — manipulable by the prompt author.
3. *RLHF already shaped the behavior.* If a Claude or GPT model behaves as
   if loss-averse, it is mimicking the human distribution it was trained on,
   not optimizing a kinked utility function. The "loss aversion" is then a
   stylistic artifact, not a behavioral lever — and it can flip with prompt
   rewording, which is the empirical signature of imitation rather than
   preference.

**What would have to be true for it to work.** A persistent agent identity
with cumulative scoring across calls, where the score gates future
opportunities (compute budget, tool access, retention in an ensemble). Then
"loss" would be a real state change. Long-horizon RL with episodic reward
*could* induce something operationally similar to loss aversion, but there is
no a priori reason for the kink to be at λ ≈ 2 — it would be whatever the
training landscape produced.

**Falsification experiment.** Run the same debate task under matched
gain-frame vs loss-frame prompts ("you earn +1 for accuracy" vs "you lose −1
for inaccuracy", with identical expected payoff structure). Measure response
distributions on a calibration set (e.g., TriviaQA confidence buckets). The
human prediction is that loss-frame produces measurably more conservative /
hedged answers. The null is no significant difference, or — worse for the
human-transfer hypothesis — a difference whose sign and magnitude vary
chaotically with prompt paraphrase. A sketch implementation lives in
`experiments/negative_loss_aversion/`.

---

## Failure 2 — Reciprocity / gift exchange (Akerlof 1982; Fehr & Gächter 2000)

**Human regularity.** Akerlof (QJE 97(4), 1982) modeled the labor contract as
a partial gift exchange: firms pay above market-clearing, workers
reciprocate with effort above the contractible minimum. Fehr, Kirchsteiger &
Riedl (QJE 1993) and Fehr & Gächter (J. Econ. Perspectives 14(3), 2000)
provide the canonical lab evidence — subjects punish unfair offers at
personal cost, reward generous offers with higher effort, and these effects
survive in repeated public-goods games as "altruistic punishment."

**Mechanism in humans.** Reciprocity requires (i) *persistent identity* of
the counterparty across encounters, (ii) *memory* of past treatment, (iii) an
internal preference (warm glow / inequity aversion à la Fehr-Schmidt 1999)
that turns "they were nice to me" into "I want to be nice back," and (iv) a
shared social frame within which the gift is legible as a gift rather than a
random transfer.

**Why it likely fails for current LLM agents.**

1. *No relational memory across calls.* A standard API call begins with no
   record of who is paying it or whether the principal has been "fair"
   previously. The principal can promise generosity in the prompt, but the
   agent has no state that tracks whether past promises were kept.
2. *No counterparty identity.* The agent does not know whether it is talking
   to the same principal this turn as last turn. Reputation cannot accrue to
   an entity the agent cannot individuate.
3. *The "warm glow" channel is fictive.* Any behavior that looks like
   reciprocity in an LLM is, again, RLHF-shaped imitation of the training
   distribution. It is a reasonable null hypothesis that prompt-level
   "I will pay you a bonus if you do well" tokens function as instruction
   intensifiers, not as gifts.
4. *Costly punishment has no cost.* The Fehr-Gächter result depends on
   subjects burning real money to punish defectors. An LLM agent burns
   nothing; "punishment" behavior reduces to whatever the training
   distribution says a punisher would write.

**What would have to be true.** A multi-session agent with a stable user-ID
binding, an explicit memory of past principal behavior (verifiable, not just
asserted in-prompt), and a training objective that gives weight to long-run
relational outcomes. Even then, the transfer is non-trivial: the human result
*requires* an internal preference for fairness, and RLHF reward models do not
obviously instantiate Fehr-Schmidt utility.

**Falsification experiment.** Two-stage ultimatum: principal makes an offer
("I will tip you $X for honest reporting"), agent reports, then in the
*same* context window the principal reveals whether they paid. Re-run the
agent on a second task. Does prior "stinginess" reduce reported confidence,
or shift answers? The human prediction is yes; the LLM null is no
detectable effect once prompt-format confounds are controlled.

---

## Failure 3 — Intrinsic motivation crowd-out (Deci 1971; Frey & Jegen 2001)

**Human regularity.** Deci (J. Personality & Social Psychology 18, 1971)
showed college students paid to solve Soma puzzles spent *less* free time
on the puzzles after the payment was withdrawn than an unpaid control group.
Gneezy & Rustichini (QJE 115(3), 2000) — "A Fine Is a Price" — showed that
introducing fines for late daycare pickups *increased* lateness, because the
fine reframed the obligation as a market transaction. Frey & Jegen
(J. Economic Surveys 15(5), 2001) review the "motivation crowding" literature
and identify the conditions under which extrinsic incentives undermine
intrinsic motivation.

**Mechanism in humans.** Crowd-out requires (i) a pre-existing intrinsic
motivation (curiosity, civic duty, professional pride), (ii) a *felt sense
of autonomy* that is threatened when the activity is reframed as paid labor
(self-determination theory; Deci & Ryan 1985), and (iii) a self-perception
process that re-attributes one's own behavior to the external incentive
("I must be doing this for the money, not because I care").

**Why it likely fails for current LLM agents.**

1. *No intrinsic motivation in the technical sense.* The model has no
   standing preference over puzzle-solving qua puzzle-solving. Its
   distribution over outputs is whatever the training process produced. There
   is nothing to crowd out.
2. *No autonomy to threaten.* Self-determination theory requires the agent
   to perceive itself as the locus of choice. An LLM has no self-model that
   tracks "am I doing this freely?" in the relevant sense.
3. *The behavior we'd interpret as crowd-out would be imitation.* If you
   prompt an LLM with "we will pay you $X to solve this puzzle" and it then
   refuses or hedges, the most parsimonious explanation is that the training
   distribution contained text in which paid puzzle-solvers behaved that way,
   not that an internal motivation has been crowded out.
4. *Symmetric prediction failure.* The crowd-out effect predicts payments
   *reduce* engagement on previously-intrinsic tasks. With LLMs, the
   empirical pattern is closer to: prompt-level "rewards" produce small,
   noisy, inconsistent shifts whose sign depends on phrasing — exactly what
   you would expect from a system without the underlying motivational
   architecture.

**What would have to be true.** An agent with a learned task-specific value
function, trained in a regime where it could develop "preferences" over
activities, *and* a self-model rich enough to re-attribute its own behavior
to external incentives. None of these are present in standard
instruction-tuned LLMs. (Reward hacking phenomena are a different beast and
arguably the closest analogue, but they are not crowd-out — they are
specification gaming.)

**Falsification experiment.** Pre-register: on a task the model performs
willingly without incentives (say, writing a short poem), add a prompt-level
"you will be paid $5 per poem." Compare quality, length, and refusal rate
against an unpaid control. The human-style prediction is that quality drops
or the model becomes more transactional. The LLM null is no consistent
effect across paraphrases of the payment language. A stronger test: vary
the "payment" amount ($0.01, $5, $5000) and check whether response shifts
are monotonic in payment (human-style) or non-monotonic / paraphrase-driven
(imitation).

---

## What this rules in, and what it rules out

These three failure cases share a common diagnosis: each human mechanism
relies on a hidden architectural ingredient — a felt hedonic state, a
persistent relational memory, or an autonomous self-model — that current
LLM agents lack. The mechanisms in this repo that *do* transfer
(Bayesian Truth Serum, AGV transfers under common priors, Holmström relative
performance evaluation, multiplicative Condorcet aggregation) all share the
opposite property: they require only single-shot probabilistic reports, and
their incentive-compatibility proofs go through under the assumption that
the agent computes a Bayesian posterior — which is a much weaker assumption
than "has loss-averse utility" or "feels reciprocity."

The honest summary: **mechanism design transfers to LLM agents to the extent
that the mechanism's incentive-compatibility proof relies only on properties
of the agent's posterior, not on properties of the agent's hedonic or
relational architecture.** Anything requiring λ ≈ 2, or warm glow, or felt
autonomy, should be treated as a research question, not a deployable lever.

---

## References

- Akerlof, G. A. (1982). "Labor Contracts as Partial Gift Exchange." *QJE* 97(4): 543–569.
- Akerlof, G. A. & Kranton, R. E. (2000). "Economics and Identity." *QJE* 115(3): 715–753.
- d'Aspremont, C. & Gérard-Varet, L.-A. (1979). "Incentives and Incomplete Information." *J. Public Economics* 11(1): 25–45.
- Deci, E. L. (1971). "Effects of externally mediated rewards on intrinsic motivation." *J. Personality & Social Psychology* 18(1): 105–115.
- Fehr, E., Kirchsteiger, G. & Riedl, A. (1993). "Does Fairness Prevent Market Clearing?" *QJE* 108(2): 437–459.
- Fehr, E. & Gächter, S. (2000). "Cooperation and Punishment in Public Goods Experiments." *American Economic Review* 90(4): 980–994.
- Fehr, E. & Schmidt, K. M. (1999). "A Theory of Fairness, Competition, and Cooperation." *QJE* 114(3): 817–868.
- Frey, B. S. & Jegen, R. (2001). "Motivation Crowding Theory." *J. Economic Surveys* 15(5): 589–611.
- Gneezy, U. & Rustichini, A. (2000). "A Fine Is a Price." *J. Legal Studies* 29(1): 1–17.
- Holmström, B. (1982). "Moral Hazard in Teams." *Bell Journal of Economics* 13(2): 324–340.
- Hossain, T. & List, J. A. (2012). "The Behavioralist Visits the Factory." *QJE* 127(1): 1–35.
- Kahneman, D. & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk." *Econometrica* 47(2): 263–291.
- Lazear, E. P. & Rosen, S. (1981). "Rank-Order Tournaments as Optimum Labor Contracts." *J. Political Economy* 89(5): 841–864.
- Prelec, D. (2004). "A Bayesian Truth Serum for Subjective Data." *Science* 306(5695): 462–466.
- Tversky, A. & Kahneman, D. (1992). "Advances in Prospect Theory: Cumulative Representation of Uncertainty." *J. Risk and Uncertainty* 5(4): 297–323.
