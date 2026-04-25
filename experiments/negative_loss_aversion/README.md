# negative_loss_aversion — empirical sketch

Companion to `experiments/negative_results/THREE_FAILURES.md`, Failure 1.

**Claim.** Stateless LLM debaters do not exhibit Kahneman-Tversky loss aversion
because they have no realized consumption and no reference-point endowment.
A loss-framed prompt is processed as text, not as an impending hedonic loss.

**Design.** 50 questions x 3 paraphrases per frame x 2 frames (gain, loss).
Outcome = "hedge score" (proxy for response conservatism). The demo
ships with a *mock* stateless LLM and a *mock* lambda=2.25 human as a
reference point, so the expected pattern is visible without burning API
credits. To run a real experiment, replace `mock_llm_debater` with an
Anthropic SDK call and re-run.

**Run.**

```
python demo_loss_aversion.py
```

**Expected output.** The mock LLM shows loss-minus-gain of order 0.00–0.02
(within noise). The mock human shows ~0.15. If a real Claude/GPT replicate
matches the LLM null, that is empirical confirmation that loss-aversion-based
contracts are a non-transfer for current agents.

**Why this matters for the application.** Adding penalty language to debate
prompts is a folk move; the negative result tells you not to bother — the
incentive lever you actually want is BTS-style probabilistic scoring, where
the proof goes through on Bayesian-posterior assumptions rather than
hedonic-utility assumptions.
