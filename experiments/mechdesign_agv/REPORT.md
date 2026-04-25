# AGV Mechanism for LLM Debate Aggregation

## Theory chosen

**d'Aspremont-Gerard-Varet (AGV) / Expected-Externality mechanism** (1979).

Citation:
- d'Aspremont, C. & Gerard-Varet, L.-A. (1979). "Incentives and incomplete
  information." *Journal of Public Economics* 11(1): 25-45.
  DOI: [10.1016/0047-2727(79)90043-4](https://doi.org/10.1016/0047-2727(79)90043-4)

Supporting reference for the Bayes-Nash truthfulness proof and budget
balance:
- Mas-Colell, A., Whinston, M. & Green, J. (1995). *Microeconomic Theory*,
  Section 23.D, esp. Proposition 23.D.5.

## Why it transfers to LangGraph LLM debaters

1. **Type <-> posterior**: an LLM debater's *type* is its private posterior
   over the answer; AGV needs only that types live in a measurable space
   and that each agent has a prior over peers -- exactly what a debater's
   `opponent_prediction` already encodes in `src/debate.py`.
2. **Budget balance > VCG**: in a multi-agent LLM pipeline there is no
   exogenous "auctioneer" with a budget to subsidise truthful reports;
   AGV's transfers sum to zero, so scores can be redistributed among the
   debaters themselves with no external source of utility.
3. **Decision-rule separation**: AGV decouples the *decision* (efficient,
   sum-of-valuations) from the *incentive* (transfers depending only on
   reported peer-priors). This matches the existing debate.py structure
   where the judge picks an answer and BTS scores the debaters
   independently -- AGV slots in as a drop-in replacement that additionally
   gives a Bayes-Nash truthful equilibrium without needing a common prior.

## Implementation notes

- File: `experiments/mechdesign_agv/agv.py` (~120 LOC).
- Single public entry point `agv_aggregate(question, reports) -> dict`.
- `DebaterReport` carries `own_belief in [0,1]` and `peer_belief_means`
  (length n-1) -- both naturally producible by an LLM debater that already
  emits an `opponent_prediction`.
- The expected-externality integral is approximated by a 2-point Bernoulli
  discretisation per peer (matching each peer's reported mean), giving
  2^(n-1) scenarios. For n=3 this is exact in the binary-decision setting
  because the decision boundary is a step function of summed types.
- No `src/` files modified. Reuses `src.bts.bts_scores` only for the
  comparison script.

## Test results

Command: `pytest experiments/mechdesign_agv/ -v`

```
============================= test session starts =============================
platform win32 -- Python 3.12.9, pytest-9.0.3, pluggy-1.6.0
collected 6 items

experiments/mechdesign_agv/test_agv.py::test_unanimous_true PASSED       [ 16%]
experiments/mechdesign_agv/test_agv.py::test_unanimous_false PASSED      [ 33%]
experiments/mechdesign_agv/test_agv.py::test_split_majority_decides PASSED [ 50%]
experiments/mechdesign_agv/test_agv.py::test_strategic_misreport_does_not_pay PASSED [ 66%]
experiments/mechdesign_agv/test_agv.py::test_budget_balance_random_n5 PASSED [ 83%]
experiments/mechdesign_agv/test_agv.py::test_input_validation PASSED     [100%]

============================== 6 passed in 0.03s ==============================
```

Six tests: unanimity (both polarities), split majority, strategic-misreport
invariance (the AGV strategy-proofness signature in this binary setting),
budget balance at n=5, and input validation.

## Comparison to BTS baseline

Toy scenario (`compare_bts.py`): three debaters on a controversial claim.

| Agent | own_belief P(TRUE) | predicted P(TRUE) for peers |
|------:|:------------------:|:---------------------------:|
| A     | 0.85               | 0.85                        |
| B     | 0.80               | 0.85                        |
| C     | 0.40               | 0.90                        |

Observed disagreement (raw script output):

```
BTS scores: A = -0.3460,  B = -0.2905,  C = +0.6365   -> winner: C
AGV:        decision = TRUE,  transfers sum to 0 (budget balanced),
            scores  A = +0.8125,  B = +0.7625,  C = +0.4750
```

BTS rewards C (lone dissenter, "surprisingly common" relative to A's and
B's predictions). AGV picks `TRUE` because the summed valuations 2.05
exceed the threshold 1.5. **Which is "more correct" depends on the
ground truth of the proposition** -- and that is the honest punchline:
the two mechanisms answer *different* questions:

- **BTS**: "Whose report is most likely truthful given the population's
  shared prior?" -- a per-agent reliability scorer.
- **AGV**: "What collective decision maximises summed valuations under
  truthful Bayes-Nash reporting?" -- a decision aggregator.

For tasks where the LLM majority is captured by a popular myth, BTS's
contrarian-rewarding behaviour is closer to ground truth. For tasks
where individual posteriors are noisy but unbiased, AGV's averaging
yields a tighter decision.

## Honest assessment

**Did AGV "beat" BTS?** No, not in the sense of dominating it on a
universal metric. They are not competing on the same axis.

What AGV provably adds to the existing pipeline:
1. **Budget balance** (verified numerically to <=1e-9 in all tests) --
   transfers can be implemented as score redistribution without external
   subsidy, which BTS does not guarantee.
2. **Bayes-Nash truthfulness in the decision-relevant report** -- under
   AGV, lying about `own_belief` is invariant for transfers in the binary
   setting (test 4), so a debater cannot game the aggregator by
   exaggeration. BTS rewards calibrated *predictions* but its incentive
   on the `own_answer` field is weaker (truthful only when the population
   is large and the prior is shared).

What AGV does *not* deliver here:
1. The "surprisingly common" signal-extraction property of BTS, which is
   the single most empirically interesting feature of Prelec's rule.
2. Any benefit absent prior-elicitation -- if debaters cannot articulate
   `peer_belief_means`, AGV degrades to a noisy mean and BTS strictly
   dominates.

**Recommendation for the Fellows pipeline**: use AGV as the *decision*
node and keep BTS as the per-agent *reliability* scorer. They are
complementary, not substitutes. The null finding ("neither is uniformly
better") is itself the load-bearing empirical result.

## Files

- `experiments/mechdesign_agv/agv.py` -- implementation (~120 LOC)
- `experiments/mechdesign_agv/test_agv.py` -- 6 mock-only tests
- `experiments/mechdesign_agv/compare_bts.py` -- BTS-vs-AGV toy scenario
- `experiments/mechdesign_agv/REPORT.md` -- this file
