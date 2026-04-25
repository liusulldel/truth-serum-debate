# Experiment: Holmström (1982) team-incentive contract

**Family:** Contract theory / principal-agent
**Candidate chosen:** Holmström, B. (1982). "Moral Hazard in Teams." *Bell Journal of Economics* 13(2): 324–340. DOI 10.2307/3003457 / JSTOR 3003457.

**Headline result reproduced:** under budget-balanced sharing of joint output, each of n agents receives only 1/n of marginal payoff (free-riding). Adding a *budget-breaker* — a residual claimant outside the team paying a full bonus on group success — restores first-best effort. Test `test_holmstrom_theorem_budget_breaker_restores_effort` reproduces the 1/n inequality parametrically for n ∈ {2, 3, 5, 10}.

## Why it transfers to LLM agents

BTS in `src/debate.py` distributes a fixed-sum information score between two debaters — A's gain is mechanically B's loss — which is exactly the budget-balanced regime Holmström rules out as inefficient. Treating the **judge as budget-breaker** (paying each debater a full *external* bonus iff their report agrees with a verdict the judge certifies as correct) makes the contract no longer budget-balanced, and Holmström predicts strictly higher equilibrium effort.

## Implementation notes

- ~140 LOC in `team_contract.py`. `forcing_contract(reports, verdict, ground_truth, bonus, cost)` returns per-agent net payoff, no LLM calls.
- The "judge as residual claimant" assumption requires that the judge's verdict be a sufficient statistic for ground truth — when it is not, the forcing contract pays for noise. Documented as the main caveat.
- Sibling `compare_to_bts.py` runs a 3-debater Riemann-hypothesis mock head-to-head.

## Test results

```
$ pytest experiments/contract_holmstrom_teams/ -v
test_forcing_contract_pays_correct_team PASSED                              [ 10%]
test_forcing_contract_no_pay_when_wrong PASSED                              [ 20%]
test_forcing_contract_no_pay_when_truly_wrong PASSED                        [ 30%]
test_budget_balanced_splits_pot PASSED                                      [ 40%]
test_dissenter_unpaid PASSED                                                [ 50%]
test_holmstrom_theorem_budget_breaker_restores_effort[2] PASSED             [ 60%]
test_holmstrom_theorem_budget_breaker_restores_effort[3] PASSED             [ 70%]
test_holmstrom_theorem_budget_breaker_restores_effort[5] PASSED             [ 80%]
test_holmstrom_theorem_budget_breaker_restores_effort[10] PASSED            [ 90%]
test_net_payoff_subtracts_cost PASSED                                       [100%]
============================== 10 passed in 0.06s ==============================
```

## Comparison to BTS baseline

3-debater "Riemann hypothesis proven?" mock with 1 truthful dissenter (C) and 2 confidently-wrong agreers (A, B):
- **BTS** picks correctly: A=+0.789, B=−0.443, C=−0.346 → verdict **false** (correct).
- **Holmström forcing contract** with naive confidence-weighted majority: verdict **true** (wrong).

But the contract still **refuses to pay the wrong-consensus team**, so the truthful dissenter's effort cost is not subsidising the noisy majority — the *incentive guarantee* holds even when the *aggregator* fails.

## Honest assessment

**Did it beat BTS?** **No** on aggregation accuracy in the toy. But the comparison is unfair: BTS is a *scoring rule* (post-hoc surprise correction handling correlated bias), Holmström-1982 is an *ex-ante contract* (effort elicitation via residual-claimant funding). They solve different problems.

**Productive synthesis:** the natural next experiment is a **hybrid** — Holmström forcing contract to elicit effort, BTS as the aggregator inside the contract's correctness check. Neither piece dominates alone.

## Reference

Holmström, B. (1982). "Moral Hazard in Teams." *Bell Journal of Economics* 13(2): 324–340. DOI 10.2307/3003457.
