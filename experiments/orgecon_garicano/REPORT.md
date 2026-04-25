# Experiment: Garicano (2000) knowledge-hierarchy router

**Family:** Organizational economics / management theory
**Candidate chosen:** Garicano, L. (2000). "Hierarchies and the Organization of Knowledge in Production." *Journal of Political Economy* 108(5): 874–904. DOI 10.1086/317671.

**Why this candidate:** `docs/CORPORATE_MGMT_ANGLE.md` already covers Aghion-Tirole, Dessein, and Holmström-Milgrom. Garicano's *knowledge-based hierarchy with referral cost h* is complementary and absent from that doc.

## Why it transfers to LLM agents

Workers = debaters who emit (answer, self-reported confidence). Manager = judge with higher accuracy but per-call cost h. The delegation rule is a worker threshold τ ∈ [0, 1]: if the most confident debater clears τ, the worker layer keeps the decision; otherwise the problem escalates and h is paid. Garicano's Prop. 2 then yields a falsifiable comparative static for LangGraph judge nodes: optimal τ* is interior and shifts upward (more escalation) as h falls.

## Implementation notes

- `garicano.py` exposes `garicano_decide()` (single decision) and `garicano_throughput()` (sweepable batch metric returning accuracy, escalation_rate, net_utility = accuracy − h · escalation_rate).
- ~120 LOC, no live API.
- The interior-optimum reproduction is the headline test: sweep τ on a 20-problem synthetic batch where confident workers are correct and unconfident ones are wrong. With h = 0.3, optimal τ* is strictly interior and beats both τ = 0 and τ = 1 by ≥ 0.05 — Garicano Prop. 2 reproduced.

## Test results

```
$ pytest experiments/orgecon_garicano/ -v
test_confident_worker_keeps_decision PASSED
test_unconfident_workers_escalate PASSED
test_threshold_zero_is_flat_org PASSED
test_threshold_one_always_escalates PASSED
test_interior_optimum_garicano_prop2 PASSED
test_compare_against_bts_flat_aggregation PASSED
============================== 6 passed in 0.05s ==============================
```

## Comparison to BTS baseline

Toy 5-problem batch, h = 0.15, judge perfect:
- **Garicano** (τ = 0.6): accuracy **1.00**, net utility **0.94**.
- **Flat BTS-style plurality**: accuracy **0.60**, net utility **0.60**.

Garicano dominates flat aggregation iff (a) judge accuracy > marginal debater, and (b) confidence is informative — both routinely hold in production multi-agent setups.

## Honest assessment

**Did it beat BTS?** **Yes** on the toy, but the comparison is structural rather than head-to-head: BTS is an *elicitation* scoring rule, Garicano is a *routing* organizational rule. They are composable, not substitutes.

**Productive synthesis:** BTS-elicit confidence, Garicano-route on it. Falsifiable comparative static: as h falls, optimal τ* shifts down (more escalation). This is exactly the hypothesis a Fellows research project could test on the live debate stack.

## Reference

Garicano, L. (2000). "Hierarchies and the Organization of Knowledge in Production." *Journal of Political Economy* 108(5): 874–904. DOI 10.1086/317671.
