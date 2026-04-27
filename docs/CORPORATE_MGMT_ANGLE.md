# Org-Econ Framing for Multi-Agent LLM Systems

This note maps LangGraph-style orchestration to standard ideas in organizational economics. It is a conceptual bridge, not an empirical claim about any specific system.

## Core Mapping

- Node -> role or agent with a specified responsibility.
- Edge -> communication or delegation rule.
- State -> shared information.
- Conditional edge -> contingency rule.
- Tool call -> access to an external resource.
- Subgraph -> department or subsidiary.
- Human checkpoint -> retained principal authority.

Once that dictionary is in place, design questions such as "should this be one node or three?" or "should the planner have veto power?" become familiar org-design questions.

## Why It Matters

- Scalable oversight is a principal-agent problem with a capability gap.
- Debate protocols are a form of adversarial information aggregation under a third-party decision-maker.
- Multi-agent collusion is a contract-design problem, not just a prompting problem.
- Reward hacking is a multi-task incentive problem.

The BTS-debate protocol in this repository is one concrete instance of that last point: the scoring rule changes which reports are informative.

## Practical Patterns

- RAG-with-verification maps to an internal-controls or audit pattern.
- Multi-agent debate maps to an adversarial litigation pattern.
- Human-in-the-loop review maps to retained formal authority with delegated day-to-day execution.

## Open Questions

- When does topology itself predict failure modes?
- When does delegation help, and when does it simply add coordination cost?
- Which mechanisms remain robust when a third colluding agent enters the picture?

## References

- Aghion, P. & Tirole, J. (1997). Formal and Real Authority in Organizations. *Journal of Political Economy* 105(1), 1-19.
- Baker, G. (2002). Distortion and Risk in Optimal Incentive Contracts. *Journal of Human Resources* 37(4), 728-751.
- Dessein, W. (2002). Authority and Communication in Organizations. *Review of Economic Studies* 69(4), 811-838.
- Dewatripont, M. & Tirole, J. (1999). Advocates. *Journal of Political Economy* 107(1), 1-39.
- Holmstrom, B. (1979). Moral Hazard and Observability. *Bell Journal of Economics* 10(1), 74-91.
- Holmstrom, B. (1982). Moral Hazard in Teams. *Bell Journal of Economics* 13(2), 324-340.
- Holmstrom, B. & Milgrom, P. (1991). Multitask Principal-Agent Analyses. *JLEO* 7, 24-52.
- Irving, G., Christiano, P., Amodei, D. (2018). AI Safety via Debate. arXiv:1805.00899.
- Khan, A., Hughes, J., Valentine, D., et al. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. ICML 2024.
- Kydland, F. & Prescott, E. (1977). Rules Rather than Discretion. *Journal of Political Economy* 85(3), 473-491.
- Laffont, J.-J. & Martimort, D. (1997). Collusion under Asymmetric Information. *Econometrica* 65(4), 875-911.
- Prelec, D. (2004). A Bayesian Truth Serum for Subjective Data. *Science* 306, 462-466.
- Tirole, J. (1986). Hierarchies and Bureaucracies. *JLEO* 2(2), 181-214.
