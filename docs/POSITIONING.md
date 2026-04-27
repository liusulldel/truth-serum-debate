# Positioning

This repository is a small research prototype. It asks whether Bayesian Truth
Serum scoring can be useful as a judge-side signal in LLM debate when debaters
may make correlated errors.

## What It Claims

- BTS-style scoring gives a principled way to reward reports that are both
  individually stated and predictively calibrated against peers.
- In the included synthetic benchmark, BTS plus alpha-MEU has higher accuracy on
  the subset of questions where it chooses not to abstain.
- The abstention result is useful only if abstained questions can be routed to a
  stronger model, a human reviewer, or another review process.

## What It Does Not Claim

- It does not show that BTS-scored debate beats vanilla debate at scale on real
  production tasks.
- It does not show that current Claude models satisfy the common-prior
  assumption used in the theory note.
- It does not replace doubly efficient debate, prover-estimator debate, or other
  scalable-oversight protocols.
- It does not prove multi-round strategy-proofness without additional
  assumptions.

## Best Reading

The contribution is a compact bridge between peer prediction, debate, and
coverage-aware evaluation. The repository is most useful as a falsifiable pilot:
the mechanism should be judged by committed accuracy, full-coverage accuracy,
abstention rate, fallback quality, cost, and latency.
