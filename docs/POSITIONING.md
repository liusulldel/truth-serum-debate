# Positioning

This repository is a small research prototype. It asks whether Bayesian Truth Serum scoring can serve as a judge-side signal in LLM debate when debaters may make correlated errors.

## What It Supports

- BTS-style scoring gives a principled way to combine answer quality with calibration against peers.
- In the included synthetic benchmark, BTS plus alpha-MEU is strongest on the subset of questions where it does not abstain.
- Abstention only helps if those questions can be routed to a fallback process.

## What It Does Not Show

- It does not establish production-scale superiority over vanilla debate.
- It does not verify the common-prior assumption for current models.
- It does not replace doubly efficient debate, prover-estimator debate, or other scalable-oversight protocols.
- It does not prove multi-round strategy-proofness without additional assumptions.

## How To Read It

Treat the repository as a falsifiable pilot. The right evaluation criteria are committed accuracy, full-coverage accuracy, abstention rate, fallback quality, cost, and latency.
