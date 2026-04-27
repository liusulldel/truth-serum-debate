# Project Assessment

This is a public-facing assessment of the repository's current maturity.

## Strengths

- The benchmark is small and reproducible.
- The repo includes negative results and regimes where the mechanism is not
  useful.
- The theory note states the common-prior assumption instead of hiding it.
- Tests cover the synthetic benchmark and core scoring code.

## Limits

- The main benchmark is synthetic, not a live deployment result.
- The best result is coverage-aware: BTS plus alpha-MEU abstains often and is
  only directly useful with a fallback path.
- The live API pilot is small.
- The theory note should be read as a mechanism-design sketch, not a production
  safety proof.

## Next Step

The clean next validation is a larger live-debater run with fixed prompts,
cached transcripts, adversarial collusion prompts, and a pre-registered analysis
that reports committed accuracy, full-coverage accuracy, fallback accuracy,
cost, and latency.
