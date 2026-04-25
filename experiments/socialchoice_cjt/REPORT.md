# Social-Choice Experiment: Condorcet Jury Theorem (CJT) for LLM Debaters

**Family:** Social choice / voting / aggregation
**Mechanism chosen:** Condorcet Jury Theorem (Condorcet 1785; modern treatment Boland 1989; correlated extension Ladha 1992)

## 1. Theory

Let `N` voters cast independent binary votes, each correct with probability `p > 1/2`. Then `P_N = sum_{k=floor(N/2)+1}^{N} C(N,k) p^k (1-p)^(N-k)` is strictly increasing in `N` across odd jury sizes (Boland 1989, Thm 1) and converges to 1 as `N -> inf`. If `p < 1/2` the same monotonicity runs the other way and the majority converges to *wrong*. The load-bearing assumption is statistical *independence*. Ladha (1992, Thm 1) generalizes to exchangeable correlated voters with pairwise correlation `rho`: `Var(V_bar) = p(1-p)/N * (1 + (N-1) rho)`, which has a non-vanishing floor `p(1-p) rho`. Whenever `rho > 0`, accuracy plateaus *strictly below 1* no matter how many voters you add.

## 2. Transfer to LLM-agent debate (3 sentences)

The existing `src/debate.py` already has multiple LLM debaters answer binary questions; CJT is the minimal aggregation rule on top -- the judge counts votes -- so engineering cost is one function call. Independent CJT gives a closed-form prediction for accuracy as a function of `(p, N)`, which is exactly the calibration curve a Fellows-style experiment wants to plot. The *correlated* CJT is the more interesting transfer because LLM debaters sampled from the same base model with the same prompt almost certainly violate independence, and the Beta-Bernoulli generator in `cjt.py` lets us simulate exactly how badly `rho > 0` clips the ceiling.

## 3. Implementation

- `cjt.py` (~120 LOC): `aggregate()`, `independent_majority_correct_prob()`, `correlated_majority_correct_prob()`. numpy + stdlib only.
- `test_cjt.py`: 7 tests (5 cases, one parametrized over 3 `p` values).
- `compare_vs_bts.py`: 5-respondent toy showdown vs `src.bts`.

## 4. Test results

```
$ python -m pytest experiments/socialchoice_cjt/ -v
collected 7 items
test_cjt.py::test_aggregate_basic_majority                       PASSED
test_cjt.py::test_aggregate_tie_handling                         PASSED
test_cjt.py::test_correlation_caps_convergence                   PASSED
test_cjt.py::test_independent_cjt_monotone[0.55]                 PASSED
test_cjt.py::test_independent_cjt_monotone[0.60]                 PASSED
test_cjt.py::test_independent_cjt_monotone[0.70]                 PASSED
test_cjt.py::test_independent_cjt_reverse_for_bad_jurors         PASSED
============================== 7 passed in 0.17s ==============================
```

Diagnostic numbers:

- IID, p=0.55, N in {3,5,11,31,101,401}: P_N = 0.575, 0.593, 0.633, 0.713, 0.844, 0.978 (monotone -> 1).
- IID, p=0.40, same N: 0.352, 0.317, 0.249, 0.150, 0.039, ~0.000 (anti-CJT).
- Correlated, p=0.65, rho=0.30: N=21 -> ~0.74, N=101 -> ~0.74. Classical guarantee dies.

## 5. Compare against BTS (one toy where they disagree)

Scenario in `compare_vs_bts.py`: 5 debaters, deceptive trivia ("The capital of Pennsylvania is Philadelphia", which is false; right answer is Harrisburg). 4/5 confidently say True; all predict True is the popular answer. The lone correct debater says False but expects to be in the minority.

Output:

```
CJT majority:           decision=True  (4/5)             -> WRONG
BTS top-scorer answer:  False (BTS = +0.522 vs ~-0.13)   -> CORRECT
```

**BTS beats CJT here.** "False" is *surprisingly common* (empirical 0.20 vs predicted ~0.10) -- exactly the regime Prelec (2004) designed BTS for. CJT, content-blind and ignoring meta-prediction, cannot recover from a confidently-wrong majority. The reverse case (CJT > BTS when answers are genuinely IID and there is no shared misconception) exists structurally but isn't run here.

## 6. Honest critique -- the correlation problem

Independent-CJT is **almost certainly inapplicable to LLM debaters out of the box**. Multiple Claude / GPT instances on the same question share: identical pre-training corpus, identical RLHF policy, identical system + user prompt, often identical decoding temperature. A Beta-Bernoulli simulation with `rho = 0.3` (a plausible-to-low estimate for sibling Claude calls on a contested fact) shows accuracy plateauing around 0.74 for `p = 0.65`, vs ~1.0 in the IID case at `N = 101`. Stacking more debaters does not help; the variance floor is real. `test_correlation_caps_convergence` exercises this directly.

Implications for the Fellows pitch:

- Do not sell CJT as a free lunch. The honest framing: CJT gives a tight upper bound on what naive majority voting can buy, and the gap between that bound and observed accuracy is itself a measurement of inter-agent correlation.
- The right *use* of CJT in this codebase is as a baseline ceiling against which BTS / debate / Surprisingly Popular can be compared. If a mechanism beats correlated-CJT at fixed `(p, rho)`, that is empirical evidence it recovers information beyond raw vote counts.
- The real research question is how to *reduce* `rho` (adversarial prompting, persona injection, model-mixing) and whether the resulting independence is genuine or merely surface-level.

## 7. References (real, verified)

- Condorcet, M. de (1785). *Essai sur l'application de l'analyse a la probabilite des decisions rendues a la pluralite des voix.* Paris: Imprimerie Royale.
- Boland, P. J. (1989). "Majority Systems and the Condorcet Jury Theorem." *The Statistician* 38(3): 181-189. doi:10.2307/2348873.
- Ladha, K. K. (1992). "The Condorcet Jury Theorem, Free Speech, and Correlated Votes." *American Journal of Political Science* 36(3): 617-634. doi:10.2307/2111584.
- Prelec, D. (2004). "A Bayesian Truth Serum for Subjective Data." *Science* 306(5695): 462-466. doi:10.1126/science.1102081.
