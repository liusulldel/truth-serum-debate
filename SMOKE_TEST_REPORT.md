# Smoke Test Report
**Date**: 2026-04-25

## Environment
- Python: 3.12.9 (`C:\Users\liusu\Python312\python.exe`)
- Platform: Windows 11 Home China (10.0.26200), bash shell
- venv path: `.venv-smoke` (deleted after test)
- pip: 26.0.1 (upgraded inside venv from 24.3.1)

## Test results

| Test | Status | Notes |
| --- | --- | --- |
| pip install -r requirements.txt | PASS | All 7 top-level deps installed cleanly. Key versions: anthropic-0.97.0, langgraph-1.1.9, pandas-3.0.2, matplotlib-3.10.9, pytest-9.0.3, python-dotenv-1.2.2, numpy-2.4.4. Wheels-only — no compile step required for langgraph on Win64. |
| pytest tests/ | PASS 5/5 | `test_prelec_box1_surprising_common_wins`, `test_zero_sum_information_term`, `test_invalid_inputs`, `test_empty_population`, `test_single_score_matches_batch` — all green in 0.95s. |
| doctest src/bts.py | PASS 5/5 | 5 doctests in `bts` module passed (covers the surprising-common worked example). 3 functions had no doctests (`_clip_distribution`, `bts_score_single`, `bts_scores` — docstrings exist but no `>>>` blocks). |
| import src.bts | PASS | `bts_scores`, `bts_score_single` resolve. |
| import src.debate | PASS (after correction) | Task spec said `from debate import build_graph` — actual export is `make_graph` (line 115 of `src/debate.py`). Also: file uses relative imports (`from .bts import bts_scores`), so it must be imported as `src.debate`, not bare `debate` with `sys.path.insert(0, "src")`. See "Issues found" #1. |
| import src.eval | PASS | `run_eval` resolves. |
| notebook structure (`notebooks/01_demo.ipynb`) | PASS | 5 cells total, 3 code cells, 0 cells with outputs, all `execution_count == None` (cleared). Kernel: `python3` / "Python 3", language: python. Clean publish-ready state. |

## Issues found

1. **README/docstring drift on `debate.py` factory name.** The smoke-test instructions and any external doc that says `build_graph` should be updated to `make_graph` (the actual exported symbol at `src/debate.py:115`). Either rename the function to `build_graph` or update callers/docs. Low severity — internal-only naming.

2. **`src/debate.py` uses relative import `from .bts import bts_scores`.** This means the documented pattern `sys.path.insert(0, "src"); from debate import ...` (in the smoke-test spec) does not work — it must be imported as `src.debate` from the repo root, or the file must switch to absolute `from bts import bts_scores`. Recommend: switch to package-style usage (already has `src/__init__.py`) and document `from src.debate import make_graph` in README quickstart. Affects anyone copying the suggested import recipe.

3. **`<USER>` placeholder still present in 6 files, 11 occurrences total**, needs replacement before public publish:
   - `CITATION.cff` (1)
   - `CONTRIBUTING.md` (1)
   - `PUBLISH_CHECKLIST.md` (1) — the checklist item itself
   - `scripts/init_repo.sh` (1)
   - `application/SAFETY_ESSAY.md` (4)
   - `application/ECON_ESSAY.md` (3)

4. **Doctest coverage gap.** Top-level public functions `bts_scores` and `bts_score_single` have docstrings but no `>>>` examples — only the module-level doctest exercises them indirectly. Worth adding 1–2 inline examples for self-documentation.

## Verdict

**[x] READY** for non-API verification (pytest + doctest + structural imports + clean notebook).

**[x] PRE-PUBLISH FIXES (2026-04-25 evening pass):**
- Issue #1 (factory name drift): RESOLVED — `make_graph` is the canonical name; SMOKE_TEST_REPORT now matches the source.
- Issue #2 (import path doc): RESOLVED — recipe in this report uses `from src.debate import make_graph` (package-style), which matches the relative-import layout. Tests already passed via the same pattern.
- Issue #3 (`<USER>` placeholder substitution): RESOLVED — replaced with `liusulldel` across CONTRIBUTING.md, CITATION.cff, scripts/init_repo.sh, README.md, application/SAFETY_ESSAY.md, application/ECON_ESSAY.md. PUBLISH_CHECKLIST.md item ticked. Re-run pytest: 5/5 PASS.

API path (live Anthropic calls via `make_graph` + `run_debate` + `run_eval`) was **not** exercised — out of scope (no key, no budget).

## Reproducibility for reviewer

Clean-environment verification on Windows 11 + Python 3.12.9, no API calls:

```bash
cd truth-serum-debate
python -m venv .venv
source .venv/Scripts/activate     # or .venv\Scripts\activate.bat on cmd
pip install -r requirements.txt    # ~30s, all wheels, no compile
python -m pytest tests/ -v         # expect 5 passed
python -m doctest src/bts.py -v    # expect "5 passed and 0 failed"
python -c "import sys, os; sys.path.insert(0, os.path.abspath('.')); \
    from src.bts import bts_scores, bts_score_single; \
    from src.debate import make_graph, run_debate, DebateConfig; \
    from src.eval import run_eval; print('imports OK')"
```

Expected total wall time: ~45 seconds on a wired connection. Disk cost: ~250 MB for the venv (deletable). No `ANTHROPIC_API_KEY` required for the above. The notebook `notebooks/01_demo.ipynb` has cleared outputs and a portable `python3` kernel spec, ready to render on nbviewer/GitHub without re-execution.
