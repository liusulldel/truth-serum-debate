# Contributing to truth-serum-debate

Thanks for your interest. This is a research codebase; contributions that
improve clarity, reproducibility, or empirical rigor are most welcome.

## Dev setup

```bash
git clone https://github.com/liusulldel/truth-serum-debate.git
cd truth-serum-debate

# Create + activate a virtualenv (Python >= 3.11 required)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Editable install + dev tools
pip install -e .
pip install pytest ruff pre-commit nbstripout

# Install git hooks (auto-runs ruff + nbstripout on every commit)
pre-commit install
```

Copy `.env.example` to `.env` and fill in your API keys before running any
script that hits a live model. Tests do not require real keys.

## Running tests

```bash
pytest -v                          # full suite
pytest tests/test_bts.py -v        # one file
pytest -k "scoring" -v             # by keyword
```

CI runs the same suite on Python 3.11 and 3.12. Tests must pass on both.

## Style

- `ruff check .` and `ruff format --check .` must both pass — the pre-commit
  hook fixes most issues automatically.
- Line length 100. Follow existing module conventions for naming.
- Notebooks: outputs are stripped on commit by `nbstripout`. Do not commit
  cell outputs by hand.

## Pull requests

1. Branch from `main` with a descriptive name (`fix/bts-divergence`,
   `feat/judge-ensemble`).
2. Keep diffs focused — one logical change per PR.
3. Add or update tests for new behavior.
4. If the change affects results, update the relevant section of `README.md`
   or `THEORY.md` in the same PR.
5. Run `pytest` and `ruff check .` locally before opening the PR.

## Reporting issues

Include: Python version, OS, the exact command run, full traceback, and
(if relevant) which model / API endpoint was being called. Mock-based repros
are strongly preferred over ones that require live API access.
