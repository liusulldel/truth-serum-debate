# Publish Checklist

Run through this list before pushing to GitHub. Every box must be ticked.

- [ ] SCRUB_PLAN: all items marked 🔴 have been resolved
- [ ] `pytest` passes locally on Python 3.11 (and 3.12 if available)
- [ ] `nbstripout notebooks/*.ipynb` has been run (no cell outputs committed)
- [x] README badges: `<USER>` placeholder replaced with `liusulldel`
- [x] `LICENSE`: year (2026) and holder ("Liu Su (liusully@gmail.com)") set
- [ ] `.env` is **not** in `git ls-files` output (only `.env.example` should be tracked)
- [ ] `git status` is clean — no untracked or modified files left over
- [ ] `bash scripts/init_repo.sh` has been executed (creates the initial commit)
- [ ] `gh repo create truth-serum-debate --public --source=. --remote=origin` succeeded
- [ ] `git push -u origin main` succeeded
- [ ] CI on GitHub Actions passes green on the first push
- [ ] "Cite this repository" button appears on the GitHub repo page (CITATION.cff valid)
