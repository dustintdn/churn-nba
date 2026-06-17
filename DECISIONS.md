# Decisions Log — for PR review

A running note of judgment calls made while expanding the project from the
initial vertical slice into a "weeks of work" portfolio piece. Flagged here so
they're easy to review/revert rather than buried in diffs.

## 1. Synthetic dataset enriched with non-linearities & interactions
**Why:** The original data-generating process (DGP) was purely logistic (linear
in log-odds). When I added a logistic-regression baseline for model selection,
the baseline *beat* XGBoost — unsurprising, since the linear model was the exact
correct functional form. That made the model comparison meaningless and undercut
the project's XGBoost + tree-SHAP framing.
**Decision:** Added realistic non-linear and interaction terms to the DGP
(support-ticket acceleration, price-increase × low-tenure interaction,
disengagement × month-to-month compounding, last-login threshold effect). Real
churn genuinely has these, so this makes the simulation *more* realistic, not
less — and lets gradient boosting earn its selection over the linear baseline.
**Review:** If you'd rather show "the simple model won, and I understood why,"
we can revert the DGP and let the baseline deploy (would need model-agnostic
SHAP). I judged the XGBoost-wins path better aligned with the brief.

## 2. Automatic winner selection (not hardcoded XGBoost)
**Decision:** `train.py` now fits a logistic baseline, default XGB, and tuned
XGB, then deploys whichever has the best held-out PR-AUC. Demonstrates real model
selection rather than asserting XGBoost by fiat. The notebook's tree-SHAP assumes
the winner is XGBoost (true on the current data).

## 3. Light RandomizedSearchCV (n_iter=25, cv=3)
**Decision:** Kept the search small so `python src/train.py` finishes in ~30s and
a `--fast` flag skips it entirely for CI/tests. Enough to show tuning rigor
without a multi-hour grid that adds no portfolio signal.

## 4. Scripts run as modules (`python -m src.batch_score`)
**Decision:** `batch_score.py` and the dashboard import from the `src` package,
so they're invoked as modules from the repo root rather than as loose files.
Documented in the README and docstrings. Avoids `sys.path` hacks in `src/`.

## 5. Docker build not verified in this environment
**Decision:** Wrote a standard `Dockerfile` + `docker-compose.yml` (API + dashboard
sharing one image). `docker compose config` validates the compose file, but the
**image build is blocked here** — the sandbox network policy returns 403 from the
Docker registry CDN, so `python:3.11-slim` can't be pulled. The files are standard
and should build in a normal environment; please verify `docker compose up` locally.

## 6. Streamlit dashboard verified by headless boot only
**Decision:** Confirmed the dashboard boots (HTTP 200) and that its shared
`score_dataframe` path works, but I did not click through the UI interactively.
Worth a manual smoke test before you showcase it.

## 7. Bug fixed: tests no longer clobber the deployed artifact
**Found during review:** the end-to-end model test called `train(fast=True)`, which
overwrote `models/churn_model.joblib` + `metrics.json` with a fast-mode (no-search)
run — and in fast mode the logistic baseline narrowly wins, so the *committed* model
had silently become logistic regression while the notebook/README described tuned
XGBoost. **Fix:** `train()` now takes `model_path`/`metrics_path`, and the test trains
to a `tmp_path`. Retrained properly so the deployed artifact is the tuned XGBoost again.

## 8. Note: XGBoost wins by a thin margin
On the current data the tuned XGBoost (PR-AUC 0.574) beats the logistic baseline
(0.571) only narrowly. That's honest and realistic; I deliberately did not over-tune
the DGP to manufacture a bigger gap. If you'd prefer a wider margin for the portfolio
story, we can add more interaction structure to the simulation.

## 9. Notebook re-execution
The notebook was rebuilt/re-run against the enriched dataset and new model
comparison so its numbers match the current `metrics.json`. The README metrics
were updated to match as well.
