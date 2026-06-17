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
