# Predicting Student Health Risk — Kaggle Playground Series S6E7

**Result:** ~0.950 balanced accuracy (public leaderboard), from a single LightGBM model with class weighting.

Hyperparameter tuning and a CatBoost ensemble were both tested and did **not** beat this baseline — and figuring out *why* turned out to be the most interesting part of the project: a diagnosed, evidenced ceiling caused by label noise near the class boundaries.

This repo walks through the EDA that found the key signal, the baseline model, the error analysis that explained the model's mistakes, and the follow-up experiments (with honest negative results) that confirmed there wasn't more headroom to find.

📓 [Full writeup notebook](notebooks/student-health-risk-analysis-writeup.ipynb) · [Competition notebook](notebooks/student-health-risk-notebook-competition.ipynb) · [Kaggle](https://www.kaggle.com/code/jamesnyoro/student-health-risk-notebook-competition)

---

## Repo structure

```
.
├── README.md
└── notebooks/
    ├── student-health-risk-analysis-writeup.ipynb    # polished walkthrough (start here)
    └── student-health-risk-notebook-competition.ipynb # working competition notebook
```

## Problem

The competition task is a 3-class classification problem — predicting a student's `health_condition` (`at-risk`, `fit`, `unhealthy`) from lifestyle and biometric features (sleep, heart rate, BMI, calorie expenditure, step count, exercise duration, water intake, diet type, stress level, physical activity level, smoking/alcohol use, gender).

The target is heavily imbalanced — `at-risk` makes up 86% of the data, with `unhealthy` (8%) and `fit` (6%) as minority classes. The competition is scored on **balanced accuracy** (mean per-class recall), so the minority classes carry real weight in the final score, and stratified cross-validation is essential to keep class ratios consistent across folds.

## Approach

1. **EDA** — checked missingness patterns (found to be effectively random, so simple imputation was safe), then looked at numeric and categorical features by class. The key signal turned out to be the interaction between `stress_level` and `physical_activity_level`.
2. **Baseline model** — a single LightGBM classifier with `class_weight="balanced"`, evaluated with 5-fold stratified CV. This produced the best score of the entire project: **CV 0.9495, public LB 0.9501**.
3. **Error analysis** — rather than stopping at the CV score, I dug into *where* the model was wrong: isolating the two largest confusion-matrix error groups and comparing their feature distributions against correctly classified rows. This surfaced a plausible label-noise explanation for the remaining errors.
4. **Follow-up experiments** (all negative results, but informative ones):
   - Removing class weighting — hurt the score, confirming it was necessary.
   - A hyperparameter sweep over `num_leaves` / `min_child_samples` — no meaningful gain over defaults.
   - A CatBoost model trained on the same folds — scored 0.9491, statistically indistinguishable from LightGBM. The two models agreed on 99.44% of predictions, and each "rescued" fewer than 6% of the other's errors — strong evidence the errors are shared, near-irreducible noise rather than a blind spot in either model.
   - A probability-averaged ensemble of both models — changed only a small fraction of predictions and did not beat the single-model baseline.

## Takeaways

- **The biggest gain came from EDA, not modeling.** Finding the `stress_level` × `physical_activity_level` interaction and confirming class weighting mattered accounted for essentially all of the achievable score.
- **A plateau isn't the same as "nothing left to try" unless you can explain *why*.** The error analysis and the LightGBM/CatBoost agreement check turned an unexplained flat score into a diagnosed, evidenced ceiling — which is what actually justifies stopping here rather than continuing to tune blindly.
- **Cross-checking two structurally different models' error overlap** is a cheap, effective way to distinguish "my model has a blind spot" from "the data has a genuine ceiling" — worth doing before assuming more tuning or a bigger model will help.

**Final submission:** the original single LightGBM baseline (CV 0.9495, LB 0.9501) — still the best result after every follow-up experiment.

## Stack

`pandas` · `numpy` · `scikit-learn` · `lightgbm` · `catboost` · `matplotlib` · `seaborn`
