Titanic — Survival Prediction

Binary classification project predicting passenger survival on the Titanic
using demographic, ticket, and family information.

## Problem Statement

Build a classifier that identifies *which types of passengers were most
likely to survive* the Titanic disaster, and quantify the contribution
of each factor (gender, class, age, fare, family size).

**Target:** `Survived` (0 = No, 1 = Yes) — 38.4% base rate (mild class imbalance)

## Dataset

- **Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Rows:** 891 passengers (training set)
- **Features:** 11 (mix of numeric, categorical, and free-text)
- **Missing values:** Age (20%), Cabin (77%), Embarked (0.2%)

## Approach

| Stage | What I did |
|---|---|
| **EDA** | Survival distributions across Sex, Pclass, Age, Fare, SibSp/Parch |
| **Cleaning** | Group-median imputation for Age (by Pclass × SibSp), mode for Embarked, binary flag for Cabin presence |
| **Feature Engineering** | `is_child`, `fare_bin` (quartiles), `log_fare`, `family_size`, `family_type` (Alone/Small/Large) |
| **Modelling** | GridSearchCV across LogisticRegression, KNN, DecisionTree, RandomForest with stratified 5-fold CV |
| **Experiment** | Trained separate male/female models to test gender-conditional modelling |

## Key Findings

- **Sex is the dominant predictor** — females survived at ~74%, males at ~19%
- **Pclass shows a clean monotonic gradient** — 1st: 63%, 2nd: 47%, 3rd: 24%
- **Cabin presence (not value) predicts survival** — 67% vs 30% for those without a cabin record
- **Family size matters non-linearly** — small families (2–4) survive more than solo travellers or large families

## Results

| Model | Test Accuracy | CV F1 | CV ROC-AUC |
|---|---|---|---|
| **Random Forest (best)** | **0.80** | 0.75 | 0.86 |
| Logistic Regression | 0.78 | 0.74 | 0.86 |
| Decision Tree | 0.77 | 0.73 | 0.81 |
| KNN | 0.76 | 0.71 | 0.83 |

**Best parameters:** `RandomForestClassifier(max_depth=9, n_estimators=100)`

### Segmented Modelling Experiment
Training separate male/female models *reduced* combined F1 from 0.80 → 0.71.
The unified model captures Sex × Pclass interactions that segmented models cannot.

> **Takeaway:** more models ≠ better. A single well-tuned model with the
> right features beats naive segmentation.

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `missingno`
