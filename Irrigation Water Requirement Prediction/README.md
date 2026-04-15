# Irrigation Need Prediction — Multi-Class Classification

Predict whether an agricultural field's irrigation need is **Low**, **Medium**,
or **High** based on soil conditions, weather, crop characteristics, and
farming practices.

## Problem Statement

Predicting irrigation requirements is critical for optimizing water usage
and preventing crop loss:
- **Under-irrigation** → crop stress and yield loss
- **Over-irrigation** → wasted water and soil degradation

This project builds a 3-class classifier and explores how class imbalance
should be handled when the *cost of errors is asymmetric* — missing a
"High" need (false negative) is far more damaging than a false alarm.

## Dataset

- **Rows:** 10,000 fields
- **Features:** 19 (11 numeric, 8 categorical)
- **Target distribution:** Low (58.6%), Medium (38.0%), **High (3.4%)** — significant class imbalance
- **Missing values:** None

**Feature groups:**
- *Soil:* pH, moisture, organic carbon, electrical conductivity
- *Weather:* temperature, humidity, rainfall, sunlight, wind speed
- *Crop & practice:* crop type, growth stage, season, irrigation type, water source, mulching, region

## Approach

| Stage | What I did |
|---|---|
| **EDA** | KDE + boxplots by class; identified Soil_Moisture, Temperature, Wind_Speed as strongest numeric separators; Crop_Growth_Stage and Mulching as strongest categorical signals |
| **Imbalance Handling** | 2×2 ablation study on LogisticRegression: `{simple, stratified} × {no-weight, class_weight='balanced'}` to isolate the impact of each technique |
| **Modelling** | GridSearchCV across LogisticRegression, KNN, DecisionTree, RandomForest with stratified 5-fold CV, scoring on macro-F1 |
| **Interpretability** | Feature importance + SHAP beeswarm to validate the model is learning physically coherent relationships |

## Key Experiment — Why class_weight matters more than stratification

| Method | High-class Recall | High-class Precision | Accuracy |
|---|---|---|---|
| Simple K-Fold | 0.44 | 0.64 | 0.82 |
| Stratified K-Fold | 0.44 | 0.64 | 0.82 |
| **Class-weighted** | **0.85** | 0.32 | 0.78 |

**Findings:**
- **Stratification provided no measurable lift** — the minority class (3%)
  was numerous enough in absolute terms (~50 per fold) to be reliably
  represented under random splitting. It remains best practice as a safeguard.
- **`class_weight='balanced'` nearly doubled High-class recall** (0.44 → 0.85)
  at the cost of precision. For irrigation, where missing a "High" field
  means crop death, **recall on the minority class is the right objective**,
  even though macro-F1 favoured the unweighted model.

> **Takeaway:** metric choice must reflect real-world cost asymmetry,
> not just statistical balance.

## Results

| Model | Test Macro-F1 |
|---|---|
| **Decision Tree (max_depth=9, balanced)** | **0.984** |
| Logistic Regression (balanced) | 0.676 |

The tree-based model massively outperformed LogReg, confirming the EDA
hypothesis that categorical features (which showed only marginal *univariate*
signal) carry substantial predictive value through **interactions** that
linear models cannot capture.

### Confusion Matrix (row-normalized)
- **Low & Medium:** 99% correctly classified
- **High:** 92.86% correct; the remaining 7.14% misclassified as Medium —
  the model **degrades gracefully**, never flipping a "High" need to "Low"
  (no catastrophic misclassifications).

### Feature Importance & SHAP

Top drivers of "High" irrigation need (validated via SHAP beeswarm):
1. **Soil_Moisture** (low)
2. **Temperature** (high)
3. **Wind_Speed** (high)
4. **Mulching_Used** (absent)
5. **Rainfall** (low)

This decision structure is **physically coherent** — it matches agronomic
intuition, increasing confidence the model has learned genuine relationships
rather than spurious correlations. Features that looked uninformative in
EDA (Region, Irrigation_Type, Water_Source) had negligible SHAP contributions
even after accounting for interactions, closing the hypothesis loop.
