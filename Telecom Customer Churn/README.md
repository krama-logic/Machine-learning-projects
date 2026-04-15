# Telecom Customer Churn — Predictive Engine for Customer Retention

Identify the primary drivers of customer churn at a telecom company and
build a predictive model that flags high-risk customers **before** they leave,
enabling targeted retention campaigns.

## Problem Statement

Customer acquisition costs are 5–7× higher than retention costs. This project
builds a binary classifier that:

1. **Flags high-risk customers** before they churn (high recall priority)
2. **Quantifies the business drivers** of churn so retention teams know
   *why* a customer is at risk, not just *that* they are
3. **Translates statistical findings into actionable business strategy**
   (contract migration, onboarding overhaul, payment method nudges)

## Dataset

- **Source:** [Kaggle — Telco Customer Churn (IBM Sample)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 21 (demographics, services subscribed, account info, charges)
- **Target:** `Churn` (Yes/No) — ~26.5% churn rate (mild class imbalance)

## Approach

| Stage | What I did |
|---|---|
| **Data Cleaning** | Found `TotalCharges` had hidden missing values stored as blank strings (not NaN); these were all new customers with `tenure=0` → imputed with 0.0 to preserve data integrity |
| **EDA** | Categorical churn rates by feature, KDE distributions of numeric features by churn status, contract × payment method × internet service interactions |
| **Feature Engineering** | Engineered `tenure_bins` (New / Stabilised / Loyal / VIP) to capture the **non-linear** relationship between tenure and churn that a continuous variable would dilute |
| **Leakage Prevention** | Dropped `TotalCharges` (= MonthlyCharges × tenure → leakage) and raw `tenure` (replaced by binned version); scaling fitted only on training fold inside pipeline |
| **Modelling** | GridSearchCV across LogisticRegression, KNN, DecisionTree, RandomForest with stratified 5-fold CV |
| **Metric Choice** | Optimised for **F1 and recall** rather than accuracy — missing a churner (false negative) costs more than a false alarm |

## Key EDA Findings

### The Onboarding Crisis
The vast majority of churn occurs in the **first 6 months** of customer
tenure. After 20 months, churn drops dramatically. This motivated the
`tenure_bins` feature, which isolates the high-risk onboarding phase
(0–20 months at ~44% churn) from the loyalty phases.

### The Fiber Optic Paradox
Premium **Fiber Optic users churn at ~42%**, vs only ~19% for basic DSL users —
counterintuitive, since premium customers should be stickier. This points
to either a **service reliability problem** or a **price-to-value mismatch**.

### Contract Length Dominates
- **Month-to-month contracts:** ~43% churn
- **1-year contracts:** near-zero churn
- **2-year contracts:** near-zero churn

Zero contractual barrier to exit is the single largest predictor of churn.

### Four "Sustained Flight Risk" Cohorts
The combined heatmap analysis identified four customer segments where
churn risk does **not decay with tenure**:
1. Senior Citizens
2. Fiber Optic subscribers
3. Month-to-month contract holders
4. Electronic Check payers

## Results

**Best model:** Logistic Regression (C=100) with StandardScaler on MonthlyCharges

| Metric | Score |
|---|---|
| **CV F1** | **0.588** |
| Test F1 | 0.562 |
| ROC-AUC | ~0.84 |

Logistic Regression beat KNN, Decision Tree, and Random Forest — the
churn signal is **largely linear and additive**, with the engineered
`tenure_bins` and one-hot encoded service features providing enough
non-linearity for a regularised linear model to capture.

### Top Churn Drivers (positive coefficients)
- Fiber Optic internet
- Month-to-month contract
- Electronic check payment
- Paperless billing
- Senior citizen status

### Top Churn Anchors (negative coefficients — features that retain customers)
- 2-year contract
- 1-year contract
- Long tenure (VIP bin)
- Tech support subscription
- Online security subscription

## Business Recommendations

1. **Investigate the Fiber Optic Paradox** — strongest positive coefficient.
   Determine if this is a pricing issue or a service reliability/outage issue.
2. **Contract migration campaign** — offer high-risk month-to-month customers
   a 10% discount to lock in a 1-year commitment. Removes the zero-barrier-to-exit problem.
3. **Onboarding overhaul** — introduce a "Month 3 Check-in" or introductory rate
   that converts new customers into 1-year contracts before they hit the high-risk window.
4. **Payment method nudge** — offer auto-pay incentives to migrate Electronic
   Check users to credit card / bank transfer (consistently lower churn).
5. **Bundle Tech Support + Online Security** with high-risk plans — both are
   strong negative coefficients (anchors).
