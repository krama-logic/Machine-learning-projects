# Student Exam Performance — Predictive Analytics for Early Intervention

A multi-method machine learning project combining **regression**,
**classification**, and **clustering** to shift educational interventions
from *reactive grading* to *proactive risk identification*.

## Problem Statement

Historically, educational interventions happen *after* a student fails an exam.
This project builds a predictive engine that identifies "At-Risk" students
**weeks before** their final exams, allowing school counsellors to allocate
limited resources (free tutoring, study workshops) where they are
mathematically proven to have the highest impact.

Three modelling lenses are applied to the same dataset:

| Lens | Question Answered |
|---|---|
| **Regression** | *How many points* does each factor add to or subtract from the exam score? |
| **Classification** | *Which students* are at risk of falling into the lowest grade band? |
| **Clustering** | *What hidden student archetypes* exist, independent of exam outcomes? |

## Dataset

- **Source:** [Kaggle — Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **Rows:** ~6,600 students
- **Features:** 19 (academic, behavioural, socioeconomic)
- **Target:** `Exam_Score` (0–100, mean ≈ 67.23, std ≈ 3.9 — narrow spread)

## Approach

| Stage | What I did |
|---|---|
| **EDA** | KDE plots by grade band, correlation heatmap, **ANOVA significance testing** to validate which features genuinely differ across performance tiers |
| **Cleaning** | Mode imputation for <5% missing categoricals, removal of impossible scores (>100) |
| **Encoding** | Ordinal encoding for ranked categoricals (Low/Medium/High), one-hot for nominal |
| **Feature Engineering** | Created `grades` band (At Risk / Average / Good / Excellent) for stratified analysis |
| **Modelling** | GridSearchCV across multiple model families for each ML task |

## Results

### 1. Regression — Score Elasticity

**Best model:** Ridge (α=10) | **Test R² = 0.748** | **Train R² = 0.730**

- Linear models (Ridge, Lasso) outperformed tree-based models (RF, GBM, DT) —
  relationships are largely **linear and additive** with no strong
  interactions for trees to exploit.
- R² ≈ 0.75 means ~25% of variance is irreducible noise (exam-day stress,
  question variance, motivation on the day).
- **Top boosters:** Attendance, Hours_Studied, Previous_Scores, Access_to_Resources
- **Top detractors:** Learning_Disabilities, Distance_from_Home (Far), low Parental_Education

### 2. Classification — At-Risk Identification

| Model | F1 | Accuracy | Recall | Precision |
|---|---|---|---|---|
| **Logistic Regression (L1, C=10)** | **0.870** | **0.875** | 0.875 | 0.884 |
| Random Forest | 0.699 | 0.72 | 0.72 | 0.72 |
| Decision Tree | 0.675 | 0.67 | 0.67 | 0.67 |
| KNN | 0.673 | 0.68 | 0.68 | 0.70 |

L1-regularised LogReg won decisively, performing implicit feature selection
and confirming that the predictive signal is **linear and sparse**.

### 3. Clustering — Student Archetypes

**KMeans with k=4** revealed four distinct student profiles based on
behavioural and socioeconomic features (excluding `Exam_Score` to avoid leakage):

| Cluster | Size | Profile |
|---|---|---|
| 0 | 2,287 | Mainstream cohort |
| 1 | 3,175 | Largest group — average behaviour |
| 2 | 498 | Small specialised segment |
| 3 | 646 | Distinct minority cluster |

Clustering enables **intervention targeting before any exam results exist** —
a counsellor can act on cluster membership during the term, not after grades.

## Statistical Validation

ANOVA tests confirmed that most numeric features differ significantly
across grade bands (p < 0.05), with one notable exception:

> **Sleep_Hours fails the significance test (p = 0.157)** —
> we fail to reject the null hypothesis that sleep is equal across grade bands.
> Counter to popular belief, sleep hours do not predict exam outcomes
> in this dataset.

## Key Findings

1. **Attendance and study hours** are the two most controllable predictors
   — both behavioural, both directly actionable.
2. **Sleep hours show no statistically significant impact** (p = 0.157).
3. **Socioeconomic factors** (Family_Income, Access_to_Resources) measurably
   influence outcomes — equity interventions are statistically justified.
4. **Four distinct student archetypes** exist — each requiring a different
   institutional response.
5. The combined regression + classification + clustering pipeline enables
   **proactive intervention before exam results exist**.

## Recommendations

- **Early warning system** — flag students below 75% attendance for immediate pastoral intervention.
- **Study skills workshops** for first-year students.
- **Tutoring prioritisation** — each additional tutoring session correlates with score improvement; route resources to students flagged by the classifier.

