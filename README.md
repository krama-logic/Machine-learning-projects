# Machine Learning Projects

A growing collection of end-to-end machine learning projects in Python,
documenting my journey through EDA, feature engineering, statistical testing,
and model building as a Data Science graduate student at Indiana University.

Each project is **self-contained** — problem statement, data cleaning, EDA,
modelling, learnings, and business recommendations — and reflects the
techniques I'm currently studying and applying to real-world datasets.

---

## Projects

### [Titanic — Survival Prediction](./Titanic)
**Binary Classification** | Random Forest | Test Accuracy: 0.80

Predict passenger survival using demographic, ticket, and family information.
Includes a deliberate experiment training **separate male/female models** to
test whether gender-conditional modelling outperforms a single unified model
(it doesn't — combined F1 dropped from 0.80 → 0.71).

**Key techniques:** Group-median imputation, feature engineering (`is_child`,
`fare_bin`, `family_size`), GridSearchCV across 4 model families, segmented
modelling experiment.

---

### [Telecom Customer Churn](./Telecom-Customer-Churn)
**Binary Classification** | Logistic Regression | F1: 0.59 | ROC-AUC: 0.84

Identify primary drivers of customer churn at a telecom company and build a
predictive engine to flag high-risk customers. Strong emphasis on translating
statistical findings into **actionable business strategy**.

**Key findings:** The "Fiber Optic Paradox" (premium users churn 2× more than
basic), the onboarding crisis (most churn in first 6 months), and four
"sustained flight risk" cohorts immune to tenure-based retention.

**Key techniques:** Hidden missing-value detection, leakage-safe pipelines,
non-linear feature engineering (`tenure_bins`), recall-prioritised metric selection.

---

### [Student Exam Performance](./Student-Exam-Performance)
**Regression + Classification + Clustering** | Multi-method analysis

A multi-lens project applying three modelling approaches to the same dataset —
each answering a different question:
- **Regression** (Ridge, R² = 0.748): *How many points* does each factor add?
- **Classification** (LogReg L1, F1 = 0.870): *Which students* are at risk?
- **Clustering** (KMeans, k=4): *What student archetypes* exist?

**Statistical validation:** ANOVA testing revealed `Sleep_Hours` has no
significant impact on exam scores (p = 0.157) — counter to popular belief.

**Key techniques:** Ordinal vs nominal encoding strategy, ANOVA hypothesis
testing, multi-task ML pipeline, clustering for proactive intervention targeting.

---

### [Irrigation Need Prediction](./Irrigation-Prediction)
**Multi-Class Classification** | Decision Tree | Macro-F1: 0.984

Predict whether agricultural fields need Low/Medium/High irrigation based on
soil, weather, crop, and farming practice data. Significant class imbalance
(High class = 3.4%) drove a **2×2 ablation study** on stratification vs class
weighting to identify which technique actually moves the needle.

**Key finding:** `class_weight='balanced'` nearly doubled minority-class
recall (0.44 → 0.85). Stratified K-Fold provided no measurable lift over
standard K-Fold — useful as a safeguard but not a silver bullet.

**Key techniques:** Multi-class evaluation, cost-asymmetric metric selection,
SHAP interpretability for class-specific drivers.

---
### [Inventory Demand Forecasting](./Inventory-Demand-Forecasting)
**Regression (Time-Series)** | LightGBM / Ridge | Val RMSLE: 0.495
Forecast weekly demand for a single SKU (Mantecadas Vainilla) from Grupo
Bimbo's 74M-row transactional dataset, reduced to 2.15M rows. Focus on
**leakage-safe feature engineering at scale** — 521k unique clients, only
7 weeks of history, and a severely right-skewed target (median 4, max 1,815).

**Key finding:** A zero-model client-mean lookup (0.512 RMSLE) captured ~80%
of extractable signal. Log-transforming 4 skewed features improved Ridge more
than switching to LightGBM — proving **preprocessing > model choice** on this
data. M2 enrichment features (client_std, client_median) ranked high in tree
importance but added zero generalization, and 81 hyperparameter configs
converged within 0.0002 RMSLE — confirming a hard **feature ceiling at ~0.495**.

**Key techniques:** Hierarchical mean encoding (client → route → depot →
channel), temporal train/val split with TimeSeriesSplit CV, log1p target
transformation, tri-point promotional encoding, price segmentation,
high-cardinality aggregation strategy.



---

## Approach

Every project follows the same structure so progression is easy to track:

1. **Problem framing** — what's being predicted and why it matters
2. **Data assessment** — dtypes, missingness, distributions
3. **Cleaning + feature engineering** — documented decisions, not just code
4. **EDA** — visual + statistical (ANOVA, correlation) where relevant
5. **Modelling** — pipelines + GridSearchCV with metric choice justified
6. **Learnings** — concise takeaways at each checkpoint
7. **Recommendations** — translating model output into actionable insights
