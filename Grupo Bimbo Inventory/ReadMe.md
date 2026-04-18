# Grupo Bimbo — Inventory Demand Forecasting

Forecasting weekly demand for a single SKU (Mantecadas Vainilla, Producto_ID 1240) from Bimbo's 74M-row transactional dataset. The goal was not competition-grade scoring, but demonstrating disciplined time-series ML workflow — feature engineering at high cardinality, leakage-safe aggregations, and honest model comparison.

## Dataset

- **Source:** [Grupo Bimbo Inventory Demand](https://www.kaggle.com/c/grupo-bimbo-inventory-demand) (Kaggle)
- **Raw size:** ~74M rows across all products
- **Filtered:** 2.15M rows for Producto_ID 1240 across weeks 3–9
- **Features:** Depot, channel, route, client IDs + sales/return units and pesos
- **Target:** `Demanda_uni_equil` (adjusted demand — net of returns)
- **Key challenge:** 521,786 unique clients, only 7 weeks of history, severe right-skew (median 4, max 1,815)

## Results

| Model | Val RMSLE | Notes |
|-------|-----------|-------|
| Global mean baseline | 0.676 | Predicts constant; establishes floor |
| Client-mean baseline | 0.512 | Groupby lookup — no model |
| Ridge (raw features) | 0.550 | Below baseline — shape mismatch |
| Ridge (log features) | **0.496** | log1p on skewed features closed the gap |
| LightGBM (default) | **0.495** | M1 features, native categoricals |
| LightGBM (tuned) | 0.498 | Tuning confirmed feature ceiling |

All properly-configured models converged within 0.003 RMSLE, establishing a clear **feature ceiling at ~0.495**.

## Project Workflow

### 1. Data Assessment & EDA
- Renamed columns to English snake_case for readability
- Explored demand, sales, and return patterns across weeks and channels
- Identified right-skewed target → log1p transformation essential
- Derived price segments (wholesale / discount / standard / premium) — 96% of sales at ₱8.40
- Mapped client ordering frequency — only 21% order every week; 42% order ≤3 times

### 2. Feature Engineering (Leakage-Safe)
- **Hierarchical mean encoding:** client → route → depot → channel mean demand, computed strictly on training data and mapped to validation
- **Promotional features:** `is_promo`, `promo_last_week` (lead/current/lag tri-point encoding)
- **Client loyalty:** weekly order frequency per client
- **Price segmentation:** categorical bucketing based on unit cost distribution
- **M2 extensions:** `client_std`, `client_median`, `route_std` — ranked high in tree importance but did not improve generalization

### 3. Validation Strategy
- **Temporal split:** Weeks 3–7 train, weeks 8–9 validation (no random shuffle)
- **Cross-validation:** `TimeSeriesSplit(n_splits=3)` within training data
- Leakage columns (`sales_units`, `return_units`, `sales_pesos`, `return_pesos`) excluded from features

### 4. Modeling — M1 vs M2 Comparison
- **M1 (10 features):** Ridge and LightGBM with core aggregation + promo + loyalty features
- **M2 (13 features):** Added std/median features → negligible Ridge improvement (−0.001), LightGBM degraded (+0.008)
- Confirmed: additional per-client statistics describe the same distribution as `client_mean` — no independent signal

### 5. Hyperparameter Tuning
- GridSearchCV on LightGBM (27 configs × 3 folds = 81 fits)
- Best: `num_leaves=15, learning_rate=0.05, n_estimators=200`
- Top 10 configs within 0.0002 RMSLE of each other → tuning confirmed the ceiling, didn't break it

## Key Learnings

1. **Client identity dominates** — the client-mean lookup captured ~80% of extractable signal with zero modeling
2. **Preprocessing > model choice** — log-transforming 4 features improved Ridge more than switching to LightGBM
3. **High-cardinality categoricals require aggregation, not encoding** — 521k clients compressed into 3 continuous features
4. **Time-based validation is non-optional** — random KFold would leak future data into training
5. **Feature importance ≠ predictive value** — `client_std` ranked #2 in tree importance but added zero generalization
6. **Honest baselines expose model value** — without the client-mean baseline, Ridge at 0.550 would have looked reasonable
7. **Hyperparameter tuning is diagnostic** — 81 configs within 0.0002 RMSLE confirmed the ceiling, not a failure of search

## Next Steps

1. Demonstrate time-aware vs random-KFold gap empirically
2. Add lag features from prior weeks
3. Expand beyond a single product
4. Introduce external data
