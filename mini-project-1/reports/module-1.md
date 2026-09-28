# 📊 Module 1 — Baseline Report

## ✅ Validation Metrics

| Metric | Value |
|--------|-------|
| RMSE   | 6.55 |
| MAE    | 4.29 |

## ⚙️ Setup

| | |
|---|---|
| **Data** | NYC TLC Green Taxi trip data (Parquet) — one month train / one month validation |
| **Features** | `PU_DO` (pickup–dropoff pair), `trip_distance` |
| **Target** | trip `duration` (minutes) |
| **Vectorizer** | DictVectorizer |
| **Model** | LinearRegression |

## 📝 Notes

> This is the baseline notebook (`notebooks/00-baseline.ipynb`), kept deliberately messy — the *"before"* picture. It will be replaced with a structured, tested, packaged version in the following steps of Module 1.
