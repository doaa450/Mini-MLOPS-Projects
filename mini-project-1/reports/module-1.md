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


## 🔄 Serialization format comparison

| Format   | Human-readable | Cross-language | Schema-enforced | Safe from untrusted source |
|----------|-----------------|-----------------|-------------------|-------------------------------|
| JSON     | Yes             | Yes             | No                | Yes                           |
| Protobuf | No              | Yes             | Yes               | Yes                           |
| Pickle   | No              | No              | No                | **No**                        |
| ONNX     | No              | Yes             | Yes               | Yes                           |

**Decision:** This service serves predictions with **ONNX**, because it is safe to load
from an untrusted source (unlike pickle), cross-language, and — as the benchmark below
shows — meaningfully faster at inference time.

> ⚠️ **Pickle executes arbitrary code on load.** Never load a `.pkl` file you did not
> produce yourself. The pickle model in this project is used only for training and local
> comparison, never served directly to end users.

## ✅ ONNX vs Pickle parity

`tests/test_parity.py` runs 500 validation rows through both the pickle model and the
ONNX model and asserts `np.allclose(pred_pkl, pred_onnx, atol=1e-4)`. **Result: PASSED.**

## ⏱️ Latency benchmark (500 validation rows, single-row inference)

| Model  | Mean latency | p95 latency |
|--------|---------------|---------------|
| Pickle | 0.203 ms      | 0.228 ms      |
| ONNX   | 0.031 ms      | 0.035 ms      |

ONNX is roughly 6.5x faster than pickle on both mean and p95 latency.
