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

## 🐳 Containerization (Module 1, Step 07)

### Image size: with vs without `.dockerignore`

| Build | Content Size |
|-------|---------------|
| Without `.dockerignore` | 252 MB |
| With `.dockerignore`    | 252 MB |

No measurable difference. The `Dockerfile` already copies only specific files
(`pyproject.toml`, `README.md`, `src/`) rather than `COPY . .`, so most of what
`.dockerignore` would normally exclude (`.venv`, `notebooks/`, `tests/`, `data/`)
was never copied in the first place. `.dockerignore` is kept anyway as a safeguard
in case the `Dockerfile` is later changed to `COPY . .`.

### Single-stage vs multi-stage build

| Build         | Content Size |
|---------------|---------------|
| Single-stage  | 258 MB |
| Multi-stage   | 252 MB |

The difference in content size is small (6 MB) for this project, because the
`Dockerfile` already copies a minimal, specific set of files in both cases. The
real benefit of the multi-stage build is **not raw size here** but **isolation**:
build-time tools and intermediate files (pip's build dependencies, wheel caches)
live only in the `builder` stage and are never present in the final `runtime`
image — only the installed package itself is copied forward via
`COPY --from=builder /install /usr/local`. This keeps the production image
cleaner and reduces what an attacker could find or exploit if the container
were compromised.

### Non-root user confirmed

The container runs as `appuser` (UID 1000), not `root`.

## 📈 MLOps Maturity Self-Assessment

**Current level: Level 2 (ML Pipeline) — partially reached**

This project has automated training (`prodml.train`), a CI-like quality gate
(pre-commit hooks + pytest with a 70% coverage gate), and basic run metadata
logged alongside the model (`models/metadata.json`: timestamp, MAE, RMSE).
These place it past Level 0 (notebooks only) and Level 1 (manual scripts, no
tracking), into the early part of Level 2.

**What's missing to fully reach Level 2, and what Module 2 will likely add:**
Real experiment tracking (e.g. MLflow) that logs and compares every training
run rather than overwriting one metadata file, and a single automated pipeline
that chains data → train → export → test → deploy, instead of these steps
being run manually in sequence as separate commands.

## ☑️ Definition of Done

1. ✅ GitHub repo public with more than one commit
2. ✅ Docker Hub token saved securely — logged in via `docker login`, image pushed successfully
3. ✅ `docker run hello-world` works; WSL2 confirmed (Docker Desktop running on WSL2 integration)
4. ⚠️ Package installs via `uv sync --extra dev` in a clean environment (project uses `uv`, not `pip install -e .`, as its package manager — same guarantee, different tool)
5. ✅ Lint passes; pre-commit hooks installed
6. ✅ Tests pass with coverage ≥ 70% (76.57%)
7. ✅ Zero `print()` statements in `src/`
8. ✅ `/health`, `/metadata`, `/predict`, `/predict/batch` all respond correctly
9. ✅ ONNX parity test passes
10. ⏳ Image pushed to Docker Hub and pullable by someone else; container confirmed to run as non-root (`appuser`) — push in progress, interrupted by network timeouts, retrying
11. ✅ `README.md` gets a stranger to a prediction in 3 commands
12. ✅ `reports/module-1.md` has: MAE, latency comparison, image-size comparison, serialization table, maturity self-assessment
13. ⏳ Pull request opened, merged; `v0.1.0` tagged — no peer reviewers available for this solo submission; PR opened and self-reviewed before merge instead of the two-peer review described in the handbook
