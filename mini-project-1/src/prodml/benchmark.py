import pickle
import time

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import to_dicts


def measure_latency(predict_fn, X: np.ndarray, n_runs: int = 500) -> dict[str, float]:
    """Time predict_fn on each row of X separately, return mean and p95 in ms."""
    times_ms = []
    for i in range(n_runs):
        row = X[i : i + 1]
        start = time.perf_counter()
        predict_fn(row)
        times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def run_benchmark() -> dict[str, dict[str, float]]:
    _, val_df = split_data(load_data())
    sample = val_df.sample(n=500, random_state=settings.random_state)

    with open(settings.model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)

    X = dv.transform(to_dicts(sample)).toarray().astype(np.float32)

    pkl_stats = measure_latency(model.predict, X)

    session = ort.InferenceSession(settings.onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_stats = measure_latency(lambda row: session.run(None, {input_name: row}), X)

    return {"pickle": pkl_stats, "onnx": onnx_stats}


if __name__ == "__main__":
    results = run_benchmark()
    for name, stats in results.items():
        print(f"{name}: mean={stats['mean_ms']:.3f}ms  p95={stats['p95_ms']:.3f}ms")
