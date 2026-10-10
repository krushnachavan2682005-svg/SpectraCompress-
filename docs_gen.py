import json
import hashlib
import sys
import tracemalloc
import time
import numpy as np

from tests.test_day3 import make_synthetic
from eval.metrics import mean_squared_reconstruction_error
from decomposition import DecompositionTransformer, DecompositionConfig, MethodType

X_train = make_synthetic(100, 10, [10, 8, 6, 4, 3, 2, 1, 0.5, 0.2, 0.1])
x_hash = hashlib.sha256(X_train.tobytes()).hexdigest()

config = DecompositionConfig(method=MethodType.sklearn_pca, k=10, seed=42)
t = DecompositionTransformer(config)

tracemalloc.start()
start = time.perf_counter()
t.fit(X_train)
fit_time = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

X_hat = t.reconstruct(X_train)
train_mse = mean_squared_reconstruction_error(X_train, X_hat)

# Top 5 val rows error analysis (at k=3)
config3 = DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)
t3 = DecompositionTransformer(config3)
t3.fit(X_train)
X_val = make_synthetic(20, 10, [10, 8, 6, 4, 3, 2, 1, 0.5, 0.2, 0.1], seed=43)
X_val_hat = t3.reconstruct(X_val)
errors = np.sum((X_val - X_val_hat)**2, axis=1)
top_5_idx = np.argsort(errors)[-5:][::-1]
per_feature_error = np.mean((X_val - X_val_hat)**2, axis=0)

ddof_result = np.allclose(t.explained_variance_, t.singular_values_**2 / (len(X_train) - 1))

data = {
    "experiment": {
        "X_train_sha256": x_hash,
        "seed": 42,
        "config": {"method": "sklearn_pca", "k": 10},
        "metrics": {"train_mse": train_mse},
        "runtime_s": fit_time
    },
    "tracemalloc_peak_bytes": peak,
    "environment": {
        "python": sys.version,
        "numpy": np.__version__
    },
    "error_analysis_k3": {
        "top_5_val_rows_indices": top_5_idx.tolist(),
        "top_5_val_rows_errors": errors[top_5_idx].tolist(),
        "per_feature_error": per_feature_error.tolist()
    },
    "ddof_result": ddof_result
}

with open("docs/day3_report.md", "w") as f:
    f.write("# Day 3 Report\n\n")
    f.write("```json\n")
    json.dump(data, f, indent=2)
    f.write("\n```\n")
