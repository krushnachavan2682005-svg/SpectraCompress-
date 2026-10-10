import sys
import platform
import json
import hashlib
import tracemalloc
import time
import numpy as np
import sklearn
import re
from split_strategy import RowSplitter
from comparison import compare_backends, format_markdown_table, DecompositionConfig, MethodType
from tests.test_day3 import make_synthetic
from eval.metrics import mean_squared_reconstruction_error
from decomposition import DecompositionTransformer

def modify_comparison():
    with open('comparison.py', 'r') as f:
        c = f.read()
    c = c.replace('"cum_var_explained_train"', '"sklearn_var_ratio"')
    if 'uncentered_energy_ratio' not in c:
        c = c.replace('"sklearn_var_ratio": cum_var_explained_train,', '"sklearn_var_ratio": cum_var_explained_train,\n                    "uncentered_energy_ratio": None,')
        c = c.replace('"sklearn_var_ratio": None,', '"sklearn_var_ratio": None,\n                    "uncentered_energy_ratio": None,')
        
        diag_old = '"sklearn_var_ratio": cum_var,'
        diag_new = '"sklearn_var_ratio": cum_var,\n                "uncentered_energy_ratio": float(np.sum(tsvd.singular_values_**2) / np.sum(X_train**2)),'
        c = c.replace(diag_old, diag_new)
        with open('comparison.py', 'w') as f:
            f.write(c)

def run():
    modify_comparison()
    
    # 1. Regenerate error analysis and experiment JSON from SAME run/data as table.
    X = make_synthetic(500, 10, [10, 8, 6, 4, 3, 2, 1, 0.5, 0.2, 0.1], seed=42)
    X += np.array([1000.0, -500.0, 200.0, 50.0, -20.0, 100.0, 0.0, -100.0, 2000.0, -500.0]) # large mean shift
    train_idx, val_idx = RowSplitter.safe_split(X, val_ratio=0.2, seed=42)
    X_train = X[train_idx]
    X_val = X[val_idx]
    
    # Run compare backends (will print table)
    from comparison import compare_backends, format_markdown_table, DecompositionConfig, MethodType
    config = DecompositionConfig(method=MethodType.sklearn_pca, seed=42)
    
    # track peak tracemalloc
    tracemalloc.start()
    results = compare_backends(X_train, X_val, [], config)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(format_markdown_table(results))
    print("\n--- JSON ---\n")
    
    # Get k=3 results from the table for JSON
    res_k3_pca = next(r for r in results if r["k"] == 3 and r["method"] == "sklearn_pca")
    
    # Compute error analysis on k=3
    config3 = DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)
    t3 = DecompositionTransformer(config3)
    t3.fit(X_train)
    X_val_hat = t3.reconstruct(X_val)
    errors = np.sum((X_val - X_val_hat)**2, axis=1)
    top_5_idx = np.argsort(errors)[-5:][::-1]
    per_feature_error = np.mean((X_val - X_val_hat)**2, axis=0)
    
    # Verify sum(per_feature_error) == val_mse at k=3
    assert np.isclose(np.sum(per_feature_error), res_k3_pca["val_mse"])
    
    # DDOF check (all backends at k=10)
    ddof_proof = {}
    for method in [MethodType.sklearn_pca, MethodType.from_scratch_covariance, MethodType.sklearn_truncated_svd]:
        try:
            t_ddof = DecompositionTransformer(DecompositionConfig(method=method, k=10, seed=42))
            t_ddof.fit(X_train)
            ddof_proof[method.value] = t_ddof.explained_variance_.tolist()
        except Exception as e:
            ddof_proof[method.value] = str(e)
            
    x_train_hash = hashlib.sha256(X_train.tobytes()).hexdigest()
    
    data = {
      "experiment": {
        "X_train_sha256": x_train_hash,
        "val_indices": val_idx.tolist(),
        "seed": 42,
        "config": {"method": "sklearn_pca", "k": 3},
        "metrics": {
            "train_mse": res_k3_pca["train_mse"],
            "val_mse": res_k3_pca["val_mse"]
        },
        "runtime_s": {
            "fit_time_median": res_k3_pca["fit_time_s"],
            "transform_time_median": res_k3_pca["transform_time_s"]
        }
      },
      "tracemalloc_peak_bytes": peak_bytes,
      "environment": {
        "python": sys.version,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cpu": platform.processor()
      },
      "error_analysis_k3": {
        "sum_per_feature_error": np.sum(per_feature_error),
        "table_val_mse": res_k3_pca["val_mse"],
        "top_5_val_rows_indices": top_5_idx.tolist(),
        "top_5_val_rows_errors": errors[top_5_idx].tolist(),
        "per_feature_error": per_feature_error.tolist()
      },
      "ddof_proof": ddof_proof
    }
    
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    run()
