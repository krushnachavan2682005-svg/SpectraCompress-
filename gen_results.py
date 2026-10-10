from tests.test_day3 import make_synthetic
from comparison import compare_backends, format_markdown_table, DecompositionConfig, MethodType
from split_strategy import RowSplitter
import numpy as np
import warnings
warnings.filterwarnings('ignore')

X = make_synthetic(500, 10, [10, 8, 6, 4, 3, 2, 1, 0.5, 0.2, 0.1], seed=42)
X += np.array([1000.0, -500.0, 200.0, 50.0, -20.0, 100.0, 0.0, -100.0, 2000.0, -500.0]) # large mean shift
train_idx, val_idx = RowSplitter.safe_split(X, val_ratio=0.2, seed=42)
X_train = X[train_idx]
X_val = X[val_idx]

config = DecompositionConfig(method=MethodType.sklearn_pca)
results = compare_backends(X_train, X_val, [], config)
with open("results_out.txt", "w") as f:
    f.write('--- Comparison Table ---\n')
    f.write(format_markdown_table(results))
    f.write('\n\n--- T13 Stability Table ---\n')
    kappas = [1e2, 1e4, 1e6, 1e7, 1e8]
    n, d = 100, 5
    f.write('| kappa | err_cov | err_svd | cov >= svd | cov >= 100*svd |\n')
    f.write('|---|---|---|---|---|\n')
    for kappa in kappas:
        s = np.array([1.0, 0.5, 0.2, 0.1, 1.0/kappa])
        rng = np.random.default_rng(42)
        Z_raw = rng.normal(size=(n, d))
        Z_raw -= np.mean(Z_raw, axis=0)
        Z, _ = np.linalg.qr(Z_raw)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        X = Z @ np.diag(s) @ Q.T
        X -= np.mean(X, axis=0)
        X[0] -= np.sum(X, axis=0)
        sigma_min_true = 1.0 / kappa
        C = X.T @ X / (n - 1)
        evals, _ = np.linalg.eigh(C)
        sigma_min_cov = np.sqrt(max(0, evals[0] * (n - 1)))
        _, S_svd, _ = np.linalg.svd(X)
        sigma_min_svd = S_svd[-1]
        err_cov = np.abs(sigma_min_cov - sigma_min_true) / sigma_min_true
        err_svd = np.abs(sigma_min_svd - sigma_min_true) / sigma_min_true
        v_100 = str(err_cov >= 100*err_svd) if kappa == 1e8 else "-"
        f.write(f'| {kappa} | {err_cov} | {err_svd} | {err_cov >= err_svd} | {v_100} |\n')
