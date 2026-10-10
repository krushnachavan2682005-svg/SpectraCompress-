import numpy as np
from comparison import compare_backends
from decomposition import DecompositionConfig, MethodType

np.random.seed(42)
X_train = np.random.randn(100, 5) * 10 + np.array([100, 20, -50, 0, 10])
X_val = np.random.randn(20, 5) * 10 + np.array([100, 20, -50, 0, 10])

config = DecompositionConfig(method=MethodType.sklearn_pca, k=1, seed=42)
res = compare_backends(X_train, X_val, [1], config)
for r in res:
    if r['method'] in ['sklearn_pca', 'diag_uncentered_tsvd']:
        print(r['method'], r['train_mse'])
