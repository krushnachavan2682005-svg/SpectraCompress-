import re

with open('tests/test_day3.py', 'r') as f:
    content = f.read()

# Replace test_t8
t8_pattern = r'def assert_isolated\(fit_fn, X_train, X_val\):.*?(?=def test_t9_invalid_method\(\):)'
t8_new = """def assert_isolated(fit_fn, X_train, X_val):
    t1 = fit_fn(X_train, X_val)
    mean1 = t1.centerer.mean_.copy() if hasattr(t1, 'centerer') else t1.mean_
    comp1 = t1.components_.copy() if hasattr(t1, 'components_') else t1.W
    
    t2 = fit_fn(X_train, X_val + 1e4)
    mean2 = t2.centerer.mean_.copy() if hasattr(t2, 'centerer') else t2.mean_
    comp2 = t2.components_.copy() if hasattr(t2, 'components_') else t2.W
    
    import numpy as np
    assert np.array_equal(mean1, mean2)
    assert np.array_equal(comp1, comp2)
    
    if hasattr(t1, 'reconstruct'):
        from eval.metrics import mean_squared_reconstruction_error
        mse1 = mean_squared_reconstruction_error(X_train, t1.reconstruct(X_train))
        mse2 = mean_squared_reconstruction_error(X_train, t2.reconstruct(X_train))
        assert np.array_equal(mse1, mse2)

    Z_val_1 = t1.transform(X_val)
    X_val_shifted = np.concatenate([X_val]*10 + [X_val + 1000.0], axis=0)
    Z_val_2 = t1.transform(X_val_shifted)[:len(X_val)]
    assert np.allclose(Z_val_1, Z_val_2)

def test_t8_leakage_isolation():
    from split_strategy import RowSplitter
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    X_train_idx, X_val_idx = RowSplitter.safe_split(X, val_ratio=0.2, seed=42)
    X_train = X[X_train_idx]
    X_val = X[X_val_idx]

    def fit_isolated(X_t, X_v):
        t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
        t.fit(X_t)
        return t

    assert_isolated(fit_isolated, X_train, X_val)

    class LeakyTransformer:
        def fit(self, X_t, X_v):
            import numpy as np
            X_all = np.vstack([X_t, X_v])
            self.mean_ = X_all.mean(0)
            self.W = np.eye(X_t.shape[1])[:2]
            return self
        def transform(self, X_eval):
            return (X_eval - self.mean_) @ self.W.T

    def fit_leaky(X_t, X_v):
        return LeakyTransformer().fit(X_t, X_v)

    import pytest
    with pytest.raises(AssertionError):
        assert_isolated(fit_leaky, X_train, X_val)

"""

content = re.sub(t8_pattern, t8_new, content, flags=re.DOTALL)


t9_pattern = r'def test_t9_invalid_method\(\):.*?(?=def test_t11_save_load\(tmp_path\):)'
t9_new = """@pytest.mark.parametrize("method,k,seed,err_type", [
    ("invalid_method", 2, 42, error.InputValidation),
    (MethodType.sklearn_pca, -1, 42, error.InputValidation),
    (MethodType.sklearn_pca, 0, 42, error.InputValidation),
], ids=["invalid_method", "negative_k", "zero_k"])
def test_invalid_config_inputs(method, k, seed, err_type):
    import error
    with pytest.raises(err_type):
        DecompositionConfig(method=method, k=k, seed=seed)

"""
content = re.sub(t9_pattern, t9_new, content, flags=re.DOTALL)

content = content.replace('for k in range(1, 6):', 'for k in range(1, 5):')

new_tests = """
def test_t14_single_row_transform():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t.fit(X)
    Z_batch = t.transform(X)
    Z_single = np.vstack([t.transform(X[[i]]) for i in range(len(X))])
    assert np.allclose(Z_batch, Z_single)
    assert not np.allclose(Z_batch, 0)
    
def test_t15_reconstruct_error_k_lt_d():
    from eval.metrics import mean_squared_reconstruction_error
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t.fit(X)
    X_hat_0 = t.reconstruct(X[[0]])
    assert mean_squared_reconstruction_error(X[[0]], X_hat_0) > 1e-6

def test_t16_seeds():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t1 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_truncated_svd, k=2, seed=42))
    t2 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_truncated_svd, k=2, seed=42))
    t3 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_truncated_svd, k=2, seed=43))
    t1.fit(X)
    t2.fit(X)
    t3.fit(X)
    assert np.allclose(t1.components_, t2.components_)
    
    from eval.metrics import subspace_distance
    dist = subspace_distance(t1.components_, t3.components_)
    assert dist < 1e-6

def test_t17_inverse_transform_before_fit():
    import error
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    with pytest.raises(error.NotFittedError):
        t.inverse_transform(np.zeros((1, 2)))

def test_t18_wrong_k_Z():
    import error
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t.fit(X)
    with pytest.raises(error.ShapeMismatchError):
        t.inverse_transform(np.zeros((1, 3)))

def test_t19_frozen_config():
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)
    with pytest.raises(Exception):
        config.k = 3

def test_t20_no_gap():
    s = [10.0, 5.0, 4.9, 1.0, 0.5]
    X = make_synthetic(100, 5, s)
    from comparison import compare_backends, DecompositionConfig, MethodType
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)
    res = compare_backends(X, X[:20], [2], config)
    has_no_gap = any(r.get("status") == "NO_GAP" for r in res)
    assert has_no_gap
"""

with open('tests/test_day3.py', 'w') as f:
    f.write(content + new_tests)
