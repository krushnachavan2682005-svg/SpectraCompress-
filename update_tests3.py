with open('tests/test_day3.py', 'r') as f:
    c = f.read()

import re

# Remove __import__('error')
c = c.replace('__import__(\'error\').InputValidation', 'error.InputValidation')
if 'import error' not in c:
    c = 'import error\n' + c

# Fix test_t16
t16_old = r'def test_t16_seeds\(\):.*?assert dist < 1e-6'
t16_new = '''def test_t16_seeds():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t1 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_truncated_svd, k=2, seed=42))
    t2 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_truncated_svd, k=2, seed=42))
    t3 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_truncated_svd, k=2, seed=43))
    t1.fit(X)
    t2.fit(X)
    t3.fit(X)
    import numpy as np
    assert np.allclose(t1.components_, t2.components_)
    
    from eval.metrics import subspace_distance
    dist = subspace_distance(t1.components_, t3.components_)
    assert dist < 1e-6
    
    t4 = DecompositionTransformer(DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42))
    t5 = DecompositionTransformer(DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=43))
    t4.fit(X)
    t5.fit(X)
    assert subspace_distance(t4.components_, t5.components_) < 1e-6
    assert np.allclose(t4.components_, t5.components_)'''
c = re.sub(t16_old, t16_new, c, flags=re.DOTALL)

# Fix test_t8
t8_old = r'def test_t8_leakage_isolation\(\):.*?with pytest.raises\(AssertionError\):\s*assert_isolated\(fit_leaky, X_train, X_val\)'
t8_new = '''def test_t8_fit_leakage():
    from split_strategy import RowSplitter
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    X_train_idx, X_val_idx = RowSplitter.safe_split(X, val_ratio=0.2, seed=42)
    X_train = X[X_train_idx]
    X_val = X[X_val_idx]

    def assert_isolated(fit_fn, X_train, X_val):
        t1 = fit_fn(X_train, X_val)
        mean1 = t1.centerer.mean_.copy()
        comp1 = t1.components_.copy()
        
        t2 = fit_fn(X_train, X_val + 1e4)
        mean2 = t2.centerer.mean_.copy()
        comp2 = t2.components_.copy()
        
        import numpy as np
        assert np.array_equal(mean1, mean2)
        assert np.array_equal(comp1, comp2)
        
        from eval.metrics import mean_squared_reconstruction_error
        mse1 = mean_squared_reconstruction_error(X_train, t1.reconstruct(X_train))
        mse2 = mean_squared_reconstruction_error(X_train, t2.reconstruct(X_train))
        assert np.array_equal(mse1, mse2)

    def fit_isolated(X_t, X_v):
        t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
        t.fit(X_t)
        return t

    assert_isolated(fit_isolated, X_train, X_val)

    class LeakyTransformer:
        def fit(self, X_t, X_v):
            import numpy as np
            X_all = np.vstack([X_t, X_v])
            class MockCenterer:
                pass
            self.centerer = MockCenterer()
            self.centerer.mean_ = X_all.mean(0)
            self.components_ = np.eye(X_t.shape[1])[:2]
            return self
        def reconstruct(self, X_eval):
            return X_eval

    def fit_leaky(X_t, X_v):
        return LeakyTransformer().fit(X_t, X_v)

    import pytest
    with pytest.raises(AssertionError):
        assert_isolated(fit_leaky, X_train, X_val)

def test_t8b_transform_leakage():
    import numpy as np
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    X_val = X[:20]
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t.fit(X[20:])
    
    Z_val_1 = t.transform(X_val)
    X_val_shifted = np.concatenate([X_val]*10 + [X_val + 1000.0], axis=0)
    Z_val_2 = t.transform(X_val_shifted)[:len(X_val)]
    assert np.allclose(Z_val_1, Z_val_2)
'''
c = re.sub(t8_old, t8_new, c, flags=re.DOTALL)

# Fix test_t19
t19_old = r'def test_t19_frozen_config\(\):\s*config = DecompositionConfig\(method=MethodType\.sklearn_pca, k=2, seed=42\)\s*with pytest\.raises\(Exception\):\s*config\.k = 3'
t19_new = '''def test_t19_frozen_config():
    from dataclasses import FrozenInstanceError
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)
    import pytest
    with pytest.raises(FrozenInstanceError):
        config.k = 3'''
c = re.sub(t19_old, t19_new, c, flags=re.DOTALL)

# Make sure all parametrize have ids.
c = c.replace('@pytest.mark.parametrize("value", [None, "string", [1, 2], {"a": 1}])', '@pytest.mark.parametrize("value", [None, "string", [1, 2], {"a": 1}], ids=["none", "string", "list", "dict"])')

with open('tests/test_day3.py', 'w') as f:
    f.write(c)
