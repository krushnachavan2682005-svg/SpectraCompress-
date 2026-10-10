import logging
import error
import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from sklearn.decomposition import TruncatedSVD
from feature_pipeline import Centerer
from decomposition import DecompositionConfig, MethodType, DecompositionTransformer, clip_psd_eigenvalues
from split_strategy import RowSplitter
from eval.metrics import subspace_distance, mean_squared_reconstruction_error, component_cosine_similarity
from comparison import compare_backends

def make_synthetic(n: int, d: int, singular_values: list[float], seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, d)))
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return U @ np.diag(singular_values) @ V.T

def test_t0_split():
    data = np.arange(100)
    train_val_idx, test_split_idx = RowSplitter.safe_split(data, val_ratio=0.2, seed=42)
    train_val_data = data[train_val_idx]
    
    train_idx_rel, val_idx_rel = RowSplitter.safe_split(train_val_data, val_ratio=0.2, seed=43)
    
    train_split = train_val_data[train_idx_rel]
    val_split = train_val_data[val_idx_rel]
    test_split = data[test_split_idx]
    
    assert len(set(train_split) & set(val_split)) == 0
    assert len(set(train_split) & set(test_split)) == 0
    assert len(set(val_split) & set(test_split)) == 0
    
    assert set(train_split) | set(val_split) | set(test_split) == set(data)
    
    train_val2, test_split2 = RowSplitter.safe_split(data, val_ratio=0.2, seed=42)
    assert np.array_equal(train_val_idx, train_val2)
    assert np.array_equal(test_split_idx, test_split2)

def test_t1_centerer_contract():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    X += 10.0
    centerer = Centerer()
    centerer.fit(X)
    assert np.allclose(centerer.mean_, X.mean(axis=0))
    Xc = centerer.transform(X)
    assert np.allclose(Xc.mean(axis=0), 0.0)
    assert np.allclose(centerer.inverse_transform(Xc), X)

def test_t2_3x2_example():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected_variance = 8.0
    
    for method in [MethodType.sklearn_pca, MethodType.from_scratch_covariance]:
        config = DecompositionConfig(method=method, k=1, seed=42)
        transformer = DecompositionTransformer(config)
        transformer.fit(X)
        assert np.allclose(transformer.singular_values_[0]**2 / (len(X) - 1), expected_variance)
        assert np.allclose(np.abs(transformer.components_[0]), [1/np.sqrt(2), 1/np.sqrt(2)])

def test_t3_clip_psd(caplog):
    evals = np.array([10.0, 1.0, -1e-10, 0.0])
    with caplog.at_level(logging.WARNING):
        clipped = clip_psd_eigenvalues(evals, rtol=1e-5)
    
    assert np.array_equal(clipped, [10.0, 1.0, 0.0, 0.0])
    assert "negative eigenvalues were clipped to zero" in caplog.text

def test_t4_pca_and_covariance_equivalence():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    
    t_pca = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42))
    t_pca.fit(X)
    
    t_cov = DecompositionTransformer(DecompositionConfig(method=MethodType.from_scratch_covariance, k=3, seed=42))
    t_cov.fit(X)
    
    assert np.allclose(t_pca.singular_values_, t_cov.singular_values_)
    cos_sim = component_cosine_similarity(t_pca.components_, t_cov.components_)
    assert np.allclose(np.abs(cos_sim), 1.0)
    
    assert np.allclose(t_pca.transform(X), t_cov.transform(X))
    assert np.allclose(np.abs(t_pca.components_), np.abs(t_cov.components_))

def test_from_scratch_uses_day2_eigensolver(monkeypatch):
    import decomposition
    original_top_k = decomposition.top_k_eigenpairs
    call_count = 0
    
    def mock_top_k(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_top_k(*args, **kwargs)
        
    def mock_eigh(*args, **kwargs):
        raise AssertionError("Should not use eigh")
        
    monkeypatch.setattr(decomposition, "top_k_eigenpairs", mock_top_k)
    monkeypatch.setattr(np.linalg, "eigh", mock_eigh)
    
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t_cov = DecompositionTransformer(DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42))
    t_cov.fit(X)
    
    assert call_count == 1

def test_nonconvergence_is_typed():
    import error
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t_cov = DecompositionTransformer(DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42, eig_max_iter=1))
    
    with pytest.raises(error.NonConvergenceError) as exc_info:
        t_cov.fit(X)
        
    assert isinstance(exc_info.value, error.InputValidation)

def test_t5_truncated_svd_vs_pca():
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from decomposition import DecompositionTransformer, DecompositionConfig, MethodType
    
    X = make_synthetic(100, 5, [1.0, 0.5, 0.2, 0.1, 0.05])
    X += 1000.0
    
    tsvd = TruncatedSVD(n_components=2)
    tsvd.fit(X)
    
    mean_dir = X.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    pc1 = tsvd.components_[0]
    cos_sim = np.abs(np.dot(pc1, mean_dir))
    assert cos_sim > 0.99
    
    uncentered_energy_ratio = np.sum(tsvd.singular_values_**2) / np.sum(X**2)
    
    pca = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    pca.fit(X)
    pca_ratio = np.sum(pca.explained_variance_ratio_)
    
    print('uncentered_energy_ratio:', uncentered_energy_ratio)
    print('pca_ratio:', pca_ratio)
    assert uncentered_energy_ratio > pca_ratio

def test_t6_k_equals_d():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t_pca = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=5, seed=42))
    t_pca.fit(X)
    X_hat = t_pca.reconstruct(X)
    assert np.allclose(X, X_hat)
    assert np.allclose(np.sum(t_pca.explained_variance_ratio_), 1.0)

def test_t7_k_greater_than_d():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=6, seed=42)
    with pytest.raises(Exception):
        DecompositionTransformer(config).fit(X)

def test_t7_eckart_young_baseline():
    from baseline import TopKVarianceBaseline
    
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t_pca = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t_pca.fit(X)
    pca_err = mean_squared_reconstruction_error(X, t_pca.reconstruct(X))
    
    b_state = TopKVarianceBaseline.fit_baseline(X, 2)
    X_base_hat = np.zeros_like(X)
    X_base_hat[:] = X.mean(axis=0)
    top_k = b_state.top_k_indices
    X_base_hat[:, top_k] = X[:, top_k]
    base_err = mean_squared_reconstruction_error(X, X_base_hat)
    
    assert pca_err < base_err

def assert_isolated(fit_fn, X_train, X_val):
    t1 = fit_fn(X_train, X_val)
    mean1 = t1.centerer.mean_.copy() if hasattr(t1, 'centerer') else t1.mean_
    comp1 = t1.components_.copy() if hasattr(t1, 'components_') else t1.W
    
    t2 = fit_fn(X_train, X_val + 1e4)
    mean2 = t2.centerer.mean_.copy() if hasattr(t2, 'centerer') else t2.mean_
    comp2 = t2.components_.copy() if hasattr(t2, 'components_') else t2.W
    assert np.array_equal(mean1, mean2)
    assert np.array_equal(comp1, comp2)
    
    if hasattr(t1, 'reconstruct'):
        mse1 = mean_squared_reconstruction_error(X_train, t1.reconstruct(X_train))
        mse2 = mean_squared_reconstruction_error(X_train, t2.reconstruct(X_train))
        assert np.array_equal(mse1, mse2)

    Z_val_1 = t1.transform(X_val)
    X_val_shifted = np.concatenate([X_val]*10 + [X_val + 1000.0], axis=0)
    Z_val_2 = t1.transform(X_val_shifted)[:len(X_val)]
    assert np.allclose(Z_val_1, Z_val_2)

def test_t8_fit_leakage():
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
        assert np.array_equal(mean1, mean2)
        assert np.array_equal(comp1, comp2)
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
    with pytest.raises(AssertionError):
        assert_isolated(fit_leaky, X_train, X_val)

def test_t8b_transform_leakage():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    X_val = X[:20]
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t.fit(X[20:])
    
    Z_val_1 = t.transform(X_val)
    X_val_shifted = np.concatenate([X_val]*10 + [X_val + 1000.0], axis=0)
    Z_val_2 = t.transform(X_val_shifted)[:len(X_val)]
    assert np.allclose(Z_val_1, Z_val_2)

    class LeakyTransformTransformer:
        def __init__(self):
            self.components_ = t.components_
        def transform(self, X_eval):
            # Leaky: subtracts evaluation batch mean
            Xc = X_eval - X_eval.mean(axis=0)
            return Xc @ self.components_.T

    leaky_t = LeakyTransformTransformer()
    Z_val_leaky_1 = leaky_t.transform(X_val)
    Z_val_leaky_2 = leaky_t.transform(X_val_shifted)[:len(X_val)]
    with pytest.raises(AssertionError):
        assert np.allclose(Z_val_leaky_1, Z_val_leaky_2)


@pytest.mark.parametrize("method,k,seed,err_type", [
    ("invalid_method", 2, 42, error.InputValidation),
    (MethodType.sklearn_pca, -1, 42, error.InputValidation),
    (MethodType.sklearn_pca, 0, 42, error.InputValidation),
], ids=["invalid_method", "negative_k", "zero_k"])
def test_invalid_config_inputs(method, k, seed, err_type):
    import error
    with pytest.raises(err_type):
        DecompositionConfig(method=method, k=k, seed=seed)

def test_t11_save_load(tmp_path):
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t.fit(X)
    p = tmp_path / "model.joblib"
    t.save(str(p))
    
    t2 = DecompositionTransformer.load(str(p))
    assert np.allclose(t.components_, t2.components_)
    assert np.allclose(t.singular_values_, t2.singular_values_)

@pytest.mark.parametrize("bad_val", [None, "string", [1,2], np.array([1,2])])
def test_t12_parametrized_failures(bad_val):
    import error
    if not isinstance(bad_val, int) or bad_val <= 0:
        with pytest.raises(Exception):
            DecompositionConfig(method=MethodType.sklearn_pca, k=bad_val, seed=42)
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    with pytest.raises(Exception):
        t.fit(bad_val)

def test_t12b_ddof():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    n = X.shape[0]
    for method in [MethodType.sklearn_pca, MethodType.from_scratch_covariance, MethodType.sklearn_truncated_svd]:
        t = DecompositionTransformer(DecompositionConfig(method=method, k=2, seed=42))
        t.fit(X)
        assert np.allclose(t.explained_variance_, t.singular_values_**2 / (n - 1))

def test_t13_stability():
    kappas = [1e2, 1e4, 1e6, 1e7, 1e8]
    n, d = 100, 5
    err_covs, err_svds = [], []
    for kappa in kappas:
        s = np.array([1.0, 0.5, 0.2, 0.1, 1.0/kappa])
        rng = np.random.default_rng(42)
        
        U_raw = rng.normal(size=(n, d))
        U_raw -= np.mean(U_raw, axis=0)
        U, _ = np.linalg.qr(U_raw)
        U -= np.mean(U, axis=0)
        U, _ = np.linalg.qr(U)
        
        V, _ = np.linalg.qr(rng.normal(size=(d, d)))
        X = U @ np.diag(s) @ V.T
        
        cond = np.linalg.cond(X)
        assert np.isclose(cond, kappa, rtol=1e-3)
        assert np.max(np.abs(np.mean(X, axis=0))) < 1e-12
        
        sigma_min_true = 1.0 / kappa
        C = X.T @ X / (n - 1)
        evals, _ = np.linalg.eigh(C)
        sigma_min_cov = np.sqrt(max(0, evals[0] * (n - 1)))
        
        _, S_svd, _ = np.linalg.svd(X)
        sigma_min_svd = S_svd[-1]
        
        err_cov = np.abs(sigma_min_cov - sigma_min_true) / sigma_min_true
        err_svd = np.abs(sigma_min_svd - sigma_min_true) / sigma_min_true
        
        assert err_svd <= err_cov + 1e-12
        err_covs.append(err_cov)
        err_svds.append(err_svd)
        
    assert err_covs[-1] >= 100 * err_svds[-1]

def test_metrics():
    import error
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = component_cosine_similarity(W, -W)
    assert np.allclose(sim, 1.0)
    with pytest.raises(error.ShapeMismatchError):
        component_cosine_similarity(W, np.array([[1.0]]))

def test_comparison():
    X_train = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    X_val = make_synthetic(20, 5, [10.0, 5.0, 2.0, 1.0, 0.5], seed=43)
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)
    res = compare_backends(X_train, X_val, [2], config)
    assert len(res) > 0

def test_diag_mse_vs_pca_mse():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    X += 10.0
    for k in range(1, 5):
        t_pca = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=k, seed=42))
        t_pca.fit(X)
        pca_err = mean_squared_reconstruction_error(X, t_pca.reconstruct(X))
        
        tsvd = TruncatedSVD(n_components=k, random_state=42)
        tsvd.fit(X)
        tsvd_err = mean_squared_reconstruction_error(X, tsvd.inverse_transform(tsvd.transform(X)))
        
        assert tsvd_err >= pca_err

def test_t14_single_row_transform():
    X = make_synthetic(100, 5, [10.0, 5.0, 2.0, 1.0, 0.5])
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    t.fit(X)
    Z_batch = t.transform(X)
    Z_single = np.vstack([t.transform(X[[i]]) for i in range(len(X))])
    assert np.allclose(Z_batch, Z_single)
    assert not np.allclose(Z_batch, 0)
    
def test_t15_reconstruct_error_k_lt_d():
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
    dist = subspace_distance(t1.components_, t3.components_)
    assert dist < 1e-6
    
    t4 = DecompositionTransformer(DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42))
    t5 = DecompositionTransformer(DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=43))
    t4.fit(X)
    t5.fit(X)
    assert subspace_distance(t4.components_, t5.components_) < 1e-6
    assert np.allclose(t4.components_, t5.components_)

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
    with pytest.raises(FrozenInstanceError):
        config.k = 3

def test_t20_no_gap():
    s = [10.0, 5.0, 4.95, 1.0, 0.5]
    X = make_synthetic(10000, 5, s)
    train_idx, val_idx = RowSplitter.safe_split(X, val_ratio=0.2, seed=42)
    config = DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42)
    res = compare_backends(X[train_idx], X[val_idx], [2], config)
    has_no_gap = any(r.get("status") == "NO_GAP" for r in res)
    assert has_no_gap
