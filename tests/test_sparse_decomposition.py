import pytest
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD

from error import (
    InputValidation,
    ShapeMismatchError,
    NotFittedError,
    MemoryGuardError,
)
from stability_diagnostics import (
    StabilityConfig,
    SparseNotSupportedError,
    NonFiniteInputError,
    ConditionReport,
    InputDiagnostics,
)
from decomposition import (
    DecompositionConfig,
    MethodType,
    DecompositionTransformer,
)
from eval.metrics import (
    subspace_distance,
    component_cosine_similarity,
    mean_squared_reconstruction_error,
    sparse_reconstruction_mse,
)


# =====================================================================
# 1. Dense Regression: Existing Methods Retain Previous Behaviour
# =====================================================================

def test_dense_regression_all_methods():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5))

    for method in [MethodType.from_scratch_covariance, MethodType.sklearn_pca, MethodType.sklearn_truncated_svd]:
        config = DecompositionConfig(method=method, k=2, seed=42)
        transformer = DecompositionTransformer(config)
        transformer.fit(X)

        assert transformer.n_features_in_ == 5
        assert transformer.requested_rank_ == 2
        assert transformer.effective_rank_ == 2
        assert transformer.n_components == 2
        assert transformer.centered_ is True
        assert transformer.components_.shape == (2, 5)
        assert transformer.singular_values_.shape == (2,)
        assert transformer.explained_variance_.shape == (2,)
        assert transformer.explained_variance_ratio_.shape == (2,)
        assert np.allclose(transformer.mean_, np.mean(X, axis=0))
        assert np.allclose(transformer.feature_mean_, transformer.mean_)
        assert isinstance(transformer.condition_report_, ConditionReport)
        assert isinstance(transformer.input_diagnostics_, InputDiagnostics)

        # Transform and inverse transform
        Z = transformer.transform(X)
        assert Z.shape == (50, 2)
        X_rec = transformer.inverse_transform(Z)
        assert X_rec.shape == (50, 5)
        assert np.allclose(transformer.reconstruct(X), X_rec)


# =====================================================================
# 2. Sparse Covariance Path vs. Dense Reference
# =====================================================================

@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_sparse_covariance_vs_dense_reference(fmt):
    rng = np.random.default_rng(123)
    dense = rng.normal(size=(40, 6))
    dense[dense < 0.3] = 0.0  # Introduce sparsity
    sparse_mat = fmt(dense)

    config_cov = DecompositionConfig(method=MethodType.from_scratch_covariance, k=3, seed=42)

    # Fit on dense reference
    t_dense = DecompositionTransformer(config_cov)
    t_dense.fit(dense)

    # Fit on sparse matrix
    t_sparse = DecompositionTransformer(config_cov)
    t_sparse.fit(sparse_mat)

    assert np.allclose(t_sparse.mean_, t_dense.mean_)
    assert np.allclose(t_sparse.total_variance_, t_dense.total_variance_)
    assert np.allclose(t_sparse.singular_values_, t_dense.singular_values_, rtol=1e-8, atol=1e-8)
    assert np.allclose(t_sparse.explained_variance_, t_dense.explained_variance_, rtol=1e-8, atol=1e-8)

    # Subspace distance must be negligible
    dist = subspace_distance(t_sparse.components_, t_dense.components_)
    assert dist < 1e-8

    # Transformation must yield numerically identical embeddings
    Z_sparse = t_sparse.transform(sparse_mat)
    Z_dense = t_dense.transform(dense)
    assert np.allclose(Z_sparse, Z_dense, rtol=1e-8, atol=1e-8)


# =====================================================================
# 3. Sparse Implicit Centered Method vs. Dense Reference
# =====================================================================

@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_scipy_implicit_centered_vs_dense_reference(fmt):
    rng = np.random.default_rng(456)
    dense = rng.normal(size=(60, 8))
    dense[dense < 0.2] = 0.0
    sparse_mat = fmt(dense)

    # Reference using dense sklearn PCA
    pca_cfg = DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)
    t_pca = DecompositionTransformer(pca_cfg)
    t_pca.fit(dense)

    # Implicit centered on sparse matrix
    scipy_cfg = DecompositionConfig(method=MethodType.scipy_implicit_centered, k=3, seed=42)
    t_implicit = DecompositionTransformer(scipy_cfg)
    t_implicit.fit(sparse_mat)

    assert np.allclose(t_implicit.mean_, np.mean(dense, axis=0))
    assert np.allclose(t_implicit.singular_values_, t_pca.singular_values_, rtol=1e-6, atol=1e-6)

    # Subspace alignment
    dist = subspace_distance(t_implicit.components_, t_pca.components_)
    assert dist < 1e-6

    # Reconstruction error using sparse_reconstruction_mse vs dense mean_squared_reconstruction_error
    dense_mse = mean_squared_reconstruction_error(dense, t_pca.reconstruct(dense))
    sparse_mse = sparse_reconstruction_mse(sparse_mat, t_implicit)
    assert np.isclose(dense_mse, sparse_mse, rtol=1e-5)


# =====================================================================
# 4. sklearn_pca with Sparse Input Raises SparseNotSupportedError
# =====================================================================

def test_sklearn_pca_sparse_raises_sparse_not_supported():
    csr = sp.csr_matrix(np.ones((20, 5)))
    cfg = DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)
    t = DecompositionTransformer(cfg)

    with pytest.raises(SparseNotSupportedError) as exc_info:
        t.fit(csr)

    assert "requires dense input" in exc_info.value.message


# =====================================================================
# 5. sklearn_truncated_svd with Sparse Input Sets centered_ = False & Warns
# =====================================================================

def test_sklearn_truncated_svd_sparse_uncentered_warning():
    rng = np.random.default_rng(789)
    dense = rng.normal(loc=5.0, size=(30, 4))
    dense[dense < 4.0] = 0.0
    csr = sp.csr_matrix(dense)

    cfg = DecompositionConfig(method=MethodType.sklearn_truncated_svd, k=2, seed=42)
    t = DecompositionTransformer(cfg)

    with pytest.warns(UserWarning, match="uncentered TruncatedSVD"):
        t.fit(csr)

    assert t.centered_ is False
    assert t.mean_ is None
    assert any("uncentered TruncatedSVD" in w for w in t.fit_warnings_)

    # Transform must NOT subtract mean
    Z = t.transform(csr)
    assert np.allclose(Z, (csr @ t.components_.T).toarray() if hasattr(csr @ t.components_.T, "toarray") else csr @ t.components_.T)


# =====================================================================
# 6. Sparse Transform Without Densifying Input
# =====================================================================

def test_sparse_transform_does_not_densify_input(monkeypatch):
    rng = np.random.default_rng(42)
    dense = rng.normal(size=(25, 6))
    dense[dense < 0.0] = 0.0
    csr = sp.csr_matrix(dense)

    cfg = DecompositionConfig(method=MethodType.scipy_implicit_centered, k=3, seed=42)
    t = DecompositionTransformer(cfg)
    t.fit(csr)

    # Monkeypatch toarray and todense on a new test matrix to guarantee no densification
    test_csr = sp.csr_matrix(dense)

    def forbidden_call(*args, **kwargs):
        raise AssertionError("Transform must NOT call toarray or todense on sparse inputs!")

    monkeypatch.setattr(test_csr, "toarray", forbidden_call)
    monkeypatch.setattr(test_csr, "todense", forbidden_call)

    # Transform without densifying
    Z = t.transform(test_csr)
    assert isinstance(Z, np.ndarray)
    assert Z.shape == (25, 3)

    # Check numerical accuracy against dense reference
    expected_Z = (dense - t.mean_) @ t.components_.T
    assert np.allclose(Z, expected_Z)


# =====================================================================
# 7. Rank Fallback on Rank-Deficient Data (No Zero-Padding)
# =====================================================================

def test_rank_fallback_no_zero_padding():
    # Construct rank-2 matrix of shape (30, 5): features 2, 3, 4 are duplicates of 0 and 1
    rng = np.random.default_rng(42)
    base = rng.normal(size=(30, 2))
    dense = np.column_stack([base[:, 0], base[:, 1], base[:, 0], base[:, 1], base[:, 0] + base[:, 1]])
    csr = sp.csr_matrix(dense)

    # Request k=4 components on a rank-2 matrix
    cfg = DecompositionConfig(method=MethodType.from_scratch_covariance, k=4, seed=42)
    t = DecompositionTransformer(cfg)

    with pytest.warns(UserWarning, match="Numerical rank .* is below requested rank"):
        t.fit(csr)

    assert t.requested_rank_ == 4
    assert t.effective_rank_ == 2
    assert t.components_.shape == (2, 5)
    assert t.singular_values_.shape == (2,)
    assert t.explained_variance_.shape == (2,)

    # Verify orthonormality of retained components
    gram_comp = t.components_ @ t.components_.T
    assert np.allclose(gram_comp, np.eye(2), atol=1e-10)

    # Missing components are NOT zero-padded
    assert not np.any(np.all(t.components_ == 0, axis=1))


# =====================================================================
# 8. Constant-Only and Zero-Rank Cases
# =====================================================================

def test_constant_only_and_zero_rank_data():
    # Shape (20, 4) with zero variance (all 7.0)
    dense = np.full((20, 4), 7.0, dtype=np.float64)
    csr = sp.csr_matrix(dense)

    cfg = DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42)
    t = DecompositionTransformer(cfg)

    with pytest.warns(UserWarning, match="Numerical rank"):
        t.fit(csr)

    assert t.effective_rank_ == 0
    assert t.components_.shape == (0, 4)
    assert t.singular_values_.shape == (0,)
    assert t.condition_report_.numerical_rank == 0
    assert t.condition_report_.kappa == float("inf")

    # Transform returns shape (20, 0)
    Z = t.transform(csr)
    assert Z.shape == (20, 0)

    # Inverse transform returns constant mean
    X_rec = t.inverse_transform(Z)
    assert X_rec.shape == (20, 4)
    assert np.allclose(X_rec, 7.0)


# =====================================================================
# 9. Cancellation Warning and Path Selection
# =====================================================================

def test_cancellation_detection_and_path_selection():
    # Synthetic dataset with large mean and small variance causing catastrophic subtraction cancellation
    rng = np.random.default_rng(42)
    dense = rng.normal(loc=1e6, scale=1.0, size=(50, 4))
    csr = sp.csr_matrix(dense)

    cfg = DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42)
    t = DecompositionTransformer(cfg)

    with pytest.warns(UserWarning, match="Catastrophic cancellation detected"):
        t.fit(csr)

    assert any("Catastrophic cancellation detected" in w for w in t.fit_warnings_)
    assert t.effective_rank_ == 2
    assert t.components_.shape == (2, 4)


# =====================================================================
# 10. Memory Guard on Oversized Inverse Reconstruction
# =====================================================================

def test_memory_guard_oversized_reconstruction():
    rng = np.random.default_rng(42)
    dense = rng.normal(size=(20, 5))

    stab_cfg = StabilityConfig(max_dense_bytes=500)  # Low budget: 500 bytes
    t = DecompositionTransformer(
        DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42),
        stability_config=stab_cfg
    )
    t.fit(dense)

    Z = t.transform(dense)
    # Required bytes: 20 * 5 * 8 = 800 bytes > 500 bytes budget
    with pytest.raises(MemoryGuardError) as exc_info:
        t.inverse_transform(Z)

    assert "exceeds maximum allowed budget" in exc_info.value.message


# =====================================================================
# 11. Consistent NaN / Inf Validation for Dense and Sparse Inputs
# =====================================================================

def test_nan_inf_validation_dense_and_sparse():
    cfg = DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42)

    # Dense NaN
    dense_nan = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    t1 = DecompositionTransformer(cfg)
    with pytest.raises(NonFiniteInputError):
        t1.fit(dense_nan)

    # Dense Inf
    dense_inf = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
    t2 = DecompositionTransformer(cfg)
    with pytest.raises(NonFiniteInputError):
        t2.fit(dense_inf)

    # Sparse NaN
    sparse_nan = sp.csr_matrix(dense_nan)
    t3 = DecompositionTransformer(cfg)
    with pytest.raises(NonFiniteInputError):
        t3.fit(sparse_nan)

    # Sparse Inf
    sparse_inf = sp.csr_matrix(dense_inf)
    t4 = DecompositionTransformer(cfg)
    with pytest.raises(NonFiniteInputError):
        t4.fit(sparse_inf)


# =====================================================================
# 12. sparse_reconstruction_mse Metric Functionality
# =====================================================================

def test_sparse_reconstruction_mse_metric():
    rng = np.random.default_rng(42)
    orig = rng.normal(size=(40, 6))
    recon = orig + 0.05 * rng.normal(size=(40, 6))

    dense_mse = mean_squared_reconstruction_error(orig, recon)
    sparse_mse = sparse_reconstruction_mse(sp.csr_matrix(orig), recon)

    assert np.isclose(dense_mse, sparse_mse)

    # Test passing fitted transformer
    cfg = DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)
    t = DecompositionTransformer(cfg).fit(orig)
    mse_direct = sparse_reconstruction_mse(sp.csr_matrix(orig), t)
    mse_manual = mean_squared_reconstruction_error(orig, t.reconstruct(orig))
    assert np.isclose(mse_direct, mse_manual)

    # Shape mismatch raises ShapeMismatchError
    with pytest.raises(ShapeMismatchError):
        sparse_reconstruction_mse(orig, recon[:, :3])
