import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from error import InputValidation, ShapeMismatchError, NotFittedError, MemoryGuardError
from feature_pipeline import (
    sparse_column_mean_var,
    sparse_center,
    to_dense_centered,
    Centerer,
)


# =====================================================================
# 1. Tests for sparse_column_mean_var
# =====================================================================

def test_sparse_column_mean_var_tiny_hand_checkable_csr():
    # Matrix shape (3, 2):
    # [ [0, 4],
    #   [2, 0],
    #   [0, 2] ]
    # Column 0: values [0, 2, 0] -> mean = 2/3, var (ddof=0) = ((0-2/3)^2 + (2-2/3)^2 + (0-2/3)^2)/3 = (4/9 + 16/9 + 4/9)/3 = 24/27 = 8/9
    # Column 1: values [4, 0, 2] -> mean = 6/3 = 2, var (ddof=0) = ((4-2)^2 + (0-2)^2 + (2-2)^2)/3 = (4 + 4 + 0)/3 = 8/3
    dense = np.array([[0.0, 4.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    csr = sp.csr_matrix(dense)
    csr_copy = csr.copy()

    mean, var = sparse_column_mean_var(csr, ddof=0)

    assert mean.shape == (2,)
    assert var.shape == (2,)
    assert mean.dtype == np.float64
    assert var.dtype == np.float64

    expected_mean = np.array([2.0 / 3.0, 2.0])
    expected_var = np.array([8.0 / 9.0, 8.0 / 3.0])

    assert np.allclose(mean, expected_mean)
    assert np.allclose(var, expected_var)

    # Verify input unchanged
    assert (csr != csr_copy).nnz == 0
    assert np.array_equal(csr.indices, csr_copy.indices)
    assert np.array_equal(csr.indptr, csr_copy.indptr)


def test_sparse_column_mean_var_csc_equivalence():
    dense = np.array([[0.0, 4.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    csc = sp.csc_matrix(dense)
    csc_copy = csc.copy()

    mean, var = sparse_column_mean_var(csc, ddof=0)

    expected_mean = np.array([2.0 / 3.0, 2.0])
    expected_var = np.array([8.0 / 9.0, 8.0 / 3.0])

    assert np.allclose(mean, expected_mean)
    assert np.allclose(var, expected_var)

    # Verify input unchanged
    assert (csc != csc_copy).nnz == 0


def test_sparse_column_mean_var_implicit_zeros_and_constant_column():
    # Matrix shape (5, 4):
    # Col 0: mostly implicit zeros with a single non-zero
    # Col 1: completely implicit zeros (all zeros)
    # Col 2: non-zero constant column (value 7.0 everywhere)
    # Col 3: arbitrary non-zeros and implicit zeros
    dense = np.zeros((5, 4), dtype=np.float64)
    dense[0, 0] = 10.0
    dense[:, 2] = 7.0
    dense[1, 3] = -2.0
    dense[3, 3] = 4.0

    for fmt in [sp.csr_matrix, sp.csc_matrix]:
        sparse_mat = fmt(dense)
        mean, var = sparse_column_mean_var(sparse_mat, ddof=0)
        expected_mean = np.mean(dense, axis=0)
        expected_var = np.var(dense, axis=0)

        assert np.allclose(mean, expected_mean)
        assert np.allclose(var, expected_var)

        # Constant column variance must be exactly 0 (or within machine epsilon)
        assert np.isclose(var[1], 0.0)
        assert np.isclose(var[2], 0.0)


def test_sparse_column_mean_var_validation():
    # Dense array rejected
    with pytest.raises(InputValidation) as exc1:
        sparse_column_mean_var(np.zeros((5, 2)))
    assert "Expected scipy.sparse matrix" in exc1.value.message

    # NaN in data rejected
    dense = np.array([[1.0, 0.0], [0.0, np.nan]])
    with pytest.raises(InputValidation) as exc2:
        sparse_column_mean_var(sp.csr_matrix(dense))
    assert "NaN or infinite" in exc2.value.message

    # Inf in data rejected
    dense_inf = np.array([[1.0, 0.0], [0.0, np.inf]])
    with pytest.raises(InputValidation) as exc3:
        sparse_column_mean_var(sp.csr_matrix(dense_inf))
    assert "NaN or infinite" in exc3.value.message

    # Invalid ddof
    csr = sp.csr_matrix(np.ones((5, 2)))
    with pytest.raises(InputValidation) as exc4:
        sparse_column_mean_var(csr, ddof=-1)
    assert "ddof must be a non-negative integer" in exc4.value.message

    with pytest.raises(InputValidation) as exc5:
        sparse_column_mean_var(csr, ddof="0")
    assert "ddof must be a non-negative integer" in exc5.value.message

    with pytest.raises(InputValidation) as exc6:
        sparse_column_mean_var(csr, ddof=5)
    assert "strictly greater than ddof" in exc6.value.message


# =====================================================================
# 2. Tests for sparse_center
# =====================================================================

@pytest.fixture
def sample_sparse_data():
    rng = np.random.default_rng(123)
    dense = rng.normal(size=(20, 6)).astype(np.float64)
    dense[dense < 0.2] = 0.0  # Sparsity
    dense[:, 3] = 0.0         # Zero column
    mean = np.mean(dense, axis=0)
    return dense, mean


def test_sparse_center_matvec_rmatvec_matmat_csr(sample_sparse_data):
    dense, mean = sample_sparse_data
    dense_centered = dense - mean
    csr = sp.csr_matrix(dense)
    csr_copy = csr.copy()

    op = sparse_center(csr, mean)

    assert isinstance(op, LinearOperator)
    assert op.shape == (20, 6)
    assert op.dtype == np.float64

    rng = np.random.default_rng(42)
    v = rng.normal(size=6)
    u = rng.normal(size=20)
    V = rng.normal(size=(6, 4))

    # matvec
    res_matvec = op.matvec(v)
    assert res_matvec.shape == (20,)
    assert np.allclose(res_matvec, dense_centered @ v)

    # rmatvec
    res_rmatvec = op.rmatvec(u)
    assert res_rmatvec.shape == (6,)
    assert np.allclose(res_rmatvec, dense_centered.T @ u)

    # matmat
    res_matmat = op.matmat(V)
    assert res_matmat.shape == (20, 4)
    assert np.allclose(res_matmat, dense_centered @ V)

    # Ensure input is unchanged
    assert (csr != csr_copy).nnz == 0


def test_sparse_center_csc_equivalence(sample_sparse_data):
    dense, mean = sample_sparse_data
    dense_centered = dense - mean
    csc = sp.csc_matrix(dense)

    op = sparse_center(csc, mean)

    rng = np.random.default_rng(42)
    v = rng.normal(size=6)
    u = rng.normal(size=20)
    V = rng.normal(size=(6, 4))

    assert np.allclose(op.matvec(v), dense_centered @ v)
    assert np.allclose(op.rmatvec(u), dense_centered.T @ u)
    assert np.allclose(op.matmat(V), dense_centered @ V)


def test_sparse_center_dimension_and_input_validation(sample_sparse_data):
    dense, mean = sample_sparse_data
    csr = sp.csr_matrix(dense)

    # Mean size mismatch
    with pytest.raises(ShapeMismatchError) as exc1:
        sparse_center(csr, mean[:3])
    assert "mean dimension" in exc1.value.message

    # Non-sparse input
    with pytest.raises(InputValidation) as exc2:
        sparse_center(dense, mean)
    assert "Expected scipy.sparse" in exc2.value.message

    op = sparse_center(csr, mean)

    # Invalid vector shape for matvec
    with pytest.raises((ShapeMismatchError, ValueError)):
        op.matvec(np.zeros(10))

    # Invalid vector shape for rmatvec
    with pytest.raises((ShapeMismatchError, ValueError)):
        op.rmatvec(np.zeros(10))

    # Invalid matrix shape for matmat
    with pytest.raises((ShapeMismatchError, ValueError)):
        op.matmat(np.zeros((10, 2)))



def test_sparse_center_does_not_call_toarray_or_todense(sample_sparse_data, monkeypatch):
    dense, mean = sample_sparse_data
    csr = sp.csr_matrix(dense)

    def forbidden_call(*args, **kwargs):
        raise AssertionError("Production sparse-centering code must NEVER call toarray/todense!")

    # Monkeypatch toarray and todense on the csr instance
    monkeypatch.setattr(csr, "toarray", forbidden_call)
    monkeypatch.setattr(csr, "todense", forbidden_call)

    op = sparse_center(csr, mean)

    v = np.ones(6)
    u = np.ones(20)
    V = np.ones((6, 2))

    # None of these operations should invoke toarray/todense
    op.matvec(v)
    op.rmatvec(u)
    op.matmat(V)


# =====================================================================
# 3. Tests for to_dense_centered
# =====================================================================

def test_to_dense_centered_within_budget(sample_sparse_data):
    dense, mean = sample_sparse_data
    csr = sp.csr_matrix(dense)
    n, d = dense.shape
    required_bytes = n * d * 8  # 20 * 6 * 8 = 960 bytes

    # Budget strictly greater
    out = to_dense_centered(csr, mean, max_bytes=2000)
    assert isinstance(out, np.ndarray)
    assert out.shape == (n, d)
    assert np.allclose(out, dense - mean)

    # Exact boundary budget
    out_boundary = to_dense_centered(csr, mean, max_bytes=required_bytes)
    assert np.allclose(out_boundary, dense - mean)


def test_to_dense_centered_exceeds_budget_raises_memory_guard(sample_sparse_data, monkeypatch):
    dense, mean = sample_sparse_data
    csr = sp.csr_matrix(dense)
    n, d = dense.shape
    required_bytes = n * d * 8  # 960 bytes

    def forbidden_call(*args, **kwargs):
        raise AssertionError("Dense matrix must NOT be materialized when memory limit is exceeded!")

    monkeypatch.setattr(csr, "toarray", forbidden_call)
    monkeypatch.setattr(csr, "todense", forbidden_call)

    # 1 byte less than required
    with pytest.raises(MemoryGuardError) as exc_info:
        to_dense_centered(csr, mean, max_bytes=required_bytes - 1)
    assert "exceeds maximum allowed budget" in str(exc_info.value)
    assert isinstance(exc_info.value, ValueError)


def test_to_dense_centered_validation(sample_sparse_data):
    dense, mean = sample_sparse_data
    csr = sp.csr_matrix(dense)

    # Negative max_bytes
    with pytest.raises(InputValidation) as exc1:
        to_dense_centered(csr, mean, max_bytes=-1)
    assert "max_bytes must be a non-negative integer" in exc1.value.message

    # Non-integer max_bytes
    with pytest.raises(InputValidation) as exc2:
        to_dense_centered(csr, mean, max_bytes="1000")
    assert "max_bytes must be a non-negative integer" in exc2.value.message

    with pytest.raises(InputValidation) as exc3:
        to_dense_centered(csr, mean, max_bytes=True)
    assert "max_bytes must be a non-negative integer" in exc3.value.message

    # Mismatched mean
    with pytest.raises(ShapeMismatchError) as exc4:
        to_dense_centered(csr, mean[:2], max_bytes=10000)
    assert "mean dimension" in exc4.value.message


# =====================================================================
# 4. Tests for Centerer (Dense Regression & Sparse Extensions)
# =====================================================================

def test_centerer_dense_regression_unchanged():
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    centerer = Centerer()
    centerer.fit(dense)

    assert np.allclose(centerer.mean_, [3.0, 4.0])
    assert np.allclose(centerer.feature_mean_, [3.0, 4.0])
    assert centerer.n_features_in_ == 2
    assert centerer.n_samples_seen_ == 3

    Xc = centerer.transform(dense)
    assert isinstance(Xc, np.ndarray)
    assert np.allclose(Xc.mean(axis=0), 0.0)
    assert np.allclose(centerer.inverse_transform(Xc), dense)


def test_centerer_sparse_fit_and_transform_no_densification(sample_sparse_data, monkeypatch):
    dense, mean = sample_sparse_data
    csr = sp.csr_matrix(dense)

    def forbidden_call(*args, **kwargs):
        raise AssertionError("Centerer.fit and transform must NEVER call toarray/todense!")

    monkeypatch.setattr(csr, "toarray", forbidden_call)
    monkeypatch.setattr(csr, "todense", forbidden_call)

    centerer = Centerer()
    centerer.fit(csr)

    assert np.allclose(centerer.mean_, mean)
    assert np.allclose(centerer.feature_mean_, mean)
    assert centerer.n_features_in_ == 6
    assert centerer.n_samples_seen_ == 20

    # transform must return LinearOperator without densifying
    op = centerer.transform(csr)
    assert isinstance(op, LinearOperator)
    assert op.shape == (20, 6)

    # Verify mathematical correctness
    v = np.ones(6)
    assert np.allclose(op.matvec(v), (dense - mean) @ v)


def test_centerer_unfitted_and_dimension_mismatch():
    centerer = Centerer()
    with pytest.raises(NotFittedError) as exc1:
        centerer.transform(np.zeros((5, 3)))
    assert "is not fitted" in exc1.value.message

    centerer.fit(np.zeros((5, 3)))
    with pytest.raises(ShapeMismatchError) as exc2:
        centerer.transform(np.zeros((5, 4)))
    assert "expected d=3" in exc2.value.message


def test_centerer_scaling_opt_in_with_constant_features():
    # Dense with a constant feature (column 1 has zero variance)
    dense = np.array([
        [1.0, 5.0, 10.0],
        [3.0, 5.0, 20.0],
        [5.0, 5.0, 30.0],
    ], dtype=np.float64)

    # Default is centering only
    c_default = Centerer()
    c_default.fit(dense)
    assert c_default.with_std is False
    assert c_default.scale_ is None

    # Opt-in scaling
    c_scaled = Centerer(with_std=True)
    c_scaled.fit(dense)

    assert c_scaled.with_std is True
    assert c_scaled.constant_features_ is not None
    assert np.array_equal(c_scaled.constant_features_, [False, True, False])

    # Scale for constant feature must be set to 1.0 (no division by zero)
    assert c_scaled.scale_[1] == 1.0
    assert np.isclose(c_scaled.scale_[0], 2.0)
    assert np.isclose(c_scaled.scale_[2], 10.0)

    # Transform
    scaled_Xc = c_scaled.transform(dense)
    assert np.all(np.isfinite(scaled_Xc))
    assert np.allclose(scaled_Xc[:, 1], 0.0)  # Constant feature centered at 0

    # Inverse transform
    recovered = c_scaled.inverse_transform(scaled_Xc)
    assert np.allclose(recovered, dense)


def test_centerer_sparse_scaling_opt_in():
    dense = np.array([
        [1.0, 5.0, 10.0],
        [3.0, 5.0, 20.0],
        [5.0, 5.0, 30.0],
    ], dtype=np.float64)
    csr = sp.csr_matrix(dense)

    c_sparse = Centerer(with_std=True)
    c_sparse.fit(csr)

    assert c_sparse.scale_[1] == 1.0
    op = c_sparse.transform(csr)
    assert isinstance(op, LinearOperator)

    v = np.array([1.0, 2.0, 3.0])
    dense_scaled = (dense - np.mean(dense, axis=0)) / c_sparse.scale_
    assert np.allclose(op.matvec(v), dense_scaled @ v)


# =====================================================================
# 5. Missing Audit Tests (Scaled rmatvec, matmat, Adjoint, and Edge Cases)
# =====================================================================

@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_sparse_center_scaled_rmatvec_against_dense_reference(fmt):
    """Verify scaled sparse rmatvec against a dense reference using non-uniform feature scales.

    Mathematical definition:
        A = (X - 1 @ mean.reshape(1, -1)) / scale
        A.T @ u = (X.T @ u - mean * sum(u)) / scale
    """
    dense = np.array([
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0, 0.0],
        [0.0, 7.0, 0.0, 8.0],
        [2.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 4.0, 5.0],
    ], dtype=np.float64)
    sparse_mat = fmt(dense)
    sparse_copy = sparse_mat.copy()

    # Non-uniform means and non-uniform positive scales
    mean = np.array([1.2, 3.5, 0.8, 2.4], dtype=np.float64)
    scale = np.array([0.5, 2.0, 1.5, 4.0], dtype=np.float64)

    # Dense reference
    dense_centered_scaled = (dense - mean) / scale

    op = sparse_center(sparse_mat, mean=mean, scale=scale)

    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.normal(size=6)
        expected_rmatvec = dense_centered_scaled.T @ u
        actual_rmatvec = op.rmatvec(u)

        assert actual_rmatvec.shape == (4,)
        assert np.allclose(actual_rmatvec, expected_rmatvec, rtol=1e-12, atol=1e-12)

    # Input matrix unchanged
    assert (sparse_mat != sparse_copy).nnz == 0


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_sparse_center_scaled_matmat_against_dense_reference(fmt):
    """Verify scaled sparse matmat against a dense reference using non-uniform feature scales.

    Mathematical definition:
        A = (X - 1 @ mean.reshape(1, -1)) / scale
        A @ V = (X @ (V / scale) - 1 @ (mean @ (V / scale)))
    """
    dense = np.array([
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0, 0.0],
        [0.0, 7.0, 0.0, 8.0],
        [2.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 4.0, 5.0],
    ], dtype=np.float64)
    sparse_mat = fmt(dense)
    sparse_copy = sparse_mat.copy()

    # Non-uniform means and non-uniform positive scales
    mean = np.array([1.2, 3.5, 0.8, 2.4], dtype=np.float64)
    scale = np.array([0.5, 2.0, 1.5, 4.0], dtype=np.float64)

    # Dense reference
    dense_centered_scaled = (dense - mean) / scale

    op = sparse_center(sparse_mat, mean=mean, scale=scale)

    rng = np.random.default_rng(42)
    for k in [1, 2, 4]:
        V = rng.normal(size=(4, k))
        expected_matmat = dense_centered_scaled @ V
        actual_matmat = op.matmat(V)

        assert actual_matmat.shape == (6, k)
        assert np.allclose(actual_matmat, expected_matmat, rtol=1e-12, atol=1e-12)

    # Input matrix unchanged
    assert (sparse_mat != sparse_copy).nnz == 0


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
@pytest.mark.parametrize("use_scale", [False, True])
def test_sparse_center_adjoint_consistency(fmt, use_scale):
    """Verify adjoint consistency:
    np.allclose(np.vdot(A @ v, u), np.vdot(v, A.rmatvec(u)), rtol=1e-12, atol=1e-12)
    """
    dense = np.array([
        [2.0, 0.0, 1.5, -1.0],
        [0.0, 4.2, 0.0, 3.1],
        [-1.0, 0.0, 2.8, 0.0],
        [0.5, -2.0, 0.0, 1.0],
        [3.0, 1.0, -0.5, 0.0],
    ], dtype=np.float64)
    sparse_mat = fmt(dense)
    n, d = dense.shape

    mean = np.array([0.9, 1.1, -0.4, 2.5], dtype=np.float64)
    scale = np.array([1.5, 0.8, 3.2, 0.5], dtype=np.float64) if use_scale else None

    op = sparse_center(sparse_mat, mean=mean, scale=scale)

    rng = np.random.default_rng(777)
    for _ in range(5):
        v = rng.normal(size=d)
        u = rng.normal(size=n)

        # Adjoint identity: <A @ v, u> == <v, A.rmatvec(u)>
        left_inner = np.vdot(op @ v, u)
        right_inner = np.vdot(v, op.rmatvec(u))

        assert np.allclose(left_inner, right_inner, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_sparse_column_mean_var_single_row_ddof0(fmt):
    """sparse_column_mean_var with one row (n=1) and ddof=0 returns exact values and zero variance."""
    dense = np.array([[3.5, 0.0, -2.1, 7.0]], dtype=np.float64)
    sparse_mat = fmt(dense)

    mean, var = sparse_column_mean_var(sparse_mat, ddof=0)

    assert mean.shape == (4,)
    assert var.shape == (4,)
    assert mean.dtype == np.float64
    assert var.dtype == np.float64
    assert np.allclose(mean, [3.5, 0.0, -2.1, 7.0])
    assert np.allclose(var, [0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix])
def test_sparse_column_mean_var_single_row_ddof1(fmt):
    """sparse_column_mean_var with one row (n=1) and ddof=1 verifies documented validation behaviour."""
    dense = np.array([[3.5, 0.0, -2.1, 7.0]], dtype=np.float64)
    sparse_mat = fmt(dense)

    with pytest.raises(InputValidation) as exc_info:
        sparse_column_mean_var(sparse_mat, ddof=1)

    assert "n_samples (1) must be strictly greater than ddof (1)" in exc_info.value.message


def test_centerer_constant_var_tol_configuration():
    """Verify Centerer constant_var_tol handling and validation."""
    # Default is 1e-12
    c_def = Centerer()
    assert c_def.constant_var_tol == 1e-12

    # None defaults to 1e-12
    c_none = Centerer(constant_var_tol=None)
    assert c_none.constant_var_tol == 1e-12

    # Explicit custom tolerance
    c_custom = Centerer(constant_var_tol=1e-8)
    assert c_custom.constant_var_tol == 1e-8

    # Negative tolerance rejected
    with pytest.raises(InputValidation) as exc1:
        Centerer(constant_var_tol=-1e-5)
    assert "constant_var_tol must be non-negative" in exc1.value.message

    # NaN / Inf tolerance rejected
    with pytest.raises(InputValidation) as exc2:
        Centerer(constant_var_tol=float("nan"))
    assert "constant_var_tol must be non-negative" in exc2.value.message

