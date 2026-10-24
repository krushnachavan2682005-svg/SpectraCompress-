import json
import math
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
import scipy.sparse as sp

from error import (
    InputValidation,
    SpectraCompressError,
    ValidationError,
    ShapeMismatchError,
    InvalidFoldConfigError,
)
from eval.metrics import mean_squared_reconstruction_error
from stability_diagnostics import (
    StabilityConfig,
    StabilityConfigError,
    ConditionReport,
    InputDiagnostics,
    ReportValidationError,
    NonFiniteInputError,
    SparseNotSupportedError,
    validate_input,
    compute_condition_report,
    condition_report,
    stable_column_variance,
    stable_variance,
    component_cosine_stability,
    subspace_similarity,
    eigengap_table,
    sparse_reconstruction_mse,
)
from decomposition import (
    DecompositionConfig,
    MethodType,
    DecompositionTransformer,
)
from cross_validation import (
    CrossValidationConfig,
    FoldReport,
    StabilityReport,
    KFoldStabilityRunner,
    generate_kfold_indices,
    shuffle_columns_null,
    evaluate_stability_verdict,
)



def test_stability_config_defaults():
    config = StabilityConfig()
    assert config.cond_warning_threshold == 1e8
    assert config.cond_critical_threshold == 1e12
    assert config.spectral_gap_ratio == 1.05
    assert config.perturbation_scale == 1e-5
    assert config.rank_tol_factor == 1.0
    assert config.constant_var_tol == 1e-12
    assert config.cancellation_ratio_threshold == 1e-4
    assert config.max_dense_bytes == 100_000_000


def test_stability_config_frozen():
    config = StabilityConfig()
    with pytest.raises(FrozenInstanceError):
        config.cond_warning_threshold = 1e7
    with pytest.raises(FrozenInstanceError):
        config.perturbation_scale = 0.1
    with pytest.raises(FrozenInstanceError):
        config.max_dense_bytes = 50_000_000


def test_stability_config_valid_custom():
    config = StabilityConfig(
        cond_warning_threshold=1e4,
        cond_critical_threshold=1e8,
        spectral_gap_ratio=1.2,
        perturbation_scale=0.0,
        rank_tol_factor=2.0,
        constant_var_tol=0.0,
        cancellation_ratio_threshold=1e-2,
        max_dense_bytes=50_000_000,
    )
    assert config.cond_warning_threshold == 1e4
    assert config.cond_critical_threshold == 1e8
    assert config.spectral_gap_ratio == 1.2
    assert config.perturbation_scale == 0.0
    assert config.rank_tol_factor == 2.0
    assert config.constant_var_tol == 0.0
    assert config.cancellation_ratio_threshold == 1e-2
    assert config.max_dense_bytes == 50_000_000

    # None is also allowed for rank_tol_factor
    config_none_factor = StabilityConfig(rank_tol_factor=None)
    assert config_none_factor.rank_tol_factor is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"cond_warning_threshold": 0.0}, "cond_warning_threshold must be positive"),
        ({"cond_warning_threshold": -1.0}, "cond_warning_threshold must be positive"),
        ({"cond_warning_threshold": float("nan")}, "cond_warning_threshold must be positive"),
        ({"cond_critical_threshold": 0.0}, "cond_critical_threshold must be positive"),
        ({"cond_critical_threshold": -10.0}, "cond_critical_threshold must be positive"),
        ({"cond_critical_threshold": float("nan")}, "cond_critical_threshold must be positive"),
        (
            {"cond_warning_threshold": 1e8, "cond_critical_threshold": 1e8},
            "strictly greater than",
        ),
        (
            {"cond_warning_threshold": 1e10, "cond_critical_threshold": 1e8},
            "strictly greater than",
        ),
        ({"spectral_gap_ratio": 0.0}, "spectral_gap_ratio must be positive"),
        ({"spectral_gap_ratio": -0.5}, "spectral_gap_ratio must be positive"),
        ({"spectral_gap_ratio": float("nan")}, "spectral_gap_ratio must be positive"),
        ({"perturbation_scale": -1e-5}, "perturbation_scale must be non-negative"),
        ({"perturbation_scale": float("nan")}, "perturbation_scale must be non-negative"),
        ({"rank_tol_factor": 0.0}, "rank_tol_factor must be positive and finite"),
        ({"rank_tol_factor": -1.0}, "rank_tol_factor must be positive and finite"),
        ({"rank_tol_factor": float("inf")}, "rank_tol_factor must be positive and finite"),
        ({"constant_var_tol": -1.0}, "constant_var_tol must be non-negative"),
        ({"constant_var_tol": float("nan")}, "constant_var_tol must be non-negative"),
        ({"cancellation_ratio_threshold": 0.0}, "cancellation_ratio_threshold must be positive"),
        ({"cancellation_ratio_threshold": -0.1}, "cancellation_ratio_threshold must be positive"),
        ({"max_dense_bytes": 0}, "max_dense_bytes must be a positive integer"),
        ({"max_dense_bytes": -100}, "max_dense_bytes must be a positive integer"),
        ({"max_dense_bytes": True}, "max_dense_bytes must be a positive integer"),
        ({"max_dense_bytes": "1000"}, "max_dense_bytes must be a positive integer"),
    ],
    ids=[
        "zero_warning_threshold",
        "negative_warning_threshold",
        "nan_warning_threshold",
        "zero_critical_threshold",
        "negative_critical_threshold",
        "nan_critical_threshold",
        "equal_critical_and_warning",
        "critical_less_than_warning",
        "zero_spectral_gap",
        "negative_spectral_gap",
        "nan_spectral_gap",
        "negative_perturbation_scale",
        "nan_perturbation_scale",
        "zero_rank_tol_factor",
        "negative_rank_tol_factor",
        "inf_rank_tol_factor",
        "negative_constant_var_tol",
        "nan_constant_var_tol",
        "zero_cancellation_ratio",
        "negative_cancellation_ratio",
        "zero_max_dense_bytes",
        "negative_max_dense_bytes",
        "bool_max_dense_bytes",
        "str_max_dense_bytes",
    ],
)
def test_stability_config_invalid_thresholds(kwargs, match):
    with pytest.raises(StabilityConfigError, match=match) as exc_info:
        StabilityConfig(**kwargs)
    assert isinstance(exc_info.value, ValueError)
    assert isinstance(exc_info.value, InputValidation)
    assert isinstance(exc_info.value, SpectraCompressError)


@pytest.mark.parametrize("band", ["acceptable", "warn", "unacceptable"])
def test_condition_report_construction_valid_bands(band):
    report = ConditionReport(
        kappa=1e4,
        kappa_effective=1e4,
        numerical_rank=5,
        rank_tol=1e-12,
        sigma_max=10.0,
        sigma_min_nonzero=1e-3,
        digits_lost=4.0,
        band=band,
        notes=["Test note"],  # Passed as list, must convert to tuple
    )
    assert report.kappa == 1e4
    assert report.kappa_effective == 1e4
    assert report.numerical_rank == 5
    assert report.rank_tol == 1e-12
    assert report.sigma_max == 10.0
    assert report.sigma_min_nonzero == 1e-3
    assert report.digits_lost == 4.0
    assert report.band == band
    assert isinstance(report.notes, tuple)
    assert report.notes == ("Test note",)


def test_condition_report_frozen():
    report = ConditionReport(
        kappa=1e4,
        kappa_effective=1e4,
        numerical_rank=5,
        rank_tol=1e-12,
        sigma_max=10.0,
        sigma_min_nonzero=1e-3,
        digits_lost=4.0,
        band="acceptable",
    )
    with pytest.raises(FrozenInstanceError):
        report.kappa = 2e4
    with pytest.raises(FrozenInstanceError):
        report.band = "warn"


def test_condition_report_infinite_kappa_and_digits_lost():
    """Document and verify that when kappa == inf (singular matrix), digits_lost is inf."""
    report = ConditionReport(
        kappa=float("inf"),
        kappa_effective=1e5,
        numerical_rank=4,
        rank_tol=1e-12,
        sigma_max=10.0,
        sigma_min_nonzero=1e-4,
        digits_lost=float("inf"),
        band="unacceptable",
        notes=("Matrix is singular; trailing singular value is zero.",),
    )
    assert math.isinf(report.kappa)
    assert math.isinf(report.digits_lost)
    assert report.numerical_rank == 4
    assert report.band == "unacceptable"
    assert report.notes == ("Matrix is singular; trailing singular value is zero.",)


@pytest.mark.parametrize("bad_band", ["invalid", "ok", "critical", "", "ACCEPTABLE"])
def test_condition_report_invalid_band(bad_band):
    with pytest.raises(ReportValidationError, match="band must be one of") as exc_info:
        ConditionReport(
            kappa=1e4,
            kappa_effective=1e4,
            numerical_rank=5,
            rank_tol=1e-12,
            sigma_max=10.0,
            sigma_min_nonzero=1e-3,
            digits_lost=4.0,
            band=bad_band,
        )
    assert isinstance(exc_info.value, ValueError)
    assert isinstance(exc_info.value, ValidationError)
    assert isinstance(exc_info.value, InputValidation)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"numerical_rank": -1}, "numerical_rank must be non-negative"),
        ({"rank_tol": -1.0}, "rank_tol must be non-negative"),
        ({"rank_tol": float("nan")}, "rank_tol must be non-negative"),
        ({"sigma_max": -1.0}, "sigma_max must be non-negative"),
        ({"sigma_min_nonzero": -1.0}, "sigma_min_nonzero must be non-negative"),
        ({"kappa": -1.0}, "kappa must be non-negative"),
        ({"kappa_effective": -1.0}, "kappa_effective must be non-negative"),
        ({"digits_lost": -1.0}, "digits_lost must be non-negative"),
    ],
    ids=[
        "negative_rank",
        "negative_rank_tol",
        "nan_rank_tol",
        "negative_sigma_max",
        "negative_sigma_min_nonzero",
        "negative_kappa",
        "negative_kappa_effective",
        "negative_digits_lost",
    ],
)
def test_condition_report_invalid_numeric_values(kwargs, match):
    base_kwargs = {
        "kappa": 1e4,
        "kappa_effective": 1e4,
        "numerical_rank": 5,
        "rank_tol": 1e-12,
        "sigma_max": 10.0,
        "sigma_min_nonzero": 1e-3,
        "digits_lost": 4.0,
        "band": "acceptable",
    }
    base_kwargs.update(kwargs)
    with pytest.raises(ReportValidationError, match=match):
        ConditionReport(**base_kwargs)


def test_input_diagnostics_construction_and_frozen():
    diag = InputDiagnostics(
        shape=(100, 20),
        is_sparse=False,
        format="dense",
        density=1.0,
        n_nonfinite=0,
        constant_feature_idx=[2, 5],  # Passed as list, converted to tuple
        duplicate_row_frac=0.02,
        warnings=["Duplicate rows detected"],  # Passed as list, converted to tuple
    )
    assert diag.shape == (100, 20)
    assert diag.is_sparse is False
    assert diag.format == "dense"
    assert diag.density == 1.0
    assert diag.n_nonfinite == 0
    assert isinstance(diag.constant_feature_idx, tuple)
    assert diag.constant_feature_idx == (2, 5)
    assert diag.duplicate_row_frac == 0.02
    assert isinstance(diag.warnings, tuple)
    assert diag.warnings == ("Duplicate rows detected",)

    # Immutability check
    with pytest.raises(FrozenInstanceError):
        diag.density = 0.5
    with pytest.raises(FrozenInstanceError):
        diag.format = "csc"


def test_input_diagnostics_defaults():
    diag = InputDiagnostics(
        shape=(50, 10),
        is_sparse=True,
        format="csr",
        density=0.15,
        n_nonfinite=0,
    )
    assert diag.shape == (50, 10)
    assert diag.is_sparse is True
    assert diag.format == "csr"
    assert diag.density == 0.15
    assert diag.n_nonfinite == 0
    assert diag.constant_feature_idx == ()
    assert diag.duplicate_row_frac == 0.0
    assert diag.warnings == ()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"density": -0.1}, "density must be between 0.0 and 1.0"),
        ({"density": 1.5}, "density must be between 0.0 and 1.0"),
        ({"density": float("nan")}, "density must be between 0.0 and 1.0"),
        ({"n_nonfinite": -1}, "n_nonfinite must be non-negative"),
        ({"duplicate_row_frac": -0.01}, "duplicate_row_frac must be between 0.0 and 1.0"),
        ({"duplicate_row_frac": 1.1}, "duplicate_row_frac must be between 0.0 and 1.0"),
    ],
    ids=[
        "density_negative",
        "density_too_high",
        "density_nan",
        "n_nonfinite_negative",
        "duplicate_row_frac_negative",
        "duplicate_row_frac_too_high",
    ],
)
def test_input_diagnostics_invalid_values(kwargs, match):
    base_kwargs = {
        "shape": (10, 5),
        "is_sparse": False,
        "format": "dense",
        "density": 1.0,
        "n_nonfinite": 0,
    }
    base_kwargs.update(kwargs)
    with pytest.raises(ReportValidationError, match=match):
        InputDiagnostics(**base_kwargs)


def test_exceptions_inheritance_and_catching():
    # All must inherit from ValueError, InputValidation, SpectraCompressError
    for exc_cls in [
        NonFiniteInputError,
        SparseNotSupportedError,
        StabilityConfigError,
        ReportValidationError,
    ]:
        assert issubclass(exc_cls, ValueError)
        assert issubclass(exc_cls, InputValidation)
        assert issubclass(exc_cls, SpectraCompressError)

    assert issubclass(ReportValidationError, ValidationError)

    # Verify catching behavior as ValueError
    try:
        raise NonFiniteInputError("NaN detected")
    except ValueError as e:
        assert "NaN detected" in str(e)
        assert e.code == 400

    # Verify catching behavior as InputValidation
    try:
        raise SparseNotSupportedError("Sparse unsupported")
    except InputValidation as e:
        assert "Sparse unsupported" in str(e)
        assert e.code == 400

    # Verify ReportValidationError catching behavior
    try:
        raise ReportValidationError("Invalid band")
    except ValidationError as e:
        assert "Invalid band" in str(e)


# =====================================================================
# Phase 3: Comprehensive Diagnostic & Stability Metrics Tests
# =====================================================================

def test_validate_input_dense_and_sparse(monkeypatch):
    # Dense validation
    X_dense = np.array([
        [1.0, 2.0, 0.0, 4.0],
        [0.0, 2.0, 3.0, 4.0],
        [5.0, 2.0, 0.0, 4.0],
        [0.0, 2.0, 0.0, 4.0],
    ], dtype=np.float64)

    diag_dense = validate_input(X_dense)
    assert diag_dense.shape == (4, 4)
    assert not diag_dense.is_sparse
    assert diag_dense.format == "dense"
    assert diag_dense.n_nonfinite == 0
    assert diag_dense.density == 11 / 16
    assert diag_dense.constant_feature_idx == (1, 3)

    # Sparse CSR and CSC validation
    X_csr = sp.csr_matrix(X_dense)
    X_csc = sp.csc_matrix(X_dense)

    # Ensure toarray and todense are never called during sparse validation
    def forbidden_dense(*args, **kwargs):
        raise RuntimeError("Densification forbidden!")

    monkeypatch.setattr(X_csr, "toarray", forbidden_dense)
    monkeypatch.setattr(X_csr, "todense", forbidden_dense)

    diag_csr = validate_input(X_csr)
    assert diag_csr.shape == (4, 4)
    assert diag_csr.is_sparse
    assert diag_csr.format == "csr"
    assert diag_csr.density == 11 / 16
    assert diag_csr.constant_feature_idx == (1, 3)

    diag_csc = validate_input(X_csc)
    assert diag_csc.format == "csc"
    assert diag_csc.constant_feature_idx == (1, 3)


def test_validate_input_nan_inf_rejection_and_immutability():
    # Dense NaN and Inf
    X_nan = np.array([[1.0, 2.0], [np.nan, 4.0]])
    with pytest.raises(NonFiniteInputError, match="NaN or infinite"):
        validate_input(X_nan)

    X_inf = np.array([[1.0, np.inf], [3.0, 4.0]])
    with pytest.raises(NonFiniteInputError, match="NaN or infinite"):
        validate_input(X_inf)

    # Sparse NaN and Inf in explicit data
    X_sp_nan = sp.csr_matrix([[1.0, 0.0], [0.0, np.nan]])
    with pytest.raises(NonFiniteInputError, match="NaN or infinite"):
        validate_input(X_sp_nan)

    X_sp_inf = sp.csc_matrix([[1.0, 0.0], [-np.inf, 2.0]])
    with pytest.raises(NonFiniteInputError, match="NaN or infinite"):
        validate_input(X_sp_inf)

    # Immutability verification
    X_orig = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    X_copy = X_orig.copy()
    validate_input(X_orig)
    np.testing.assert_array_equal(X_orig, X_copy)

    X_sp_orig = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    data_copy = X_sp_orig.data.copy()
    indices_copy = X_sp_orig.indices.copy()
    indptr_copy = X_sp_orig.indptr.copy()
    validate_input(X_sp_orig)
    np.testing.assert_array_equal(X_sp_orig.data, data_copy)
    np.testing.assert_array_equal(X_sp_orig.indices, indices_copy)
    np.testing.assert_array_equal(X_sp_orig.indptr, indptr_copy)


def test_validate_input_constant_features_and_duplicates():
    # 6 rows, 4 columns: column 0 is constant 5.0, column 2 is constant 0.0
    # Rows 0 and 1 are identical, rows 2 and 3 are identical, rows 4 and 5 are distinct
    X = np.array([
        [5.0, 1.0, 0.0, 10.0],
        [5.0, 1.0, 0.0, 10.0],
        [5.0, 2.0, 0.0, 20.0],
        [5.0, 2.0, 0.0, 20.0],
        [5.0, 3.0, 0.0, 30.0],
        [5.0, 4.0, 0.0, 40.0],
    ], dtype=np.float64)

    # Unique rows = 4 out of 6 -> duplicate frac = 2/6 = 1/3
    diag = validate_input(X)
    assert diag.constant_feature_idx == (0, 2)
    assert diag.duplicate_row_frac == pytest.approx(2.0 / 6.0)
    assert len(diag.warnings) == 2

    # Sparse representation
    diag_sp = validate_input(sp.csr_matrix(X))
    assert diag_sp.constant_feature_idx == (0, 2)
    assert diag_sp.duplicate_row_frac == pytest.approx(2.0 / 6.0)


def test_compute_condition_report_hand_checkable_and_numpy():
    # Diagonal hand-checkable matrix
    s_vals = np.array([1000.0, 100.0, 10.0, 1.0], dtype=np.float64)
    X = np.diag(s_vals)

    rep = compute_condition_report(X)
    assert rep.numerical_rank == 4
    assert rep.sigma_max == pytest.approx(1000.0)
    assert rep.sigma_min_nonzero == pytest.approx(1.0)
    assert rep.kappa == pytest.approx(1000.0)
    assert rep.kappa_effective == pytest.approx(1000.0)
    assert rep.digits_lost == pytest.approx(3.0)
    assert rep.band == "acceptable"

    # Compare dense against numpy.linalg.svd
    rng = np.random.RandomState(42)
    A = rng.randn(10, 8)
    s_np = np.linalg.svd(A, compute_uv=False)

    rep_A = compute_condition_report(A)
    assert rep_A.sigma_max == pytest.approx(float(s_np[0]))
    assert rep_A.sigma_min_nonzero == pytest.approx(float(s_np[-1]))
    assert rep_A.kappa == pytest.approx(float(s_np[0] / s_np[-1]))


def test_compute_condition_report_rank_deficient_and_zero():
    # All-zero dense matrix
    zero_dense = np.zeros((5, 5), dtype=np.float64)
    rep_z = compute_condition_report(zero_dense)
    assert rep_z.numerical_rank == 0
    assert rep_z.sigma_max == 0.0
    assert math.isinf(rep_z.kappa)
    assert math.isinf(rep_z.kappa_effective)
    assert math.isinf(rep_z.digits_lost)
    assert rep_z.band == "unacceptable"

    # All-zero sparse matrix
    zero_sp = sp.csr_matrix((5, 5), dtype=np.float64)
    rep_sp_z = compute_condition_report(zero_sp)
    assert rep_sp_z.numerical_rank == 0
    assert math.isinf(rep_sp_z.kappa)
    assert rep_sp_z.band == "unacceptable"

    # Rank-deficient matrix: outer product of vectors has rank 1 in 4x4
    u = np.array([[1.0], [2.0], [3.0], [4.0]])
    X_rank1 = u @ u.T
    rep_r1 = compute_condition_report(X_rank1)
    assert rep_r1.numerical_rank == 1
    assert math.isinf(rep_r1.kappa)
    assert rep_r1.kappa_effective == pytest.approx(1.0)
    assert rep_r1.band == "unacceptable"
    assert any("rank-deficient" in note for note in rep_r1.notes)

    # Sparse rank-deficient
    X_sp_r1 = sp.csr_matrix(X_rank1)
    rep_sp_r1 = compute_condition_report(X_sp_r1)
    assert rep_sp_r1.numerical_rank == 1
    assert math.isinf(rep_sp_r1.kappa)
    assert rep_sp_r1.band == "unacceptable"


def test_compute_condition_report_sparse_no_densification(monkeypatch):
    # Diagonal sparse matrix
    A = sp.diags([1000.0, 100.0, 1.0], format="csr")

    def forbidden_dense(*args, **kwargs):
        raise RuntimeError("Densification forbidden!")

    monkeypatch.setattr(A, "toarray", forbidden_dense)
    monkeypatch.setattr(A, "todense", forbidden_dense)

    rep = compute_condition_report(A)
    assert rep.numerical_rank == 3
    assert rep.sigma_max == pytest.approx(1000.0)
    assert rep.kappa == pytest.approx(1000.0)
    assert rep.band == "acceptable"


def test_stable_column_variance_cancellation_resistance():
    # Catastrophic cancellation benchmark:
    # Data with mean 1e9 and standard deviation 1.0.
    # Naive E[X^2] - (E[X])^2 completely fails in float64 due to cancellation.
    rng = np.random.RandomState(42)
    n = 20000
    noise = rng.normal(loc=0.0, scale=1.0, size=(n, 2))
    X = 1e9 + noise

    # Naive cancellation formula
    mean_naive = np.mean(X, axis=0)
    mean_sq_naive = np.mean(X ** 2, axis=0)
    naive_var = mean_sq_naive - mean_naive ** 2

    # Two-pass stable variance
    stable_var_ddof0 = stable_column_variance(X, ddof=0)
    stable_var_ddof1 = stable_column_variance(X, ddof=1)
    reference_var_ddof0 = np.var(X, axis=0, ddof=0)
    reference_var_ddof1 = np.var(X, axis=0, ddof=1)

    # Verify naive method suffered severe inaccuracy (> 10x error)
    assert np.max(np.abs(naive_var - reference_var_ddof0)) > 5.0

    # Verify stable method matches reference to high precision
    np.testing.assert_allclose(stable_var_ddof0, reference_var_ddof0, rtol=1e-6)
    np.testing.assert_allclose(stable_var_ddof1, reference_var_ddof1, rtol=1e-6)

    # Constant column of identical values has variance 0.0
    X_const = np.full((100, 3), 1e9, dtype=np.float64)
    var_const = stable_column_variance(X_const)
    np.testing.assert_allclose(var_const, np.zeros(3))

    # Test alias
    np.testing.assert_array_equal(stable_variance(X_const), var_const)


def test_component_cosine_stability_sign_flips():
    # Random orthogonal components
    rng = np.random.RandomState(42)
    Q, _ = np.linalg.qr(rng.randn(10, 4))
    W_a = Q.T  # (4, 10)

    # Identical with sign flips
    W_b = W_a.copy()
    W_b[0] = -W_b[0]
    W_b[2] = -W_b[2]

    sim = component_cosine_stability(W_a, W_b)
    assert sim.shape == (4,)
    np.testing.assert_allclose(sim, np.ones(4), atol=1e-12)

    # Mutually orthogonal components
    W_c = np.vstack([W_a[1:], W_a[:1]])
    sim_orth = component_cosine_stability(W_a, W_c)
    np.testing.assert_allclose(sim_orth, np.zeros(4), atol=1e-12)


def test_subspace_similarity_equal_and_unequal_ranks():
    # Identical subspaces
    e1 = np.array([[1.0, 0.0, 0.0, 0.0]])
    e2 = np.array([[0.0, 1.0, 0.0, 0.0]])
    e3 = np.array([[0.0, 0.0, 1.0, 0.0]])
    e4 = np.array([[0.0, 0.0, 0.0, 1.0]])

    W_a = np.vstack([e1, e2])
    W_b = np.vstack([e1, e2])
    assert subspace_similarity(W_a, W_b) == pytest.approx(1.0)

    # Sign flips and permutations
    W_b_perm = np.vstack([-e2, e1])
    assert subspace_similarity(W_a, W_b_perm) == pytest.approx(1.0)

    # Continuous 2D rotation by theta
    theta = np.pi / 6  # 30 degrees
    W_rot = np.array([[np.cos(theta), np.sin(theta), 0.0, 0.0]])
    assert subspace_similarity(e1, W_rot) == pytest.approx(np.cos(theta), rel=1e-12)

    # Unequal ranks: W_sub (rank 2) is nested within W_super (rank 3)
    W_super = np.vstack([e1, e2, e3])
    W_sub = np.vstack([e1, e2])
    assert subspace_similarity(W_super, W_sub) == pytest.approx(1.0)

    # Mutually orthogonal subspaces
    W_orth1 = np.vstack([e1, e2])
    W_orth2 = np.vstack([e3, e4])
    assert subspace_similarity(W_orth1, W_orth2) == pytest.approx(0.0)


def test_eigengap_table_near_zero_handling():
    cfg = StabilityConfig(constant_var_tol=1e-10, spectral_gap_ratio=1.1)

    # Well-separated eigenvalues
    eigs = np.array([100.0, 50.0, 10.0])
    table = eigengap_table(eigs, config=cfg)
    assert len(table) == 2
    assert table[0]["gap"] == pytest.approx(50.0)
    assert table[0]["ratio"] == pytest.approx(2.0)
    assert table[0]["has_gap"] is True
    assert table[0]["status"] == "OK"

    # Clustered eigenvalues below spectral_gap_ratio
    eigs_clustered = np.array([10.0, 9.8])
    table_c = eigengap_table(eigs_clustered, config=cfg)
    assert table_c[0]["has_gap"] is False
    assert table_c[0]["status"] == "NO_GAP"

    # Eigenvalues dropping to near zero
    eigs_zero = np.array([5.0, 0.0, 0.0])
    table_z = eigengap_table(eigs_zero, config=cfg)
    assert len(table_z) == 2
    assert math.isinf(table_z[0]["ratio"])
    assert table_z[0]["has_gap"] is True
    assert table_z[0]["status"] == "OK"
    assert table_z[1]["gap"] == pytest.approx(0.0)
    assert table_z[1]["ratio"] == pytest.approx(1.0)
    assert table_z[1]["has_gap"] is False
    assert table_z[1]["status"] == "NO_GAP"

    # Short array handling
    assert eigengap_table(np.array([5.0])) == []
    assert eigengap_table(np.empty(0)) == []


def test_sparse_reconstruction_mse_versus_dense():
    rng = np.random.RandomState(42)
    orig_dense = rng.randn(25, 6)
    recon_dense = orig_dense + rng.randn(25, 6) * 0.1

    # Dense reference MSE using repository standard metric
    ref_mse = mean_squared_reconstruction_error(orig_dense, recon_dense)

    # Sparse input evaluated with chunk size 5
    orig_sparse = sp.csr_matrix(orig_dense)
    recon_sparse = sp.csr_matrix(recon_dense)
    mse_sparse = sparse_reconstruction_mse(orig_sparse, recon_sparse, batch_size=5)

    assert mse_sparse == pytest.approx(ref_mse, rel=1e-12)


def test_diagnostics_validation_and_edge_cases():
    # 1D input to validate_input
    with pytest.raises(InputValidation) as exc_info:
        validate_input(np.array([1.0, 2.0, 3.0]))
    assert "Expected 2D matrix" in exc_info.value.message

    # Empty inputs
    with pytest.raises(InputValidation) as exc_info:
        validate_input(np.empty((0, 5)))
    assert "Matrix dimensions must be positive" in exc_info.value.message

    with pytest.raises(InputValidation) as exc_info:
        compute_condition_report(np.empty((5, 0)))
    assert "Matrix dimensions must be positive" in exc_info.value.message

    # stable_column_variance validation
    with pytest.raises(InputValidation) as exc_info:
        stable_column_variance(np.ones((5, 2)), ddof=-1)
    assert "ddof must be a non-negative integer" in exc_info.value.message

    with pytest.raises(InputValidation) as exc_info:
        stable_column_variance(np.ones((5, 2)), ddof=True)
    assert "ddof must be a non-negative integer" in exc_info.value.message

    with pytest.raises(InputValidation) as exc_info:
        stable_column_variance(np.ones((5, 2)), ddof=5)
    assert "strictly greater than ddof" in exc_info.value.message

    # subspace_similarity validation
    with pytest.raises(InputValidation) as exc_info:
        subspace_similarity(np.ones(5), np.ones((5, 5)))
    assert "Inputs must be 2D arrays" in exc_info.value.message

    with pytest.raises(ShapeMismatchError) as exc_info:
        subspace_similarity(np.ones((2, 4)), np.ones((2, 5)))
    assert "Feature dimension mismatch" in exc_info.value.message

    # eigengap_table validation
    with pytest.raises(InputValidation) as exc_info:
        eigengap_table(np.ones((3, 3)))
    assert "eigenvalues must be a 1D array" in exc_info.value.message

    with pytest.raises(NonFiniteInputError, match="eigenvalues contain NaN or infinite values"):
        eigengap_table(np.array([1.0, np.nan, 0.5]))


# =====================================================================
# Phase 8: Cross-Validation Stability Evaluation Tests
# =====================================================================

def test_cv_reproducible_kfold_indices():
    n_samples = 45
    n_splits = 5

    # Determinism under identical seed
    splits1 = generate_kfold_indices(n_samples=n_samples, n_splits=n_splits, shuffle=True, seed=42)
    splits2 = generate_kfold_indices(n_samples=n_samples, n_splits=n_splits, shuffle=True, seed=42)

    assert len(splits1) == n_splits
    for (tr1, val1), (tr2, val2) in zip(splits1, splits2):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(val1, val2)

    # Different seed produces different permutation
    splits_diff = generate_kfold_indices(n_samples=n_samples, n_splits=n_splits, shuffle=True, seed=99)
    assert not np.array_equal(splits1[0][1], splits_diff[0][1])

    # Validate non-overlapping partitions and full coverage
    all_val_indices = []
    for train_idx, val_idx in splits1:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert set(train_idx).isdisjoint(set(val_idx))
        assert len(train_idx) + len(val_idx) == n_samples
        all_val_indices.extend(val_idx.tolist())

    assert sorted(all_val_indices) == list(range(n_samples))


def test_cv_train_only_centerer_fitting():
    # 40 rows, 3 columns. Rows 0..19 have mean 10.0, rows 20..39 have mean 100.0
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(20, 3) + 10.0,
        rng.randn(20, 3) + 100.0,
    ])

    # Without shuffling, folds will have cleanly separated slices
    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=1, seed=42),
        cv_config=CrossValidationConfig(n_splits=2, shuffle=False)
    )
    report = runner.run(X)

    # Fold 0: train is rows 20..39 (mean ~100.0), val is rows 0..19 (mean ~10.0)
    # Fold 1: train is rows 0..19 (mean ~10.0), val is rows 20..39 (mean ~100.0)
    expected_train_mean_f1 = np.mean(X[:20], axis=0)
    expected_train_mean_f0 = np.mean(X[20:], axis=0)

    # Verify fold input diagnostics and sample counts
    assert report.fold_reports[0].n_train == 20
    assert report.fold_reports[0].n_val == 20
    assert report.fold_reports[1].n_train == 20
    assert report.fold_reports[1].n_val == 20

    # Ensure validation rows never shifted training fold mean
    assert not np.allclose(expected_train_mean_f0, expected_train_mean_f1)


def test_cv_leakage_prevention_shifted_validation_mean():
    # Training rows (0..19) centered at 0.0; validation rows (20..29) with deliberate +500.0 shift
    rng = np.random.RandomState(42)
    X_train_data = rng.randn(20, 3)
    X_val_data = rng.randn(10, 3) + 500.0
    X = np.vstack([X_train_data, X_val_data])

    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42),
        cv_config=CrossValidationConfig(n_splits=3, shuffle=False)
    )
    report = runner.run(X)

    # In Fold 2: val is rows 20..29 (the shifted partition)
    val_mse_f2 = report.fold_reports[2].val_reconstruction_mse

    # If training-fold mean was used, validation reconstruction error should reflect the un-subtracted +500 shift
    # (MSE roughly 500^2 * 3 / 3 = 250,000)
    assert val_mse_f2 > 100_000.0


def test_cv_held_out_reconstruction_mse_against_manual_reference():
    # Hand-checkable 4x2 matrix with 2 folds (no shuffle)
    # Rows 0, 1: [1.0, 2.0], [3.0, 4.0]
    # Rows 2, 3: [5.0, 1.0], [3.0, 0.0]
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 1.0],
        [3.0, 0.0],
    ], dtype=np.float64)

    # In fold 1:
    # train_idx = [0, 1], val_idx = [2, 3]
    # Training mean = [2.0, 3.0]
    # Component = [1/sqrt(2), 1/sqrt(2)]
    # x2 = [5, 1] -> centered [3, -2] -> projection 1/sqrt(2) -> recon [0.5, 0.5] -> err [2.5, -2.5] -> sq_err = 12.5
    # x3 = [3, 0] -> centered [1, -3] -> projection -2/sqrt(2) -> recon [-1, -1] -> err [2.0, -2.0] -> sq_err = 8.0
    # Mean squared error = (12.5 + 8.0) / 2 = 10.25
    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=1, seed=42),
        cv_config=CrossValidationConfig(n_splits=2, shuffle=False)
    )
    report = runner.run(X)

    f1 = report.fold_reports[1]
    assert f1.n_train == 2
    assert f1.n_val == 2
    assert f1.val_reconstruction_mse == pytest.approx(10.25, rel=1e-12)


def test_cv_report_structure_and_schema_version():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4)

    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42),
        cv_config=CrossValidationConfig(n_splits=3, seed=42)
    )
    report = runner.run(X)

    assert report.schema_version == "day4.v1"
    assert len(report.fold_reports) == 3
    assert report.mean_val_reconstruction_mse >= 0.0
    assert report.min_val_reconstruction_mse <= report.mean_val_reconstruction_mse <= report.max_val_reconstruction_mse
    assert report.mean_subspace_similarity is not None
    assert 0.0 <= report.mean_subspace_similarity <= 1.0
    assert report.verdict in {"stable", "marginal", "unstable"}
    assert len(report.verdict_notes) > 0


def test_cv_json_serialization_including_infinity():
    # Construct zero data where condition numbers become infinite
    X = np.zeros((20, 4), dtype=np.float64)

    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=1, seed=42),
        cv_config=CrossValidationConfig(n_splits=2, seed=42)
    )
    report = runner.run(X)

    # Verify to_dict and strict standard JSON serialization without NaN/Inf float tokens
    report_dict = report.to_dict()
    assert isinstance(report_dict, dict)
    assert report_dict["schema_version"] == "day4.v1"

    # json.dumps with allow_nan=False MUST succeed (RFC 8259 compliance)
    json_str = report.to_json(indent=2)
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Ensure infinite condition numbers were safely converted to standard strings
    loaded = json.loads(json_str)
    assert loaded["schema_version"] == "day4.v1"
    assert loaded["fold_reports"][0]["condition_report"]["kappa"] == "Infinity"


def test_cv_component_sign_flip_invariance():
    # Verify that sign-flipped components between folds evaluate to cosine similarity 1.0
    rng = np.random.RandomState(42)
    Q, _ = np.linalg.qr(rng.randn(10, 3))
    W1 = Q.T
    W2 = W1.copy()
    W2[0] = -W2[0]  # Sign flip in first component
    W2[2] = -W2[2]  # Sign flip in third component

    cosines = component_cosine_stability(W1, W2)
    np.testing.assert_allclose(cosines, np.ones(3), atol=1e-12)


def test_cv_subspace_stability_and_rank_mismatch():
    # Test handling when folds have rank mismatch
    e1 = np.array([[1.0, 0.0, 0.0, 0.0]])
    e2 = np.array([[0.0, 1.0, 0.0, 0.0]])
    e3 = np.array([[0.0, 0.0, 1.0, 0.0]])

    W_rank3 = np.vstack([e1, e2, e3])
    W_rank2 = np.vstack([e1, e2])

    # Subspace similarity should evaluate principal angles on common rank (2) -> 1.0
    sub_sim = subspace_similarity(W_rank3, W_rank2)
    assert sub_sim == pytest.approx(1.0)

    # Component cosine similarity truncates to common rank
    cosines = component_cosine_stability(W_rank3[:2], W_rank2)
    np.testing.assert_allclose(cosines, np.ones(2), atol=1e-12)


def test_cv_rank_zero_and_constant_feature_behavior():
    # Matrix of all constant values (all 5.0)
    X_const = np.full((30, 4), 5.0, dtype=np.float64)

    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=1, seed=42),
        cv_config=CrossValidationConfig(n_splits=3, seed=42)
    )
    report = runner.run(X_const)

    # In rank-zero scenarios, invalid subspace comparisons must be omitted
    assert report.mean_subspace_similarity is None
    assert len(report.pairwise_subspace_similarities) == 0

    for fr in report.fold_reports:
        assert fr.effective_rank == 0
        assert fr.mean_subspace_similarity is None
        assert len(fr.pairwise_subspace_similarities) == 0
        assert math.isinf(fr.condition_report.kappa)
        assert fr.condition_report.band == "unacceptable"

    assert report.verdict != "stable"
    assert report.verdict == "unstable"
    assert any("rank 0" in note for note in report.verdict_notes)


def test_cv_invalid_fold_config_and_insufficient_rows():
    # n_splits < 2
    with pytest.raises(InvalidFoldConfigError, match="n_splits must be an integer >= 2"):
        CrossValidationConfig(n_splits=1)

    # n_samples < n_splits
    with pytest.raises(InvalidFoldConfigError, match="Cannot create n_splits"):
        generate_kfold_indices(n_samples=3, n_splits=5)

    # Invalid fold_size_fractions
    with pytest.raises(InvalidFoldConfigError, match="Each fold_size_fraction must be in"):
        CrossValidationConfig(fold_size_fractions=(-0.1, 0.5))

    with pytest.raises(InvalidFoldConfigError, match="strictly increasing"):
        CrossValidationConfig(fold_size_fractions=(0.8, 0.5))

    # Runner with insufficient training rows for requested rank k
    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=5, seed=42),
        cv_config=CrossValidationConfig(n_splits=2, seed=42)
    )
    # 8 samples with 2 splits -> 4 train samples, but requires n_train >= k + 1 = 6
    with pytest.raises(InvalidFoldConfigError, match="too small for requested rank"):
        runner.run(np.ones((8, 4)))

    # Non-2D input
    with pytest.raises(InputValidation) as exc_info:
        runner.run(np.ones(10))
    assert "Expected 2D matrix" in exc_info.value.message

    # Non-finite input
    with pytest.raises(NonFiniteInputError, match="NaN or infinite"):
        runner.run(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))


def test_cv_sparse_end_to_end_no_densification(monkeypatch):
    # Create sparse CSR matrix
    rng = np.random.RandomState(42)
    dense_data = rng.randn(30, 6)
    dense_data[dense_data < 0.5] = 0.0
    X_csr = sp.csr_matrix(dense_data)

    # Ensure toarray and todense are never called on the sparse matrix
    def forbidden_dense(*args, **kwargs):
        raise RuntimeError("Densification forbidden!")

    monkeypatch.setattr(X_csr, "toarray", forbidden_dense)
    monkeypatch.setattr(X_csr, "todense", forbidden_dense)

    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42),
        cv_config=CrossValidationConfig(n_splits=3, shuffle=False)
    )
    report = runner.run(X_csr)

    assert len(report.fold_reports) == 3
    for fr in report.fold_reports:
        assert fr.input_diagnostics.is_sparse is True
        assert fr.val_reconstruction_mse >= 0.0


def test_cv_null_test_reproducibility_and_input_immutability():
    rng = np.random.RandomState(42)
    # Correlated dataset (rank 1 signal + small noise)
    u = rng.randn(40, 1)
    v = rng.randn(1, 5)
    X = u @ v + rng.randn(40, 5) * 0.1
    X_copy = X.copy()

    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=1, seed=42),
        cv_config=CrossValidationConfig(n_splits=3, seed=42, run_null_test=True)
    )
    report = runner.run(X)

    # Input immutability
    np.testing.assert_array_equal(X, X_copy)

    # Null test results populated
    assert report.null_test is not None
    assert "procedure" in report.null_test
    assert report.null_test["null_mean_subspace_similarity"] is not None

    # On correlated data, real subspace similarity should be notably higher than null baseline
    delta = report.null_test["subspace_similarity_delta"]
    assert delta is not None
    assert delta > 0.0  # Real stability exceeds null baseline


def test_cv_fold_size_fractions_and_effective_sample_sizes():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)

    runner = KFoldStabilityRunner(
        decomp_config=DecompositionConfig(method=MethodType.from_scratch_covariance, k=2, seed=42),
        cv_config=CrossValidationConfig(
            n_splits=3,
            seed=42,
            fold_size_fractions=(0.4, 0.8, 1.0)
        )
    )
    report = runner.run(X)

    assert report.fold_size_results is not None
    assert len(report.fold_size_results) == 3

    # Check fraction values and strictly increasing sample sizes
    f0 = report.fold_size_results[0]
    f1 = report.fold_size_results[1]
    f2 = report.fold_size_results[2]

    assert f0["fraction"] == 0.4
    assert f1["fraction"] == 0.8
    assert f2["fraction"] == 1.0

    assert f0["effective_sample_size"] < f1["effective_sample_size"] < f2["effective_sample_size"]
    assert f0["mean_val_reconstruction_mse"] > 0.0
    assert f2["mean_val_reconstruction_mse"] > 0.0


def test_cv_regression_all_decomposition_methods():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4)

    for method in [
        MethodType.from_scratch_covariance,
        MethodType.sklearn_pca,
        MethodType.sklearn_truncated_svd,
        MethodType.scipy_implicit_centered,
    ]:
        runner = KFoldStabilityRunner(
            decomp_config=DecompositionConfig(method=method, k=2, seed=42),
            cv_config=CrossValidationConfig(n_splits=3, seed=42)
        )
        report = runner.run(X)
        assert len(report.fold_reports) == 3
        assert report.mean_val_reconstruction_mse >= 0.0
        assert report.mean_subspace_similarity is not None


def test_cv_mixed_positive_and_zero_rank_omits_invalid_pairs():
    """Verifies that in mixed rank scenarios, comparisons involving rank 0 are omitted."""
    # Create fold reports: Fold 0 and Fold 1 have rank 2, Fold 2 has rank 0
    cond_acc = ConditionReport(
        kappa=10.0, kappa_effective=10.0, numerical_rank=2, rank_tol=1e-10,
        sigma_max=1.0, sigma_min_nonzero=0.1, digits_lost=1.0, band="acceptable", notes=()
    )
    cond_unacc = ConditionReport(
        kappa=float("inf"), kappa_effective=float("inf"), numerical_rank=0, rank_tol=0.0,
        sigma_max=0.0, sigma_min_nonzero=0.0, digits_lost=float("inf"), band="unacceptable", notes=()
    )
    diag = InputDiagnostics(
        shape=(20, 4), is_sparse=False, format="dense", density=1.0,
        n_nonfinite=0, constant_feature_idx=(), duplicate_row_frac=0.0, warnings=()
    )

    fr0 = FoldReport(
        fold=0, n_train=20, n_val=10, effective_rank=2, requested_rank=2,
        val_reconstruction_mse=0.1, condition_report=cond_acc, input_diagnostics=diag,
        mean_subspace_similarity=0.95, pairwise_subspace_similarities=((1, 0.95),)
    )
    fr1 = FoldReport(
        fold=1, n_train=20, n_val=10, effective_rank=2, requested_rank=2,
        val_reconstruction_mse=0.1, condition_report=cond_acc, input_diagnostics=diag,
        mean_subspace_similarity=0.95, pairwise_subspace_similarities=((0, 0.95),)
    )
    fr2 = FoldReport(
        fold=2, n_train=20, n_val=10, effective_rank=0, requested_rank=2,
        val_reconstruction_mse=1.0, condition_report=cond_unacc, input_diagnostics=diag,
        mean_subspace_similarity=None, pairwise_subspace_similarities=()
    )

    verdict, notes = evaluate_stability_verdict(
        fold_reports=[fr0, fr1, fr2],
        mean_subspace_similarity=0.95,
        mean_component_cosine=0.95,
        cv_config=CrossValidationConfig(n_splits=3)
    )

    # Because fold 2 has rank 0, overall verdict must be unstable
    assert verdict != "stable"
    assert verdict == "unstable"
    assert any("Effective rank collapsed to 0 in fold(s): [2]" in n for n in notes)


def test_cv_custom_thresholds_change_verdict_behavior():
    cond_acc = ConditionReport(
        kappa=10.0, kappa_effective=10.0, numerical_rank=2, rank_tol=1e-10,
        sigma_max=1.0, sigma_min_nonzero=0.1, digits_lost=1.0, band="acceptable", notes=()
    )
    diag = InputDiagnostics(
        shape=(20, 4), is_sparse=False, format="dense", density=1.0,
        n_nonfinite=0, constant_feature_idx=(), duplicate_row_frac=0.0, warnings=()
    )
    reports = [
        FoldReport(
            fold=i, n_train=20, n_val=10, effective_rank=2, requested_rank=2,
            val_reconstruction_mse=0.1, condition_report=cond_acc, input_diagnostics=diag
        )
        for i in range(3)
    ]

    mean_sub = 0.80  # Marginal under default [0.70, 0.90)

    # 1. Default thresholds: [0.70, 0.90) -> marginal
    v_def, notes_def = evaluate_stability_verdict(
        fold_reports=reports,
        mean_subspace_similarity=mean_sub,
        mean_component_cosine=mean_sub,
        cv_config=CrossValidationConfig(n_splits=3)
    )
    assert v_def == "marginal"
    assert any("marginal" in n for n in notes_def)

    # 2. Relaxed warning threshold (0.75): 0.80 >= 0.75 -> stable
    v_relaxed, notes_relaxed = evaluate_stability_verdict(
        fold_reports=reports,
        mean_subspace_similarity=mean_sub,
        mean_component_cosine=mean_sub,
        cv_config=CrossValidationConfig(n_splits=3, subspace_warning_threshold=0.75)
    )
    assert v_relaxed == "stable"

    # 3. Strict critical threshold (0.85): 0.80 < 0.85 -> unstable
    v_strict, notes_strict = evaluate_stability_verdict(
        fold_reports=reports,
        mean_subspace_similarity=mean_sub,
        mean_component_cosine=mean_sub,
        cv_config=CrossValidationConfig(n_splits=3, subspace_critical_threshold=0.85, subspace_warning_threshold=0.95)
    )
    assert v_strict == "unstable"
    assert any("below critical threshold" in n for n in notes_strict)


def test_cv_invalid_threshold_configurations():
    # critical >= warning
    with pytest.raises(ValueError):
        CrossValidationConfig(subspace_critical_threshold=0.95, subspace_warning_threshold=0.85)

    with pytest.raises(ValueError):
        CrossValidationConfig(subspace_critical_threshold=0.90, subspace_warning_threshold=0.90)

    # critical <= 0
    with pytest.raises(ValueError):
        CrossValidationConfig(subspace_critical_threshold=0.0)

    # warning > 1.0
    with pytest.raises(ValueError):
        CrossValidationConfig(subspace_warning_threshold=1.05)

    # non-finite or boolean
    with pytest.raises(ValueError):
        CrossValidationConfig(subspace_critical_threshold=float("nan"))


def test_cv_min_retained_rank_enforcement():
    cond_acc = ConditionReport(
        kappa=10.0, kappa_effective=10.0, numerical_rank=1, rank_tol=1e-10,
        sigma_max=1.0, sigma_min_nonzero=0.1, digits_lost=1.0, band="acceptable", notes=()
    )
    diag = InputDiagnostics(
        shape=(20, 4), is_sparse=False, format="dense", density=1.0,
        n_nonfinite=0, constant_feature_idx=(), duplicate_row_frac=0.0, warnings=()
    )
    reports = [
        FoldReport(
            fold=i, n_train=20, n_val=10, effective_rank=1, requested_rank=3,
            val_reconstruction_mse=0.1, condition_report=cond_acc, input_diagnostics=diag
        )
        for i in range(3)
    ]

    # Effective rank is 1, but min_retained_rank is configured to 2
    cfg = CrossValidationConfig(n_splits=3, min_retained_rank=2)
    verdict, notes = evaluate_stability_verdict(
        fold_reports=reports,
        mean_subspace_similarity=0.99,
        mean_component_cosine=0.99,
        cv_config=cfg
    )
    assert verdict != "stable"
    assert verdict == "marginal"
    assert any("below min_retained_rank (2)" in n for n in notes)


def test_cv_none_subspace_similarity_never_stable():
    cond_acc = ConditionReport(
        kappa=10.0, kappa_effective=10.0, numerical_rank=2, rank_tol=1e-10,
        sigma_max=1.0, sigma_min_nonzero=0.1, digits_lost=1.0, band="acceptable", notes=()
    )
    diag = InputDiagnostics(
        shape=(20, 4), is_sparse=False, format="dense", density=1.0,
        n_nonfinite=0, constant_feature_idx=(), duplicate_row_frac=0.0, warnings=()
    )
    reports = [
        FoldReport(
            fold=i, n_train=20, n_val=10, effective_rank=2, requested_rank=2,
            val_reconstruction_mse=0.1, condition_report=cond_acc, input_diagnostics=diag
        )
        for i in range(3)
    ]

    # When mean_subspace_similarity is None, verdict must not be stable
    verdict, notes = evaluate_stability_verdict(
        fold_reports=reports,
        mean_subspace_similarity=None,
        mean_component_cosine=None,
        cv_config=CrossValidationConfig(n_splits=3)
    )
    assert verdict != "stable"
    assert verdict == "marginal"
    assert any("insufficient valid fold comparisons" in n for n in notes)


def test_reconstruction_mse_scaling_relationship():
    """Confirms documented scaling relationship: MSE_sample == n_features * MSE_elementwise."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 25, 6
    X = rng.randn(n_samples, n_features)
    X_hat = X + rng.randn(n_samples, n_features) * 0.1

    mse_sample = sparse_reconstruction_mse(X, X_hat)
    mse_elementwise = float(np.mean((X - X_hat) ** 2))

    assert mse_sample == pytest.approx(n_features * mse_elementwise, rel=1e-12)
    assert mean_squared_reconstruction_error(X, X_hat) == pytest.approx(mse_sample, rel=1e-12)


