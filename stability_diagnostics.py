"""
Stability Diagnostics Module for PCA and Matrix Decompositions.

Day 4 Foundation: Configurations, Reports, Diagnostics Structures, and Exceptions.
Note: Numerical diagnostic calculation functions are not implemented in this phase.

Ownership Decisions:
- StabilityConfig is the single configuration owner for stability diagnostics,
  incorporating condition thresholds, spectral gap ratio, perturbation scale,
  rank tolerance factor, constant variance tolerance, cancellation ratio threshold,
  and memory limits (max_dense_bytes). No broad Day4Config is needed.
- InvalidFoldConfigError belongs to future cross-validation / data splitting modules
  (e.g., split_strategy.py) and is omitted here to avoid premature duplication.
- MemoryGuardError belongs to pipeline/feature layers (e.g., feature_pipeline.py or error.py)
  where dense memory allocation guards are enforced, and is not added prematurely.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, List, Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from error import InputValidation, SpectraCompressError, ValidationError, ShapeMismatchError
from feature_pipeline import sparse_column_mean_var
from eval.metrics import component_cosine_similarity, sparse_reconstruction_mse


class StabilityConfigError(ValueError, InputValidation):
    """Raised when stability configuration parameters are invalid."""
    def __init__(self, message: str = "Invalid stability configuration", code: int = 400):
        ValueError.__init__(self, message)
        InputValidation.__init__(self, message=message, code=code)


class ReportValidationError(ValueError, InputValidation, ValidationError):
    """Raised when diagnostic report values or band classifications are invalid."""
    def __init__(self, message: str = "Invalid report values", code: int = 400):
        ValueError.__init__(self, message)
        InputValidation.__init__(self, message=message, code=code)


class NonFiniteInputError(ValueError, InputValidation):
    """Raised when input matrix contains NaN or Inf values."""
    def __init__(self, message: str = "Input contains non-finite values (NaN or Inf)", code: int = 400):
        ValueError.__init__(self, message)
        InputValidation.__init__(self, message=message, code=code)


class SparseNotSupportedError(ValueError, InputValidation):
    """Raised when sparse matrix input is not supported for a specific operation."""
    def __init__(self, message: str = "Sparse matrices are not supported for this operation", code: int = 400):
        ValueError.__init__(self, message)
        InputValidation.__init__(self, message=message, code=code)


@dataclass(frozen=True)
class StabilityConfig:
    """Configuration for PCA stability diagnostics.

    Fields:
        cond_warning_threshold: Condition number threshold above which a warning is flagged (default: 1e8).
        cond_critical_threshold: Condition number threshold above which matrix is considered critical (default: 1e12).
        spectral_gap_ratio: Ratio lambda_k / lambda_{k+1} below which eigenvalues are considered clustered,
            indicating subspace sensitivity via the Davis-Kahan theorem (default: 1.05).
        perturbation_scale: Relative noise amplitude injected during perturbation diagnostics
            to quantify eigenvector / subspace stability (default: 1e-5).
        rank_tol_factor: Scale factor multiplying standard singular value tolerance
            (sigma_max * max(n, d) * eps) when estimating numerical rank (default: 1.0).
        constant_var_tol: Variance threshold below which a feature column is flagged as constant (default: 1e-12).
        cancellation_ratio_threshold: Ratio threshold for detecting catastrophic numerical cancellation
            during centering or sample covariance calculations (default: 1e-4).
        max_dense_bytes: Maximum memory in bytes allowed for dense operations or densifying sparse data
            to guard against Out-Of-Memory errors (default: 100_000_000 bytes ~100MB).
    """
    cond_warning_threshold: float = 1e8
    cond_critical_threshold: float = 1e12
    spectral_gap_ratio: float = 1.05
    perturbation_scale: float = 1e-5
    rank_tol_factor: Optional[float] = 1.0
    constant_var_tol: float = 1e-12
    cancellation_ratio_threshold: float = 1e-4
    max_dense_bytes: int = 100_000_000

    def __post_init__(self):
        if not np.isfinite(self.cond_warning_threshold) or self.cond_warning_threshold <= 0:
            raise StabilityConfigError(
                f"cond_warning_threshold must be positive and finite (> 0), got {self.cond_warning_threshold}"
            )
        if not np.isfinite(self.cond_critical_threshold) or self.cond_critical_threshold <= 0:
            raise StabilityConfigError(
                f"cond_critical_threshold must be positive and finite (> 0), got {self.cond_critical_threshold}"
            )
        if self.cond_critical_threshold <= self.cond_warning_threshold:
            raise StabilityConfigError(
                f"cond_critical_threshold ({self.cond_critical_threshold}) must be strictly greater than "
                f"cond_warning_threshold ({self.cond_warning_threshold})"
            )
        if not np.isfinite(self.spectral_gap_ratio) or self.spectral_gap_ratio <= 0:
            raise StabilityConfigError(
                f"spectral_gap_ratio must be positive and finite (> 0), got {self.spectral_gap_ratio}"
            )
        if not np.isfinite(self.perturbation_scale) or self.perturbation_scale < 0:
            raise StabilityConfigError(
                f"perturbation_scale must be non-negative and finite (>= 0), got {self.perturbation_scale}"
            )
        if self.rank_tol_factor is not None:
            if not np.isfinite(self.rank_tol_factor) or self.rank_tol_factor <= 0:
                raise StabilityConfigError(
                    f"rank_tol_factor must be positive and finite, got {self.rank_tol_factor}"
                )
        if not np.isfinite(self.constant_var_tol) or self.constant_var_tol < 0:
            raise StabilityConfigError(
                f"constant_var_tol must be non-negative and finite, got {self.constant_var_tol}"
            )
        if not np.isfinite(self.cancellation_ratio_threshold) or self.cancellation_ratio_threshold <= 0:
            raise StabilityConfigError(
                f"cancellation_ratio_threshold must be positive and finite, got {self.cancellation_ratio_threshold}"
            )
        if not isinstance(self.max_dense_bytes, int) or isinstance(self.max_dense_bytes, bool) or self.max_dense_bytes <= 0:
            raise StabilityConfigError(
                f"max_dense_bytes must be a positive integer, got {self.max_dense_bytes}"
            )


VALID_BANDS = {"acceptable", "warn", "unacceptable"}


@dataclass(frozen=True)
class ConditionReport:
    """Report summarizing numerical conditioning and rank diagnostics of a matrix.

    Fields:
        kappa: Direct condition number (sigma_max / sigma_min). Can be float('inf') for singular matrices.
        kappa_effective: Effective condition number using smallest non-zero singular value.
        numerical_rank: Estimated numerical rank (count of singular values > rank_tol).
        rank_tol: Tolerance threshold used to determine non-zero singular values.
        sigma_max: Maximum singular value.
        sigma_min_nonzero: Smallest singular value strictly above rank_tol.
        digits_lost: Estimated digits of numerical precision lost (approx log10(kappa)).
                     When kappa == inf (singular matrix), digits_lost is float('inf').
        band: Stability band classification: 'acceptable', 'warn', or 'unacceptable'.
        notes: Informational or diagnostic messages stored as an immutable tuple.
    """
    kappa: float
    kappa_effective: float
    numerical_rank: int
    rank_tol: float
    sigma_max: float
    sigma_min_nonzero: float
    digits_lost: float
    band: Literal["acceptable", "warn", "unacceptable"]
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        # Convert notes to tuple if a list/iterable was passed
        if not isinstance(self.notes, tuple):
            object.__setattr__(self, "notes", tuple(self.notes))

        if self.band not in VALID_BANDS:
            raise ReportValidationError(
                f"band must be one of {sorted(VALID_BANDS)}, got '{self.band}'"
            )

        if self.numerical_rank < 0:
            raise ReportValidationError(
                f"numerical_rank must be non-negative (>= 0), got {self.numerical_rank}"
            )

        if np.isnan(self.rank_tol) or self.rank_tol < 0:
            raise ReportValidationError(
                f"rank_tol must be non-negative, got {self.rank_tol}"
            )

        if np.isnan(self.sigma_max) or self.sigma_max < 0:
            raise ReportValidationError(
                f"sigma_max must be non-negative, got {self.sigma_max}"
            )

        if np.isnan(self.sigma_min_nonzero) or self.sigma_min_nonzero < 0:
            raise ReportValidationError(
                f"sigma_min_nonzero must be non-negative, got {self.sigma_min_nonzero}"
            )

        if np.isnan(self.kappa) or self.kappa < 0:
            raise ReportValidationError(
                f"kappa must be non-negative (or inf), got {self.kappa}"
            )

        if np.isnan(self.kappa_effective) or self.kappa_effective < 0:
            raise ReportValidationError(
                f"kappa_effective must be non-negative (or inf), got {self.kappa_effective}"
            )

        if np.isnan(self.digits_lost) or self.digits_lost < 0:
            raise ReportValidationError(
                f"digits_lost must be non-negative (or inf), got {self.digits_lost}"
            )


@dataclass(frozen=True)
class InputDiagnostics:
    """Report summarizing basic structural and sparsity properties of input data.

    Fields:
        shape: Tuple representing matrix dimensions (n_rows, n_cols).
        is_sparse: Whether the input is a SciPy sparse matrix.
        format: Format identifier (e.g., 'dense', 'csr', 'csc').
        density: Fraction of non-zero entries (nnz / (n * d)).
        n_nonfinite: Count of NaN, Inf, or -Inf entries.
        constant_feature_idx: Column indices with zero variance / constant values (stored as immutable tuple).
        duplicate_row_frac: Fraction of duplicate rows in the input.
        warnings: List of diagnostic warnings identified during inspection (stored as immutable tuple).
    """
    shape: tuple[int, ...]
    is_sparse: bool
    format: str
    density: float
    n_nonfinite: int
    constant_feature_idx: tuple[int, ...] = ()
    duplicate_row_frac: float = 0.0
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.constant_feature_idx, tuple):
            object.__setattr__(self, "constant_feature_idx", tuple(self.constant_feature_idx))
        if not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(self.warnings))

        if not np.isfinite(self.density) or self.density < 0.0 or self.density > 1.0:
            raise ReportValidationError(
                f"density must be between 0.0 and 1.0, got {self.density}"
            )
        if self.n_nonfinite < 0:
            raise ReportValidationError(
                f"n_nonfinite must be non-negative (>= 0), got {self.n_nonfinite}"
            )
        if not np.isfinite(self.duplicate_row_frac) or self.duplicate_row_frac < 0.0 or self.duplicate_row_frac > 1.0:
            raise ReportValidationError(
                f"duplicate_row_frac must be between 0.0 and 1.0, got {self.duplicate_row_frac}"
            )


# =====================================================================
# Numerical Stability Diagnostics & Analysis Functions (Day 4)
# =====================================================================

def validate_input(
    X: Union[np.ndarray, sp.spmatrix],
    config: Optional[StabilityConfig] = None
) -> InputDiagnostics:
    """Validates dense NumPy ndarrays or SciPy sparse matrices and returns an InputDiagnostics report.

    Checks:
    - Verifies 2D shape and positive dimensions.
    - Rejects NaN, Inf, or -Inf values with NonFiniteInputError.
    - Identifies constant features with variance <= constant_var_tol.
    - Estimates duplicate-row fraction without densifying sparse matrices.
    - Reports density and sparse format.
    - Does NOT mutate input or densify sparse inputs.

    Args:
        X: 2D NumPy array or SciPy sparse matrix.
        config: Optional StabilityConfig providing tolerance thresholds.

    Returns:
        InputDiagnostics: Immutable report containing input properties and warnings.
    """
    if not (isinstance(X, np.ndarray) or sp.issparse(X)):
        raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X)}")

    if X.ndim != 2:
        raise InputValidation(message=f"Expected 2D matrix, got ndim={X.ndim}")

    n, d = X.shape
    if n < 1 or d < 1:
        raise InputValidation(message=f"Matrix dimensions must be positive, got {X.shape}")

    is_sparse = sp.issparse(X)

    # Reject non-finite values (NaN / Inf)
    if is_sparse:
        if X.nnz > 0 and not np.isfinite(X.data).all():
            raise NonFiniteInputError(message="Sparse data contains NaN or infinite values")
    else:
        if not np.isfinite(X).all():
            raise NonFiniteInputError(message="Dense data contains NaN or infinite values")

    cfg = config if config is not None else StabilityConfig()
    tol = cfg.constant_var_tol

    fmt = "dense" if not is_sparse else ("csr" if sp.isspmatrix_csr(X) else ("csc" if sp.isspmatrix_csc(X) else "sparse"))
    density = float(X.nnz / (n * d)) if is_sparse else float(np.count_nonzero(X) / (n * d))

    # Variance and constant features without densifying
    if is_sparse:
        if n >= 2:
            _, vars_ = sparse_column_mean_var(X, ddof=1)
        else:
            vars_ = np.zeros(d, dtype=np.float64)
    else:
        if n >= 2:
            vars_ = np.var(X, axis=0, ddof=1, dtype=np.float64)
        else:
            vars_ = np.zeros(d, dtype=np.float64)

    constant_idx = tuple(int(i) for i in np.where(vars_ <= tol)[0])

    # Duplicate row fraction without densifying
    # Duplicate row fraction without densifying
    if is_sparse:
        csr = X.tocsr(copy=True)
        if not csr.has_canonical_format:
            csr.sum_duplicates()
            csr.sort_indices()
        csr.eliminate_zeros()
        seen = set()
        n_unique = 0
        for i in range(n):
            key = (
                tuple(csr.indices[csr.indptr[i] : csr.indptr[i + 1]]),
                tuple(csr.data[csr.indptr[i] : csr.indptr[i + 1]])
            )
            if key not in seen:
                seen.add(key)
                n_unique += 1
        duplicate_row_frac = float((n - n_unique) / n)
    else:
        n_unique = len(np.unique(X, axis=0))
        duplicate_row_frac = float((n - n_unique) / n)

    warnings_list = []
    if len(constant_idx) > 0:
        warnings_list.append(f"Detected {len(constant_idx)} constant features with variance <= {tol}.")
    if duplicate_row_frac > 0.0:
        warnings_list.append(f"Detected duplicate rows with duplicate fraction {duplicate_row_frac:.2%}.")

    return InputDiagnostics(
        shape=(n, d),
        is_sparse=is_sparse,
        format=fmt,
        density=density,
        n_nonfinite=0,
        constant_feature_idx=constant_idx,
        duplicate_row_frac=duplicate_row_frac,
        warnings=tuple(warnings_list)
    )


def compute_condition_report(
    X: Union[np.ndarray, sp.spmatrix],
    config: Optional[StabilityConfig] = None,
    k: Optional[int] = None
) -> ConditionReport:
    """Computes numerical conditioning and rank diagnostics for dense or sparse matrices.

    Args:
        X: 2D NumPy array or SciPy sparse matrix.
        config: Optional StabilityConfig providing condition thresholds and rank tolerance factor.
        k: Optional number of singular values to evaluate for large sparse inputs.

    Returns:
        ConditionReport: Report with condition numbers, numerical rank, and severity band.
    """
    if not (isinstance(X, np.ndarray) or sp.issparse(X)):
        raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X)}")

    if X.ndim != 2:
        raise InputValidation(message=f"Expected 2D matrix, got ndim={X.ndim}")

    n, d = X.shape
    if n < 1 or d < 1:
        raise InputValidation(message=f"Matrix dimensions must be positive, got {X.shape}")

    is_sparse = sp.issparse(X)
    if is_sparse:
        if X.nnz > 0 and not np.isfinite(X.data).all():
            raise NonFiniteInputError(message="Sparse data contains NaN or infinite values")
        is_all_zero = (X.nnz == 0) or bool(np.all(X.data == 0))
    else:
        if not np.isfinite(X).all():
            raise NonFiniteInputError(message="Dense data contains NaN or infinite values")
        is_all_zero = bool(np.all(X == 0))

    cfg = config if config is not None else StabilityConfig()
    rank_tol_factor = cfg.rank_tol_factor if cfg.rank_tol_factor is not None else 1.0

    # Explicit handling for all-zero matrices
    if is_all_zero:
        return ConditionReport(
            kappa=float("inf"),
            kappa_effective=float("inf"),
            numerical_rank=0,
            rank_tol=0.0,
            sigma_max=0.0,
            sigma_min_nonzero=0.0,
            digits_lost=float("inf"),
            band="unacceptable",
            notes=("Matrix is all zeros; numerical rank is 0.",)
        )

    notes_list = []
    min_dim = min(n, d)

    # Singular value extraction
    if not is_sparse:
        s = np.linalg.svd(X, compute_uv=False)
        s = np.sort(s)[::-1]
        sigma_max = float(s[0]) if len(s) > 0 else 0.0
        rank_tol = float(sigma_max * max(n, d) * np.finfo(np.float64).eps * rank_tol_factor)

        nonzero_sv = s[s > rank_tol]
        numerical_rank = int(len(nonzero_sv))
        sigma_min_nonzero = float(nonzero_sv[-1]) if numerical_rank > 0 else 0.0
        sigma_min = float(s[-1]) if numerical_rank == min_dim else 0.0
    else:
        # Sparse strategy: avoid full dense conversion
        if min_dim == 1:
            norm_val = float(np.sqrt(np.sum(X.data ** 2))) if X.nnz > 0 else 0.0
            s = np.array([norm_val], dtype=np.float64)
            sigma_max = norm_val
            rank_tol = float(sigma_max * max(n, d) * np.finfo(np.float64).eps * rank_tol_factor)
            if sigma_max > rank_tol:
                numerical_rank = 1
                sigma_min_nonzero = sigma_max
                sigma_min = sigma_max
            else:
                numerical_rank = 0
                sigma_min_nonzero = 0.0
                sigma_min = 0.0
        else:
            k_target = min_dim - 1 if k is None else min(k, min_dim - 1)
            k_target = max(1, k_target)

            try:
                s_lm = svds(X, k=k_target, which="LM", return_singular_vectors=False)
                s = np.sort(s_lm)[::-1]
            except Exception as e:
                notes_list.append(f"Sparse LM singular value solver failed: {e}")
                s = np.array([], dtype=np.float64)

            sigma_max = float(s[0]) if len(s) > 0 else 0.0
            rank_tol = float(sigma_max * max(n, d) * np.finfo(np.float64).eps * rank_tol_factor)
            nonzero_sv = s[s > rank_tol]
            sigma_min_nonzero = float(nonzero_sv[-1]) if len(nonzero_sv) > 0 else 0.0

            # Attempt to determine smallest singular value via SM
            sigma_min = None
            try:
                s_sm = svds(X, k=1, which="SM", return_singular_vectors=False)
                sm_val = float(s_sm[0])
                if sm_val <= rank_tol:
                    sigma_min = 0.0
                else:
                    sigma_min = sm_val
            except Exception:
                pass

            if len(nonzero_sv) < len(s):
                numerical_rank = int(len(nonzero_sv))
                sigma_min = 0.0
            elif sigma_min is not None and sigma_min == 0.0:
                numerical_rank = int(len(nonzero_sv))
                if numerical_rank == min_dim:
                    numerical_rank = min_dim - 1
            elif sigma_min is not None and sigma_min > 0.0 and k_target == min_dim - 1:
                numerical_rank = min_dim
            else:
                numerical_rank = int(len(nonzero_sv))
                if sigma_min is None:
                    sigma_min = sigma_min_nonzero
                    notes_list.append(
                        "Smallest singular value iterative solve did not converge; "
                        "kappa estimated from retained components."
                    )

    kappa = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")
    kappa_effective = float(sigma_max / sigma_min_nonzero) if sigma_min_nonzero > 0 else float("inf")
    digits_lost = float("inf") if np.isinf(kappa) else float(np.log10(max(1.0, kappa)))

    if kappa < cfg.cond_warning_threshold:
        band = "acceptable"
    elif kappa < cfg.cond_critical_threshold:
        band = "warn"
    else:
        band = "unacceptable"

    if numerical_rank < min_dim:
        notes_list.append(f"Matrix is rank-deficient: numerical rank {numerical_rank} < min(n, d) {min_dim}.")
    if band == "warn":
        notes_list.append(f"Condition number ({kappa:.2e}) exceeds warning threshold ({cfg.cond_warning_threshold:.2e}).")
    elif band == "unacceptable":
        notes_list.append(f"Condition number ({kappa:.2e}) exceeds critical threshold ({cfg.cond_critical_threshold:.2e}).")

    return ConditionReport(
        kappa=kappa,
        kappa_effective=kappa_effective,
        numerical_rank=numerical_rank,
        rank_tol=rank_tol,
        sigma_max=sigma_max,
        sigma_min_nonzero=sigma_min_nonzero,
        digits_lost=digits_lost,
        band=band,
        notes=tuple(notes_list)
    )

condition_report = compute_condition_report


def stable_column_variance(
    X: Union[np.ndarray, sp.spmatrix],
    ddof: int = 0
) -> np.ndarray:
    """Computes column variances stably using a two-pass algorithm without catastrophic cancellation.

    Avoids the E[x^2] - (E[x])^2 shortcut which suffers from severe subtraction cancellation.
    Note: Like all floating-point routines, precision is subject to machine epsilon (~2.22e-16).

    Args:
        X: 2D NumPy array or SciPy sparse matrix.
        ddof: Delta degrees of freedom (default: 0).

    Returns:
        1D float64 NumPy array of column variances.
    """
    if not (isinstance(X, np.ndarray) or sp.issparse(X)):
        raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X)}")

    if X.ndim != 2:
        raise InputValidation(message=f"Expected 2D matrix, got ndim={X.ndim}")

    n, d = X.shape
    if n < 1 or d < 1:
        raise InputValidation(message=f"Matrix dimensions must be positive, got {X.shape}")

    if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof < 0:
        raise InputValidation(message=f"ddof must be a non-negative integer, got {ddof}")

    if n <= ddof:
        raise InputValidation(message=f"n_samples ({n}) must be strictly greater than ddof ({ddof})")

    if sp.issparse(X):
        if X.nnz > 0 and not np.isfinite(X.data).all():
            raise NonFiniteInputError(message="Sparse data contains NaN or infinite values")
        return sparse_column_mean_var(X, ddof=ddof)[1]
    else:
        if not np.isfinite(X).all():
            raise NonFiniteInputError(message="Dense data contains NaN or infinite values")

        mean = np.mean(X, axis=0, dtype=np.float64)
        dev = X.astype(np.float64) - mean
        sq_dev_sum = np.sum(dev ** 2, axis=0, dtype=np.float64)
        divisor = n if ddof == 0 else (n - ddof)
        var = sq_dev_sum / divisor
        return np.maximum(var, 0.0).astype(np.float64)

stable_variance = stable_column_variance


def component_cosine_stability(
    W_a: np.ndarray,
    W_b: np.ndarray
) -> np.ndarray:
    """Computes absolute cosine similarity between corresponding components.

    Handles eigenvector sign flips by evaluating |<u, v>| / (||u|| ||v||).

    Args:
        W_a: 2D array of shape (k, d) where rows are components.
        W_b: 2D array of shape (k, d) where rows are components.

    Returns:
        1D float64 array of shape (k,) with cosine similarities in [0, 1].
    """
    return component_cosine_similarity(W_a, W_b)


def subspace_similarity(
    W_a: np.ndarray,
    W_b: np.ndarray
) -> float:
    """Computes subspace similarity between two sets of basis vectors.

    Supports equal and unequal ranks. Rows of W_a (k1, d) and W_b (k2, d) span the subspaces.
    Computes the average cosine of the principal angles between the two subspaces.

    Args:
        W_a: 2D array of shape (k1, d).
        W_b: 2D array of shape (k2, d).

    Returns:
        float in [0.0, 1.0] representing subspace alignment (1.0 = identical subspace, 0.0 = orthogonal).
    """
    if not isinstance(W_a, np.ndarray) or not isinstance(W_b, np.ndarray):
        raise InputValidation(message="Inputs must be numpy ndarrays")

    if W_a.ndim != 2 or W_b.ndim != 2:
        raise InputValidation(message="Inputs must be 2D arrays")

    if W_a.shape[1] != W_b.shape[1]:
        raise ShapeMismatchError(message=f"Feature dimension mismatch: {W_a.shape[1]} vs {W_b.shape[1]}")

    k1, d = W_a.shape
    k2, _ = W_b.shape

    # Extract orthonormal bases for row spaces via SVD
    _, s_a, vt_a = np.linalg.svd(W_a, full_matrices=False)
    tol_a = s_a[0] * max(k1, d) * np.finfo(np.float64).eps if len(s_a) > 0 and s_a[0] > 0 else 0.0
    Q_a = vt_a[s_a > tol_a].T  # shape (d, r1)

    _, s_b, vt_b = np.linalg.svd(W_b, full_matrices=False)
    tol_b = s_b[0] * max(k2, d) * np.finfo(np.float64).eps if len(s_b) > 0 and s_b[0] > 0 else 0.0
    Q_b = vt_b[s_b > tol_b].T  # shape (d, r2)

    r1 = Q_a.shape[1]
    r2 = Q_b.shape[1]
    r_min = min(r1, r2)

    if r_min == 0:
        return 1.0 if (r1 == 0 and r2 == 0) else 0.0

    # Singular values of Q_a^T @ Q_b are the cosines of principal angles
    M = Q_a.T @ Q_b
    s = np.linalg.svd(M, compute_uv=False)
    s_clipped = np.clip(s[:r_min], 0.0, 1.0)
    return float(np.mean(s_clipped))


def eigengap_table(
    eigenvalues: np.ndarray,
    config: Optional[StabilityConfig] = None
) -> list[dict]:
    """Computes an adjacent eigengap table with safety checks for zero or near-zero denominators.

    Evaluates gap lambda_k - lambda_{k+1} and ratio lambda_k / lambda_{k+1}.
    Flags clustered eigenvalues based on the Davis-Kahan spectral gap ratio.

    Args:
        eigenvalues: 1D array of eigenvalues sorted in descending order.
        config: Optional StabilityConfig providing spectral_gap_ratio and constant_var_tol.

    Returns:
        list of dicts with keys 'k', 'lambda_k', 'lambda_next', 'gap', 'ratio', 'has_gap', 'status'.
    """
    if not hasattr(eigenvalues, "shape") or len(eigenvalues.shape) != 1:
        raise InputValidation(message="eigenvalues must be a 1D array")

    if not np.isfinite(eigenvalues).all():
        raise NonFiniteInputError(message="eigenvalues contain NaN or infinite values")

    cfg = config if config is not None else StabilityConfig()
    tau = cfg.constant_var_tol
    spectral_gap_ratio = cfg.spectral_gap_ratio

    m = len(eigenvalues)
    if m < 2:
        return []

    rows = []
    for k in range(m - 1):
        l_k = float(eigenvalues[k])
        l_next = float(eigenvalues[k + 1])
        gap = float(l_k - l_next)

        if l_next <= tau:
            ratio = 1.0 if l_k <= tau else float("inf")
        else:
            ratio = float(l_k / l_next)

        has_gap = (ratio >= spectral_gap_ratio)
        status = "OK" if has_gap else "NO_GAP"

        rows.append({
            "k": k + 1,
            "lambda_k": l_k,
            "lambda_next": l_next,
            "gap": gap,
            "ratio": ratio,
            "has_gap": has_gap,
            "status": status,
        })

    return rows


__all__ = [
    "StabilityConfig",
    "StabilityConfigError",
    "ConditionReport",
    "ReportValidationError",
    "InputDiagnostics",
    "NonFiniteInputError",
    "SparseNotSupportedError",
    "validate_input",
    "compute_condition_report",
    "condition_report",
    "stable_column_variance",
    "stable_variance",
    "component_cosine_stability",
    "subspace_similarity",
    "eigengap_table",
    "sparse_reconstruction_mse",
]

