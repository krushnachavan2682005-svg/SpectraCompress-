import logging
from typing import Optional, Union, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from error import InputValidation, ShapeMismatchError, NotFittedError, MemoryGuardError

logger = logging.getLogger(__name__)


def sparse_column_mean_var(
    X: Union[sp.csr_matrix, sp.csc_matrix],
    ddof: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes column means and variances of a sparse matrix in two passes without densifying.

    Accounting for implicit zeros:
        Implicit zeros are observations (0.0). In Pass 1, non-zero entries are summed
        and divided by n_samples to produce the exact column mean. In Pass 2, squared
        deviations of explicit non-zeros ((x - mean)^2) are summed, and the exact squared
        deviations of implicit zeros ((0 - mean)^2 = mean^2) are added for each column.

    Numerical limitation:
        This two-pass approach evaluates deviations directly from column means, avoiding
        the severe catastrophic cancellation of the E[X^2] - (E[X])^2 shortcut. However,
        minor rounding can still occur if individual squared deviations span wide
        magnitudes, though accumulating in float64 mitigates this.

    Args:
        X: SciPy sparse matrix (CSR or CSC).
        ddof: Delta degrees of freedom. Default is 0 (population variance).

    Returns:
        (mean, var): Tuple of 1D float64 NumPy arrays, each of shape (n_features,).
    """
    if not sp.issparse(X):
        raise InputValidation(message=f"Expected scipy.sparse matrix, got {type(X)}")

    if X.ndim != 2:
        raise InputValidation(message=f"Expected 2 dimensions, got {X.ndim}")

    n, d = X.shape
    if n < 1 or d < 1:
        raise InputValidation(message=f"Matrix dimensions must be positive, got {X.shape}")

    if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof < 0:
        raise InputValidation(message=f"ddof must be a non-negative integer, got {ddof}")

    if n <= ddof:
        raise InputValidation(message=f"n_samples ({n}) must be strictly greater than ddof ({ddof})")

    # Reject non-finite values in stored non-zero elements
    if X.nnz > 0 and not np.isfinite(X.data).all():
        raise InputValidation(message="Sparse data contains NaN or infinite values")

    # Check format and safely convert only if not CSR/CSC
    if sp.isspmatrix_csr(X):
        work_X = X
        nz_per_col = np.bincount(work_X.indices, minlength=d)
        col_indices = work_X.indices
    elif sp.isspmatrix_csc(X):
        work_X = X
        nz_per_col = np.diff(work_X.indptr)
        col_indices = np.repeat(np.arange(d), nz_per_col)
    else:
        logger.info("Converting sparse matrix to CSR format for sparse_column_mean_var computation.")
        work_X = X.tocsr()
        nz_per_col = np.bincount(work_X.indices, minlength=d)
        col_indices = work_X.indices

    # Pass 1: Column mean
    col_sum = np.asarray(work_X.sum(axis=0)).ravel()
    mean = (col_sum / n).astype(np.float64)

    # Pass 2: Sum of squared deviations
    if work_X.nnz > 0:
        dev = work_X.data.astype(np.float64) - mean[col_indices]
        explicit_ss = np.bincount(col_indices, weights=dev ** 2, minlength=d)
    else:
        explicit_ss = np.zeros(d, dtype=np.float64)

    implicit_ss = (n - nz_per_col) * (mean ** 2)
    divisor = n if ddof == 0 else (n - ddof)
    var = ((explicit_ss + implicit_ss) / divisor).astype(np.float64)

    return mean, var


def sparse_center(
    X: Union[sp.csr_matrix, sp.csc_matrix],
    mean: np.ndarray,
    scale: Optional[np.ndarray] = None
) -> LinearOperator:
    """Returns a SciPy LinearOperator representing the centered (and optionally scaled) sparse matrix.

    Mathematical definition:
        X_centered = (X - ones(n_samples, 1) @ mean.reshape(1, -1)) [ / scale ]

    Memory avoidance:
        Explicit mean subtraction (X - mean) destroys sparsity and converts an (n, d)
        sparse matrix into a dense matrix requiring O(n * d) memory. The LinearOperator
        interface instead computes matrix-vector (matvec) and matrix-matrix (matmat)
        products dynamically without ever allocating or materializing the full (n, d)
        centered matrix in memory.

    Args:
        X: SciPy sparse matrix (CSR or CSC).
        mean: 1D NumPy array of feature means of shape (n_features,).
        scale: Optional 1D NumPy array of feature standard deviations of shape (n_features,).

    Returns:
        LinearOperator of shape (n_samples, n_features) and dtype float64.
    """
    if not sp.issparse(X):
        raise InputValidation(message=f"Expected scipy.sparse matrix, got {type(X)}")

    if X.ndim != 2:
        raise InputValidation(message=f"Expected 2 dimensions, got {X.ndim}")

    n, d = X.shape
    mean_arr = np.asarray(mean, dtype=np.float64)
    if mean_arr.ndim != 1 or mean_arr.shape[0] != d:
        raise ShapeMismatchError(message=f"mean dimension {mean_arr.shape} does not match n_features {d}")

    if not np.isfinite(mean_arr).all():
        raise InputValidation(message="mean contains NaN or infinite values")

    if X.nnz > 0 and not np.isfinite(X.data).all():
        raise InputValidation(message="Sparse matrix contains NaN or infinite values")

    scale_arr = None
    if scale is not None:
        scale_arr = np.asarray(scale, dtype=np.float64)
        if scale_arr.ndim != 1 or scale_arr.shape[0] != d:
            raise ShapeMismatchError(message=f"scale dimension {scale_arr.shape} does not match n_features {d}")
        if not np.isfinite(scale_arr).all() or np.any(scale_arr <= 0):
            raise InputValidation(message="scale must contain strictly positive finite values")

    def _matvec(v: np.ndarray) -> np.ndarray:
        v_arr = np.asarray(v, dtype=np.float64)
        if v_arr.shape != (d,):
            raise ShapeMismatchError(message=f"Dimension mismatch for matvec: expected ({d},), got {v_arr.shape}")
        if scale_arr is not None:
            v_arr = v_arr / scale_arr
        term1 = X @ v_arr
        proj = np.dot(mean_arr, v_arr)
        return (term1 - proj).astype(np.float64)

    def _rmatvec(u: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.shape != (n,):
            raise ShapeMismatchError(message=f"Dimension mismatch for rmatvec: expected ({n},), got {u_arr.shape}")
        term1 = X.T @ u_arr
        sum_u = np.sum(u_arr)
        res = term1 - mean_arr * sum_u
        if scale_arr is not None:
            res = res / scale_arr
        return res.astype(np.float64)

    def _matmat(V: np.ndarray) -> np.ndarray:
        V_arr = np.asarray(V, dtype=np.float64)
        if V_arr.ndim != 2 or V_arr.shape[0] != d:
            raise ShapeMismatchError(message=f"Dimension mismatch for matmat: expected ({d}, k), got {V_arr.shape}")
        if scale_arr is not None:
            V_arr = V_arr / scale_arr[:, None]
        term1 = X @ V_arr
        mean_proj = mean_arr @ V_arr
        return (term1 - mean_proj).astype(np.float64)

    return LinearOperator(
        shape=(n, d),
        matvec=_matvec,
        rmatvec=_rmatvec,
        matmat=_matmat,
        dtype=np.float64
    )


def to_dense_centered(
    X: Union[np.ndarray, sp.spmatrix],
    mean: np.ndarray,
    max_bytes: int
) -> np.ndarray:
    """Returns a dense centered matrix only when within the configured memory budget.

    Memory Guard:
        The memory requirement is estimated as n_samples * n_features * 8 bytes (float64).
        If required_bytes > max_bytes, a MemoryGuardError is raised BEFORE any dense array
        allocation is attempted. Note that actual process memory overhead can exceed this
        estimate due to Python object overhead and temporary copies.

    Args:
        X: Dense NumPy array or SciPy sparse matrix of shape (n_samples, n_features).
        mean: 1D NumPy array of feature means of shape (n_features,).
        max_bytes: Maximum memory threshold in bytes (non-negative integer).

    Returns:
        Dense float64 NumPy ndarray of shape (n_samples, n_features).
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        raise InputValidation(message=f"max_bytes must be a non-negative integer, got {max_bytes}")

    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise InputValidation(message="X must be a 2D matrix or array")

    n, d = X.shape
    if n < 1 or d < 1:
        raise InputValidation(message=f"Matrix dimensions must be positive, got {X.shape}")

    mean_arr = np.asarray(mean, dtype=np.float64)
    if mean_arr.ndim != 1 or mean_arr.shape[0] != d:
        raise ShapeMismatchError(message=f"mean dimension {mean_arr.shape} does not match n_features {d}")

    itemsize = np.dtype(np.float64).itemsize
    required_bytes = n * d * itemsize

    if required_bytes > max_bytes:
        raise MemoryGuardError(
            message=f"Required memory {required_bytes} bytes exceeds maximum allowed budget of {max_bytes} bytes."
        )

    if sp.issparse(X):
        dense_X = X.toarray()
        return (dense_X - mean_arr).astype(np.float64)
    elif isinstance(X, np.ndarray):
        return (X - mean_arr).astype(np.float64)
    else:
        raise InputValidation(message=f"Unsupported matrix type: {type(X)}")


class Centerer:
    """Centers data by subtracting the feature means, with optional variance scaling.

    Supports both dense NumPy arrays and SciPy sparse matrices.
    - For dense inputs, transform() returns an (n, d) float64 NumPy ndarray.
    - For sparse inputs, transform() returns a SciPy LinearOperator to prevent densification.

    Configuration Ownership Note:
        The authoritative configuration owner for system-level stability parameters is
        `StabilityConfig` (in `stability_diagnostics.py`). `Centerer` exposes
        `constant_var_tol` as an explicit dependency injection parameter, defaulting
        to 1e-12 as an intentional decoupled component-level default to prevent circular imports.
    """
    def __init__(
        self,
        with_std: bool = False,
        constant_var_tol: Optional[float] = 1e-12,
        with_scaling: Optional[bool] = None
    ):
        if with_scaling is not None:
            self.with_std = bool(with_scaling)
        else:
            self.with_std = bool(with_std)

        if constant_var_tol is None:
            self.constant_var_tol = 1e-12
        else:
            if not np.isfinite(constant_var_tol) or constant_var_tol < 0:
                raise InputValidation(
                    message=f"constant_var_tol must be non-negative and finite, got {constant_var_tol}"
                )
            self.constant_var_tol = float(constant_var_tol)

        self.mean_ = None
        self.feature_mean_ = None
        self.var_ = None
        self.scale_ = None
        self.constant_features_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    def _validate(self, X: Union[np.ndarray, sp.spmatrix], *, fitting: bool) -> None:
        if not (isinstance(X, np.ndarray) or sp.issparse(X)):
            raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X)}")

        if X.dtype not in (np.float32, np.float64):
            raise InputValidation(message=f"Dtype must be float32 or float64, got {X.dtype}")

        if X.ndim != 2:
            raise InputValidation(message=f"Expected 2 dimensions, got {X.ndim}")

        n, d = X.shape
        if fitting and n < 2:
            raise InputValidation(message=f"fit requires n >= 2, got {n}")
        if not fitting and n < 1:
            raise InputValidation(message=f"transform requires n >= 1, got {n}")

        if isinstance(X, np.ndarray):
            if not np.isfinite(X).all():
                raise InputValidation(message="Data contains NaN or infinite values")
        else:
            if X.nnz > 0 and not np.isfinite(X.data).all():
                raise InputValidation(message="Data contains NaN or infinite values")

        if not fitting:
            if d != self.n_features_in_:
                raise ShapeMismatchError(message=f"expected d={self.n_features_in_}, got d={d}")

    def fit(self, X: Union[np.ndarray, sp.spmatrix], y=None) -> "Centerer":
        self._validate(X, fitting=True)
        n, d = X.shape

        if isinstance(X, np.ndarray):
            mean = np.mean(X, axis=0, dtype=np.float64)
            if self.with_std:
                var = np.var(X, axis=0, ddof=1, dtype=np.float64)
                is_constant = var <= self.constant_var_tol
                if np.any(is_constant):
                    logger.warning(
                        f"Detected {np.sum(is_constant)} constant features with variance <= {self.constant_var_tol}."
                    )
                scale = np.sqrt(var)
                scale[is_constant] = 1.0
                self.var_ = var
                self.scale_ = scale
                self.constant_features_ = is_constant
        else:
            mean, var = sparse_column_mean_var(X, ddof=1)
            if self.with_std:
                is_constant = var <= self.constant_var_tol
                if np.any(is_constant):
                    logger.warning(
                        f"Detected {np.sum(is_constant)} constant features with variance <= {self.constant_var_tol}."
                    )
                scale = np.sqrt(var)
                scale[is_constant] = 1.0
                self.var_ = var
                self.scale_ = scale
                self.constant_features_ = is_constant

        self.mean_ = mean
        self.feature_mean_ = mean
        self.n_features_in_ = d
        self.n_samples_seen_ = n
        return self

    def transform(self, X: Union[np.ndarray, sp.spmatrix]) -> Union[np.ndarray, LinearOperator]:
        if not self.is_fitted:
            raise NotFittedError(message="The X is not fitted")

        self._validate(X, fitting=False)

        if isinstance(X, np.ndarray):
            Xc = (X - self.mean_).astype(np.float64)
            if self.with_std:
                Xc = Xc / self.scale_
            return Xc
        else:
            scale = self.scale_ if self.with_std else None
            return sparse_center(X, self.mean_, scale=scale)

    def inverse_transform(self, Xc: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise NotFittedError(message="The X is not fitted")

        if isinstance(Xc, LinearOperator):
            raise InputValidation(message="inverse_transform requires dense ndarray input, not LinearOperator.")

        self._validate(Xc, fitting=False)
        out = Xc.copy().astype(np.float64)
        if self.with_std:
            out = out * self.scale_
        return (out + self.mean_).astype(np.float64)