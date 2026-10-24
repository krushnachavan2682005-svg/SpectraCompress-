import time
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Union
import joblib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD

from error import (
    InputValidation,
    ShapeMismatchError,
    NotFittedError,
    NonConvergenceError,
    MemoryGuardError,
)
from feature_pipeline import Centerer, sparse_center, sparse_column_mean_var
from stability_diagnostics import (
    StabilityConfig,
    ConditionReport,
    InputDiagnostics,
    SparseNotSupportedError,
    NonFiniteInputError,
)
from linalg_core import top_k_eigenpairs


# From-scratch method madhye top-k eigenpairs calculate karnyasathi helper.
# Jar power iteration converge zali nahi, tar specific convergence error raise karto.
def _run_power_iteration(A, k, tol, max_iter, seed):
    try:
        return top_k_eigenpairs(A=A, k=k, tol=tol, max_iter=max_iter, seed=seed)
    except InputValidation as e:
        msg = getattr(e, "message", "")
        if "did not converge" in msg:
            raise NonConvergenceError(message=e.message) from e
        if "Cannot normalize zero vector" in msg:
            # Fallback for rank-deficient matrices where remaining eigenvalues are zero
            from linalg_core import power_iteration, deflate
            eigenpairs = []
            current_A = A.copy()
            for _ in range(k):
                try:
                    ep = power_iteration(A=current_A, tol=tol, max_iter=max_iter, seed=seed)
                except InputValidation as sub_e:
                    if "Cannot normalize zero vector" in getattr(sub_e, "message", ""):
                        break
                    raise sub_e
                eigenpairs.append(ep)
                current_A = deflate(A=current_A, eigenvalue=ep["eigenvalue"], eigenvector=ep["eigenvector"])
            return sorted(eigenpairs, key=lambda x: abs(x["eigenvalue"]), reverse=True)
        raise e



# Logging mule warning ani important events console/log madhye record karta yetat.
logger = logging.getLogger(__name__)


# PCA/decomposition sathi konti implementation use karaychi te define karto.
class MethodType(Enum):
    from_scratch_covariance = "from_scratch_covariance"  # Aapan manually covariance + eigenpairs vaparto.
    sklearn_pca = "sklearn_pca"                          # sklearn chi PCA implementation.
    sklearn_truncated_svd = "sklearn_truncated_svd"      # sklearn chi TruncatedSVD implementation.
    scipy_implicit_centered = "scipy_implicit_centered"  # SciPy svds on implicit centered LinearOperator.


# Decomposition algorithm la laganare settings ekatra store karto.
# frozen=True mule config object create zalyavar tyachi fields change karta yet nahit.
@dataclass(frozen=True)
class DecompositionConfig:
    method: MethodType                 # Konti decomposition method use karaychi.
    k: int = 10                        # Kiti principal components retain karayche.
    seed: Optional[int] = 42            # Randomized steps reproducible thevnyasathi seed.
    eig_tol: float = 1e-10             # Eigenvalue/eigenvector iteration chi tolerance.
    eig_max_iter: int = 1000           # Power iteration sathi maximum iterations.
    psd_clip_rtol: float = 1e-8        # Tiny negative eigenvalues clip karnyasathi relative tolerance.
    pca_svd_solver: str = "full"       # sklearn PCA madhye konta SVD solver vapraycha.
    tsvd_algorithm: str = "arpack"     # TruncatedSVD madhye konta algorithm vapraycha.
    tsvd_n_iter: int = 5               # TruncatedSVD randomized method sathi iteration setting.
    output_dtype: str = "float64"      # Output arrays cha expected numeric data type.
    max_dense_cov_dim: int = 5000      # Feature limit for dense covariance matrix formation.

    # Config values valid aahet ka te check karto.
    def __post_init__(self):
        # Method string madhye dili asel tar Enum madhye convert karto.
        if isinstance(self.method, str):
            try:
                object.__setattr__(self, 'method', MethodType(self.method))
            except ValueError:
                raise InputValidation(message=f"Unknown method {self.method}")
        elif not isinstance(self.method, MethodType):
            raise InputValidation(message="Invalid method type")

        # K kamit kami 1 asla pahije.
        if self.k < 1:
            raise InputValidation(message="k must be >= 1")

        # Eigenvalue iteration tolerance positive asavi.
        if self.eig_tol <= 0:
            raise InputValidation(message="eig_tol must be > 0")

        # Iterations chi sankhya kamit kami 1 asavi.
        if self.eig_max_iter < 1:
            raise InputValidation(message="eig_max_iter must be >= 1")

        # Relative clipping tolerance negative nasavi.
        if self.psd_clip_rtol < 0:
            raise InputValidation(message="psd_clip_rtol must be >= 0")

        # Ya implementation madhye fakt float64 output allow aahe.
        if self.output_dtype not in {"float64"}:
            raise InputValidation(message="output_dtype must be 'float64'")

        if self.max_dense_cov_dim < 1:
            raise InputValidation(message="max_dense_cov_dim must be >= 1")


# Covariance matrix PSD aslyamule theoretically eigenvalues negative nasayla havet.
# Floating-point calculation mule alele tiny negative eigenvalues 0 karto.
# Meaningfully negative eigenvalue asel tar validation error raise karto.
def clip_psd_eigenvalues(eigvals: np.ndarray, rtol: float) -> np.ndarray:
    """Clips tiny negative eigenvalues to 0. Values == -tol are clipped."""

    # Eigenvalues array empty asel tar pudhchi calculation hou shakat nahi.
    if eigvals.size == 0:
        raise InputValidation(message="Empty input")

    # Sarvat mothya absolute eigenvalue varun clipping tolerance calculate karto.
    max_val = np.max(np.abs(eigvals))
    tol = rtol * max_val

    # Tolerance peksha jast negative eigenvalue asel tar te silently fix karat nahi.
    if np.any(eigvals < -tol):
        raise InputValidation(message="Eigenvalue too negative for PSD clipping")

    # [-tol, 0) range madhil tiny negative values identify karto.
    mask = (eigvals >= -tol) & (eigvals < 0)
    count = np.sum(mask)

    # Kiti values clip kelya te warning madhye sangto.
    if count > 0:
        logger.warning(f"Warning: {count} negative eigenvalues were clipped to zero.")

    # Tiny negative values 0 karto ani output float64 madhye return karto.
    clipped = np.where(mask, 0.0, eigvals)
    return clipped.astype(np.float64)


class DecompositionTransformer:
    # Transformer create kartana config store karto ani fitted attributes initialize karto.
    def __init__(
        self,
        config: DecompositionConfig,
        stability_config: Optional[StabilityConfig] = None
    ):
        self.config_ = config
        self.stability_config_ = stability_config if stability_config is not None else StabilityConfig()
        self.centerer = None
        self.components_ = None
        self.explained_variance_ = None
        self.singular_values_ = None
        self.total_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_features_in_ = None
        self.fit_time_s_ = None

        # Day 4 Diagnostics & Sparse State
        self.effective_rank_ = None
        self.requested_rank_ = config.k
        self.centered_ = True
        self.fit_warnings_ = []
        self.condition_report_ = None
        self.input_diagnostics_ = None

    # Fitted Centerer madhun training mean access karnyasathi property.
    @property
    def mean_(self):
        if self.centerer is None:
            return None
        return self.centerer.mean_

    @property
    def feature_mean_(self):
        return self.mean_

    @property
    def n_components(self):
        return self.effective_rank_ if self.effective_rank_ is not None else self.config_.k

    def _fit_implicit_centered(
        self,
        X_train: Union[np.ndarray, sp.spmatrix],
        means: np.ndarray,
        vars_: np.ndarray,
        n: int,
        d: int,
        k: int
    ) -> None:
        """Internal helper for scipy_implicit_centered solver using LinearOperator."""
        if k >= min(n, d):
            raise InputValidation(
                message=f"scipy_implicit_centered requires k < min(n_samples, n_features). Got k={k}, min={min(n, d)}."
            )

        self.total_variance_ = float(np.sum(vars_))
        op = sparse_center(X_train, means)

        try:
            u, s, vt = svds(op, k=k, random_state=self.config_.seed)
        except Exception as e:
            if "did not converge" in str(e).lower() or "arpacknoconvergence" in type(e).__name__.lower():
                raise NonConvergenceError(message="Iterative SVD did not converge") from e
            raise e

        # svds returns singular values in ascending order; reverse to descending
        s = s[::-1]
        vt = vt[::-1, :]

        # Deterministic sign convention
        max_abs_cols = np.argmax(np.abs(vt), axis=1)
        signs = np.sign(vt[range(vt.shape[0]), max_abs_cols])
        signs[signs == 0] = 1
        vt *= signs[:, np.newaxis]

        self.components_ = vt
        self.singular_values_ = s

    # Training data var decomposition shikto.
    def fit(self, X_train: Union[np.ndarray, sp.spmatrix]) -> "DecompositionTransformer":
        start_time = time.perf_counter()
        self.fit_warnings_ = []
        self.requested_rank_ = self.config_.k
        k = self.config_.k

        # Validate input type
        if not (isinstance(X_train, np.ndarray) or sp.issparse(X_train)):
            raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X_train)}")

        if X_train.ndim != 2:
            raise InputValidation(message=f"Expected 2 dimensions, got {X_train.ndim}")

        n, d = X_train.shape
        self.n_features_in_ = d

        # Check k constraints
        if not (1 <= k <= min(n - 1, d)):
            raise InputValidation(message=f"k must be between 1 and min(n-1, d). Got k={k}")

        is_sparse = sp.issparse(X_train)

        # Validate non-finite entries
        if is_sparse:
            if X_train.nnz > 0 and not np.isfinite(X_train.data).all():
                raise NonFiniteInputError(message="Sparse data contains NaN or infinite values")
        else:
            if not np.isfinite(X_train).all():
                raise NonFiniteInputError(message="Data contains NaN or infinite values")

        # ---------------------------------------------------------
        # Compute Input Diagnostics (Training set only)
        # ---------------------------------------------------------
        fmt = "dense" if not is_sparse else ("csr" if sp.isspmatrix_csr(X_train) else ("csc" if sp.isspmatrix_csc(X_train) else "sparse"))
        density = float(X_train.nnz / (n * d)) if is_sparse else float(np.count_nonzero(X_train) / (n * d))
        constant_var_tol = self.stability_config_.constant_var_tol

        if is_sparse:
            means, vars_ = sparse_column_mean_var(X_train, ddof=1)
        else:
            means = np.mean(X_train, axis=0, dtype=np.float64)
            vars_ = np.var(X_train, axis=0, ddof=1, dtype=np.float64)

        constant_idx = tuple(int(i) for i in np.where(vars_ <= constant_var_tol)[0])
        if len(constant_idx) > 0:
            warn_msg = f"Detected {len(constant_idx)} constant features with variance <= {constant_var_tol}."
            logger.warning(warn_msg)
            self.fit_warnings_.append(warn_msg)

        self.input_diagnostics_ = InputDiagnostics(
            shape=(n, d),
            is_sparse=is_sparse,
            format=fmt,
            density=density,
            n_nonfinite=0,
            constant_feature_idx=constant_idx,
            duplicate_row_frac=0.0,
            warnings=tuple(self.fit_warnings_)
        )

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # Method Dispatch & Policy Enforcement
        # ---------------------------------------------------------
        if is_sparse and self.config_.method == MethodType.sklearn_pca:
            raise SparseNotSupportedError(
                "Method sklearn_pca requires dense input in the current implementation. "
                "Use from_scratch_covariance or scipy_implicit_centered for sparse data."
            )

        # ---------------------------------------------------------
        # Explicit Policy for Constant-Only / Zero-Rank Data
        # ---------------------------------------------------------
        if len(constant_idx) == d or np.all(vars_ <= constant_var_tol):
            self.centered_ = True
            if is_sparse:
                self.centerer = Centerer(constant_var_tol=constant_var_tol)
                self.centerer.fit(X_train)
            else:
                self.centerer = Centerer()
                self.centerer.fit(X_train)

            self.effective_rank_ = 0
            self.components_ = np.empty((0, d), dtype=np.float64)
            self.singular_values_ = np.empty(0, dtype=np.float64)
            self.explained_variance_ = np.empty(0, dtype=np.float64)
            self.explained_variance_ratio_ = np.empty(0, dtype=np.float64)
            self.total_variance_ = 0.0

            warn_msg = f"Numerical rank (0) is below requested rank ({k}). Retaining 0 components."
            logger.warning(warn_msg)
            warnings.warn(warn_msg, UserWarning)
            self.fit_warnings_.append(warn_msg)

            self.condition_report_ = ConditionReport(
                kappa=float("inf"),
                kappa_effective=float("inf"),
                numerical_rank=0,
                rank_tol=0.0,
                sigma_max=0.0,
                sigma_min_nonzero=0.0,
                digits_lost=float("inf"),
                band="unacceptable",
                notes=tuple(self.fit_warnings_)
            )
            self.fit_time_s_ = time.perf_counter() - start_time
            return self

        if is_sparse and self.config_.method == MethodType.sklearn_truncated_svd:
            self.centered_ = False
            warn_msg = "Method sklearn_truncated_svd on sparse input performs uncentered TruncatedSVD and is not equivalent to centered PCA."
            logger.warning(warn_msg)
            warnings.warn(warn_msg, UserWarning)
            self.fit_warnings_.append(warn_msg)

            if self.config_.tsvd_algorithm == "arpack" and k >= min(n, d):
                raise InputValidation(message="arpack cannot handle k >= min(n, d)")

            tsvd = TruncatedSVD(
                n_components=k,
                algorithm=self.config_.tsvd_algorithm,
                n_iter=self.config_.tsvd_n_iter,
                random_state=self.config_.seed
            )
            tsvd.fit(X_train)
            self.components_ = tsvd.components_
            self.singular_values_ = tsvd.singular_values_
            self.centerer = None
            self.total_variance_ = float(X_train.power(2).sum() / (n - 1))

        elif is_sparse and self.config_.method == MethodType.from_scratch_covariance:
            self.centered_ = True
            self.centerer = Centerer(constant_var_tol=constant_var_tol)
            self.centerer.fit(X_train)
            means = self.centerer.mean_
            max_dense_cov_dim = getattr(self.config_, "max_dense_cov_dim", 5000)

            # Catastrophic cancellation indicator: max_j (mean_j^2 / var_j)
            nonzero_var = vars_ > constant_var_tol
            ratios = np.zeros_like(means, dtype=np.float64)
            ratios[nonzero_var] = (means[nonzero_var] ** 2) / vars_[nonzero_var]
            zero_var_nonzero_mean = (~nonzero_var) & (np.abs(means) > constant_var_tol)
            ratios[zero_var_nonzero_mean] = float("inf")
            cancellation_indicator = float(np.max(ratios)) if len(ratios) > 0 else 0.0

            thresh = self.stability_config_.cancellation_ratio_threshold
            eff_thresh = (1.0 / thresh) if (0.0 < thresh < 1.0) else thresh

            if d > max_dense_cov_dim:
                warn_msg = f"Feature dimension d={d} exceeds max_dense_cov_dim={max_dense_cov_dim}. Using scipy_implicit_centered path."
                logger.warning(warn_msg)
                warnings.warn(warn_msg, UserWarning)
                self.fit_warnings_.append(warn_msg)
                self._fit_implicit_centered(X_train, means, vars_, n, d, k)
            elif cancellation_indicator > eff_thresh:
                warn_msg = (
                    f"Catastrophic cancellation detected (cancellation indicator {cancellation_indicator:.2e} "
                    f"> threshold {eff_thresh:.2e}). Using scipy_implicit_centered path."
                )
                logger.warning(warn_msg)
                warnings.warn(warn_msg, UserWarning)
                self.fit_warnings_.append(warn_msg)
                self._fit_implicit_centered(X_train, means, vars_, n, d, k)
            else:
                # Sparse Gram matrix multiplication -> d x d dense
                G = (X_train.T @ X_train).toarray().astype(np.float64)
                # Mean correction: Xc.T @ Xc = X.T @ X - n * (mean @ mean.T)
                G_centered = G - n * np.outer(means, means)
                C = (G_centered + G_centered.T) / 2.0 / (n - 1)
                self.total_variance_ = float(np.trace(C))

                eigenpairs = _run_power_iteration(
                    A=C,
                    k=k,
                    tol=self.config_.eig_tol,
                    max_iter=self.config_.eig_max_iter,
                    seed=self.config_.seed
                )
                evals = np.array([ep["eigenvalue"] for ep in eigenpairs])
                evecs = np.array([ep["eigenvector"] for ep in eigenpairs]).T
                idx = np.argsort(evals)[::-1]
                evals = evals[idx]
                evecs = evecs[:, idx]
                evals = clip_psd_eigenvalues(evals, self.config_.psd_clip_rtol)
                self.components_ = evecs.T

                max_abs_cols = np.argmax(np.abs(self.components_), axis=1)
                signs = np.sign(self.components_[range(self.components_.shape[0]), max_abs_cols])
                signs[signs == 0] = 1
                self.components_ *= signs[:, np.newaxis]
                self.singular_values_ = np.sqrt(evals * (n - 1))

        elif is_sparse and self.config_.method == MethodType.scipy_implicit_centered:
            self.centered_ = True
            self.centerer = Centerer(constant_var_tol=constant_var_tol)
            self.centerer.fit(X_train)
            self._fit_implicit_centered(X_train, self.centerer.mean_, vars_, n, d, k)

        elif not is_sparse and self.config_.method == MethodType.scipy_implicit_centered:
            self.centered_ = True
            self.centerer = Centerer(constant_var_tol=constant_var_tol)
            self.centerer.fit(X_train)
            csr_X = sp.csr_matrix(X_train)
            self._fit_implicit_centered(csr_X, self.centerer.mean_, vars_, n, d, k)

        else:
            # Dense regression paths (sklearn_pca, sklearn_truncated_svd, from_scratch_covariance)
            self.centered_ = True
            self.centerer = Centerer()
            self.centerer.fit(X_train)
            Xc = self.centerer.transform(X_train)
            self.total_variance_ = float(np.sum(Xc ** 2) / (n - 1))

            if self.config_.method == MethodType.from_scratch_covariance:
                C = (Xc.T @ Xc) / (n - 1)
                C = (C + C.T) / 2.0
                eigenpairs = _run_power_iteration(
                    A=C,
                    k=k,
                    tol=self.config_.eig_tol,
                    max_iter=self.config_.eig_max_iter,
                    seed=self.config_.seed
                )
                evals = np.array([ep["eigenvalue"] for ep in eigenpairs])
                evecs = np.array([ep["eigenvector"] for ep in eigenpairs]).T
                idx = np.argsort(evals)[::-1]
                evals = evals[idx]
                evecs = evecs[:, idx]
                evals = clip_psd_eigenvalues(evals, self.config_.psd_clip_rtol)
                self.components_ = evecs.T

                max_abs_cols = np.argmax(np.abs(self.components_), axis=1)
                signs = np.sign(self.components_[range(self.components_.shape[0]), max_abs_cols])
                signs[signs == 0] = 1
                self.components_ *= signs[:, np.newaxis]
                self.singular_values_ = np.sqrt(evals * (n - 1))

            elif self.config_.method == MethodType.sklearn_pca:
                pca = PCA(n_components=k, svd_solver=self.config_.pca_svd_solver, random_state=self.config_.seed)
                pca.fit(Xc)
                self.components_ = pca.components_
                self.singular_values_ = pca.singular_values_

            elif self.config_.method == MethodType.sklearn_truncated_svd:
                if self.config_.tsvd_algorithm == "arpack" and k >= min(n, d):
                    raise InputValidation(message="arpack cannot handle k >= min(n, d)")
                tsvd = TruncatedSVD(
                    n_components=k,
                    algorithm=self.config_.tsvd_algorithm,
                    n_iter=self.config_.tsvd_n_iter,
                    random_state=self.config_.seed
                )
                tsvd.fit(Xc)
                self.components_ = tsvd.components_
                self.singular_values_ = tsvd.singular_values_

            else:
                raise InputValidation(message=f"Unknown method {self.config_.method}")

        # ---------------------------------------------------------
        # Explained Variance & Ratios
        # ---------------------------------------------------------
        self.explained_variance_ = (self.singular_values_ ** 2) / (n - 1)
        if self.total_variance_ > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / self.total_variance_
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)

        # ---------------------------------------------------------
        # Numerical Rank & Rank Fallback Policy
        # ---------------------------------------------------------
        rank_tol_factor = self.stability_config_.rank_tol_factor if self.stability_config_.rank_tol_factor is not None else 1.0
        sigma_max = float(self.singular_values_[0]) if len(self.singular_values_) > 0 else 0.0
        rank_tol = float(sigma_max * max(n, d) * np.finfo(np.float64).eps * rank_tol_factor)

        nonzero_sv_mask = self.singular_values_ > rank_tol
        r_num = int(np.sum(nonzero_sv_mask))
        self.effective_rank_ = min(k, r_num)

        if self.effective_rank_ < k:
            warn_msg = f"Numerical rank ({self.effective_rank_}) is below requested rank ({k}). Retaining {self.effective_rank_} components."
            logger.warning(warn_msg)
            warnings.warn(warn_msg, UserWarning)
            self.fit_warnings_.append(warn_msg)

            # Truncate components, preserving orthonormality without zero-padding
            self.components_ = self.components_[:self.effective_rank_, :]
            self.singular_values_ = self.singular_values_[:self.effective_rank_]
            self.explained_variance_ = self.explained_variance_[:self.effective_rank_]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.effective_rank_]

        # ---------------------------------------------------------
        # Condition Report Diagnostics
        # ---------------------------------------------------------
        if self.effective_rank_ > 0:
            sigma_min_nonzero = float(self.singular_values_[-1])
            sigma_min = sigma_min_nonzero if self.effective_rank_ == min(n - 1, d) else (
                float(self.singular_values_[-1]) if self.effective_rank_ == k else 0.0
            )
            kappa = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")
            kappa_effective = float(sigma_max / sigma_min_nonzero) if sigma_min_nonzero > 0 else float("inf")
            digits_lost = float(np.log10(kappa)) if np.isfinite(kappa) else float("inf")
        else:
            sigma_min_nonzero = 0.0
            sigma_min = 0.0
            kappa = float("inf")
            kappa_effective = float("inf")
            digits_lost = float("inf")

        if kappa < self.stability_config_.cond_warning_threshold:
            band = "acceptable"
        elif kappa < self.stability_config_.cond_critical_threshold:
            band = "warn"
        else:
            band = "unacceptable"

        self.condition_report_ = ConditionReport(
            kappa=kappa,
            kappa_effective=kappa_effective,
            numerical_rank=r_num,
            rank_tol=rank_tol,
            sigma_max=sigma_max,
            sigma_min_nonzero=sigma_min_nonzero,
            digits_lost=digits_lost,
            band=band,
            notes=tuple(self.fit_warnings_)
        )

        # Enforce float64 output
        dtype = np.float64
        self.components_ = self.components_.astype(dtype)
        self.explained_variance_ = self.explained_variance_.astype(dtype)
        self.singular_values_ = self.singular_values_.astype(dtype)
        self.explained_variance_ratio_ = self.explained_variance_ratio_.astype(dtype)

        self.fit_time_s_ = time.perf_counter() - start_time
        return self

    # Input data la learned principal components madhye project karto.
    def transform(self, X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
        if self.components_ is None:
            raise NotFittedError(message="Transformer is not fitted.")

        if not (isinstance(X, np.ndarray) or sp.issparse(X)):
            raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X)}")

        if X.ndim != 2:
            raise InputValidation(message=f"Expected 2 dimensions, got {X.ndim}")

        if X.shape[1] != self.n_features_in_:
            raise ShapeMismatchError(message=f"expected d={self.n_features_in_}, got d={X.shape[1]}")

        if sp.issparse(X):
            if X.nnz > 0 and not np.isfinite(X.data).all():
                raise NonFiniteInputError(message="Input contains NaN or infinite values")
        else:
            if not np.isfinite(X).all():
                raise NonFiniteInputError(message="Input contains NaN or infinite values")

        n_samples = X.shape[0]
        if self.effective_rank_ == 0:
            return np.empty((n_samples, 0), dtype=np.float64)

        if sp.issparse(X):
            # Sparse projection without densifying X: Z = X @ components_.T - mean_correction
            Z = X @ self.components_.T
            if isinstance(Z, np.matrix):
                Z = np.asarray(Z)
            if self.centered_ and self.mean_ is not None:
                mean_corr = self.components_ @ self.mean_
                Z = Z - mean_corr
            return np.asarray(Z, dtype=np.float64)
        else:
            # Dense projection
            if self.centered_ and self.centerer is not None and self.centerer.is_fitted:
                Xc = self.centerer.transform(X)
            else:
                Xc = X
            Z = Xc @ self.components_.T
            return np.asarray(Z, dtype=np.float64)

    # Reduced representation Z madhun original feature space madhye reconstruction karto.
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise NotFittedError(message="Transformer is not fitted.")

        if not isinstance(Z, np.ndarray):
            raise InputValidation(message=f"Expected numpy ndarray, got {type(Z)}")

        if Z.ndim != 2:
            raise InputValidation(message="Z must be 2D")

        if Z.shape[1] != self.components_.shape[0]:
            raise ShapeMismatchError(message=f"Z shape {Z.shape} does not match k={self.components_.shape[0]}")

        n_samples = Z.shape[0]
        n_features = self.n_features_in_

        # Memory guard before dense allocation
        required_bytes = n_samples * n_features * 8
        max_bytes = getattr(self.stability_config_, "max_dense_bytes", 100_000_000)
        if required_bytes > max_bytes:
            raise MemoryGuardError(
                message=f"Required memory {required_bytes} bytes exceeds maximum allowed budget of {max_bytes} bytes."
            )

        if self.effective_rank_ == 0:
            if self.centered_ and self.mean_ is not None:
                return np.tile(self.mean_, (n_samples, 1)).astype(np.float64)
            return np.zeros((n_samples, n_features), dtype=np.float64)

        Xc_reconstructed = Z @ self.components_
        if self.centered_ and self.mean_ is not None:
            return (Xc_reconstructed + self.mean_).astype(np.float64)
        return Xc_reconstructed.astype(np.float64)

    # Ekach method madhye transform + inverse_transform karun reconstruction karto.
    def reconstruct(self, X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
        return self.inverse_transform(self.transform(X))

    # Fitted transformer chi state disk var save karto.
    def save(self, path: str):
        state = {
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "singular_values_": self.singular_values_,
            "total_variance_": self.total_variance_,
            "n_features_in_": self.n_features_in_,
            "effective_rank_": self.effective_rank_,
            "requested_rank_": self.requested_rank_,
            "centered_": self.centered_,
            "config": {
                "method": self.config_.method.value,
                "k": self.config_.k,
                "seed": self.config_.seed,
                "eig_tol": self.config_.eig_tol,
                "eig_max_iter": self.config_.eig_max_iter,
                "psd_clip_rtol": self.config_.psd_clip_rtol,
                "pca_svd_solver": self.config_.pca_svd_solver,
                "tsvd_algorithm": self.config_.tsvd_algorithm,
                "tsvd_n_iter": self.config_.tsvd_n_iter,
                "output_dtype": self.config_.output_dtype,
                "max_dense_cov_dim": getattr(self.config_, "max_dense_cov_dim", 5000),
            },
            "versions": {
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            "schema_version": 1
        }
        joblib.dump(state, path)

    # Previously saved transformer state load karun transformer object rebuild karto.
    @classmethod
    def load(cls, path: str) -> "DecompositionTransformer":
        state = joblib.load(path)
        if state["versions"]["numpy"] != np.__version__:
            logger.warning("Numpy version mismatch")
        if state["versions"]["sklearn"] != sklearn.__version__:
            logger.warning("Sklearn version mismatch")

        config = DecompositionConfig(**state["config"])
        transformer = cls(config)
        transformer.components_ = state["components_"]
        transformer.explained_variance_ = state["explained_variance_"]
        transformer.singular_values_ = state["singular_values_"]
        transformer.total_variance_ = state["total_variance_"]
        transformer.n_features_in_ = state["n_features_in_"]
        transformer.effective_rank_ = state.get("effective_rank_", transformer.components_.shape[0])
        transformer.requested_rank_ = state.get("requested_rank_", config.k)
        transformer.centered_ = state.get("centered_", True)

        if state["mean_"] is not None:
            transformer.centerer = Centerer()
            transformer.centerer.mean_ = state["mean_"]
            transformer.centerer.n_features_in_ = state["n_features_in_"]
        else:
            transformer.centerer = None

        return transformer