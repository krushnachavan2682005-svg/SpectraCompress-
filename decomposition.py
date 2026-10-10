import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD

from error import InputValidation, ShapeMismatchError, NotFittedError, NonConvergenceError
from feature_pipeline import Centerer
from linalg_core import top_k_eigenpairs

def _run_power_iteration(A, k, tol, max_iter, seed):
    try:
        return top_k_eigenpairs(A=A, k=k, tol=tol, max_iter=max_iter, seed=seed)
    except InputValidation as e:
        if "did not converge" in getattr(e, "message", ""):
            raise NonConvergenceError(message=e.message) from e
        raise e

logger = logging.getLogger(__name__)

class MethodType(Enum):
    from_scratch_covariance = "from_scratch_covariance"
    sklearn_pca = "sklearn_pca"
    sklearn_truncated_svd = "sklearn_truncated_svd"

@dataclass(frozen=True)
class DecompositionConfig:
    method: MethodType
    k: int = 10
    seed: Optional[int] = 42
    eig_tol: float = 1e-10
    eig_max_iter: int = 1000
    psd_clip_rtol: float = 1e-8
    pca_svd_solver: str = "full"
    tsvd_algorithm: str = "arpack"
    tsvd_n_iter: int = 5
    output_dtype: str = "float64"

    def __post_init__(self):
        if isinstance(self.method, str):
            try:
                object.__setattr__(self, 'method', MethodType(self.method))
            except ValueError:
                raise InputValidation(message=f"Unknown method {self.method}")
        elif not isinstance(self.method, MethodType):
            raise InputValidation(message="Invalid method type")

        if self.k < 1:
            raise InputValidation(message="k must be >= 1")
        if self.eig_tol <= 0:
            raise InputValidation(message="eig_tol must be > 0")
        if self.eig_max_iter < 1:
            raise InputValidation(message="eig_max_iter must be >= 1")
        if self.psd_clip_rtol < 0:
            raise InputValidation(message="psd_clip_rtol must be >= 0")
        if self.output_dtype not in {"float64"}:
            raise InputValidation(message="output_dtype must be 'float64'")

def clip_psd_eigenvalues(eigvals: np.ndarray, rtol: float) -> np.ndarray:
    """Clips tiny negative eigenvalues to 0. Values == -tol are clipped."""
    if eigvals.size == 0:
        raise InputValidation(message="Empty input")
    max_val = np.max(np.abs(eigvals))
    tol = rtol * max_val

    # Check for invalid values
    if np.any(eigvals < -tol):
        raise InputValidation(message="Eigenvalue too negative for PSD clipping")

    # Clip values in [-tol, 0)
    mask = (eigvals >= -tol) & (eigvals < 0)
    count = np.sum(mask)
    if count > 0:
        logger.warning(f"Warning: {count} negative eigenvalues were clipped to zero.")
    
    clipped = np.where(mask, 0.0, eigvals)
    return clipped.astype(np.float64)

class DecompositionTransformer:
    def __init__(self, config: DecompositionConfig):
        self.config_ = config
        self.centerer = None
        self.components_ = None
        self.explained_variance_ = None
        self.singular_values_ = None
        self.total_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_features_in_ = None
        self.fit_time_s_ = None

    @property
    def mean_(self):
        if self.centerer is None:
            return None
        return self.centerer.mean_

    def fit(self, X_train: np.ndarray) -> "DecompositionTransformer":
        start_time = time.perf_counter()

        self.centerer = Centerer()
        self.centerer.fit(X_train)
        Xc = self.centerer.transform(X_train)
        
        n, d = Xc.shape
        self.n_features_in_ = d
        k = self.config_.k

        if not (1 <= k <= min(n - 1, d)):
            raise InputValidation(message=f"k must be between 1 and min(n-1, d). Got k={k}")

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
            
            # Sort descending
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            # Clip negative eigenvalues
            evals = clip_psd_eigenvalues(evals, self.config_.psd_clip_rtol)
            
            self.components_ = evecs.T
            # Deterministic sign convention: largest absolute value in each component is positive
            max_abs_cols = np.argmax(np.abs(self.components_), axis=1)
            signs = np.sign(self.components_[range(self.components_.shape[0]), max_abs_cols])
            # Handle zeros
            signs[signs == 0] = 1
            self.components_ *= signs[:, np.newaxis]

            # Recompute singular_values_ from eigenvalues
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

        # Common attributes for all backends
        self.explained_variance_ = (self.singular_values_ ** 2) / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.total_variance_
        
        # Cast outputs
        dtype = np.float64
        self.components_ = self.components_.astype(dtype)
        self.explained_variance_ = self.explained_variance_.astype(dtype)
        self.singular_values_ = self.singular_values_.astype(dtype)
        self.explained_variance_ratio_ = self.explained_variance_ratio_.astype(dtype)

        self.fit_time_s_ = time.perf_counter() - start_time
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.centerer is None or not self.centerer.is_fitted:
            raise NotFittedError(message="Transformer is not fitted.")
        Xc = self.centerer.transform(X)
        Z = Xc @ self.components_.T
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.centerer is None or not self.centerer.is_fitted:
            raise NotFittedError(message="Transformer is not fitted.")
        if Z.ndim != 2:
            raise InputValidation(message="Z must be 2D")
        if Z.shape[1] != self.components_.shape[0]:
            raise ShapeMismatchError(message=f"Z shape {Z.shape} does not match k={self.components_.shape[0]}")
        Xc_reconstructed = Z @ self.components_
        return self.centerer.inverse_transform(Xc_reconstructed)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.inverse_transform(self.transform(X))

    def save(self, path: str):
        state = {
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "singular_values_": self.singular_values_,
            "total_variance_": self.total_variance_,
            "n_features_in_": self.n_features_in_,
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
            },
            "versions": {
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            "schema_version": 1
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "DecompositionTransformer":
        state = joblib.load(path)
        
        # Check versions
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
        
        transformer.centerer = Centerer()
        transformer.centerer.mean_ = state["mean_"]
        transformer.centerer.n_features_in_ = state["n_features_in_"]
        
        return transformer

