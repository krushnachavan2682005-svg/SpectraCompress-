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


# From-scratch method madhye top-k eigenpairs calculate karnyasathi helper.
# Jar power iteration converge zali nahi, tar specific convergence error raise karto.
def _run_power_iteration(A, k, tol, max_iter, seed):
    try:
        return top_k_eigenpairs(A=A, k=k, tol=tol, max_iter=max_iter, seed=seed)
    except InputValidation as e:
        if "did not converge" in getattr(e, "message", ""):
            raise NonConvergenceError(message=e.message) from e
        raise e


# Logging mule warning ani important events console/log madhye record karta yetat.
logger = logging.getLogger(__name__)


# PCA/decomposition sathi konti implementation use karaychi te define karto.
class MethodType(Enum):
    from_scratch_covariance = "from_scratch_covariance"  # Aapan manually covariance + eigenpairs vaparto.
    sklearn_pca = "sklearn_pca"                          # sklearn chi PCA implementation.
    sklearn_truncated_svd = "sklearn_truncated_svd"      # sklearn chi TruncatedSVD implementation.


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

    # Fitted Centerer madhun training mean access karnyasathi property.
    @property
    def mean_(self):
        if self.centerer is None:
            return None
        return self.centerer.mean_

    # Training data var decomposition shikto.
    # Ya method madhye centering, components, variance ani singular values calculate hotat.
    def fit(self, X_train: np.ndarray) -> "DecompositionTransformer":
        start_time = time.perf_counter()

        # Training data cha feature-wise mean shikun data center karto.
        self.centerer = Centerer()
        self.centerer.fit(X_train)
        Xc = self.centerer.transform(X_train)

        # n = rows/samples, d = features.
        n, d = Xc.shape
        self.n_features_in_ = d
        k = self.config_.k

        # K available samples/features chya constraints madhye aahe ka te check karto.
        if not (1 <= k <= min(n - 1, d)):
            raise InputValidation(message=f"k must be between 1 and min(n-1, d). Got k={k}")

        # Centered training data madhil total sample variance calculate karto.
        self.total_variance_ = float(np.sum(Xc ** 2) / (n - 1))

        # ---------------------------------------------------------
        # Method 1: From-scratch covariance decomposition
        # ---------------------------------------------------------
        if self.config_.method == MethodType.from_scratch_covariance:

            # Centered data varun covariance matrix C calculate karto.
            C = (Xc.T @ Xc) / (n - 1)

            # Floating-point asymmetry kami karnyasathi matrix symmetrize karto.
            C = (C + C.T) / 2.0

            # Power iteration vaprun top-k eigenpairs shodhto.
            eigenpairs = _run_power_iteration(
                A=C,
                k=k,
                tol=self.config_.eig_tol,
                max_iter=self.config_.eig_max_iter,
                seed=self.config_.seed
            )

            # Eigenpair results madhun eigenvalues ani eigenvectors separate karto.
            evals = np.array([ep["eigenvalue"] for ep in eigenpairs])
            evecs = np.array([ep["eigenvector"] for ep in eigenpairs]).T

            # Eigenvalues descending order madhye sort karto.
            # Sarvat motha eigenvalue pahilya component shi associate hoto.
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            # Numerical errors mule alele tiny negative eigenvalues handle karto.
            evals = clip_psd_eigenvalues(evals, self.config_.psd_clip_rtol)

            # Components cha shape (k, d) thevto; pratyek row ek component aahe.
            self.components_ = evecs.T

            # Deterministic sign convention:
            # Pratyek component madhil largest absolute loading positive thevto.
            # Eigenvector cha sign mathematically arbitrary asu shakto.
            max_abs_cols = np.argmax(np.abs(self.components_), axis=1)
            signs = np.sign(self.components_[range(self.components_.shape[0]), max_abs_cols])

            # Largest loading exactly zero asel tar sign +1 thevto.
            signs[signs == 0] = 1
            self.components_ *= signs[:, np.newaxis]

            # Covariance eigenvalues varun singular values reconstruct karto.
            self.singular_values_ = np.sqrt(evals * (n - 1))

        # ---------------------------------------------------------
        # Method 2: sklearn PCA
        # ---------------------------------------------------------
        elif self.config_.method == MethodType.sklearn_pca:

            # Centered training data var sklearn PCA fit karto.
            pca = PCA(n_components=k, svd_solver=self.config_.pca_svd_solver, random_state=self.config_.seed)
            pca.fit(Xc)

            # Learned principal directions ani singular values store karto.
            self.components_ = pca.components_
            self.singular_values_ = pca.singular_values_

        # ---------------------------------------------------------
        # Method 3: sklearn TruncatedSVD
        # ---------------------------------------------------------
        elif self.config_.method == MethodType.sklearn_truncated_svd:

            # ARPACK algorithm la k, min(n,d) peksha lahan asava lagto.
            if self.config_.tsvd_algorithm == "arpack" and k >= min(n, d):
                raise InputValidation(message="arpack cannot handle k >= min(n, d)")

            # Ya implementation madhye TruncatedSVD la centered data Xc deto.
            tsvd = TruncatedSVD(
                n_components=k,
                algorithm=self.config_.tsvd_algorithm,
                n_iter=self.config_.tsvd_n_iter,
                random_state=self.config_.seed
            )
            tsvd.fit(Xc)

            # Learned components ani singular values store karto.
            self.components_ = tsvd.components_
            self.singular_values_ = tsvd.singular_values_

        # Unsupported method asel tar error raise karto.
        else:
            raise InputValidation(message=f"Unknown method {self.config_.method}")

        # Sarva backends sathi explained variance calculate karto.
        # Singular values squared / (n-1) = sample explained variance.
        self.explained_variance_ = (self.singular_values_ ** 2) / (n - 1)

        # Pratyek component chi variance / total variance.
        # Ya ratio madhun component kiti variance capture karto te kalta.
        self.explained_variance_ratio_ = self.explained_variance_ / self.total_variance_

        # Output arrays float64 madhye convert karto.
        dtype = np.float64
        self.components_ = self.components_.astype(dtype)
        self.explained_variance_ = self.explained_variance_.astype(dtype)
        self.singular_values_ = self.singular_values_.astype(dtype)
        self.explained_variance_ratio_ = self.explained_variance_ratio_.astype(dtype)

        # Fit process la kiti vel lagla te store karto.
        self.fit_time_s_ = time.perf_counter() - start_time

        # Fitted transformer parat return karto.
        return self

    # Input data la learned principal components madhye project karto.
    # Output shape: (number of rows, k).
    def transform(self, X: np.ndarray) -> np.ndarray:

        # Fit zala nasel tar transform allow karat nahi.
        if self.components_ is None or self.centerer is None or not self.centerer.is_fitted:
            raise NotFittedError(message="Transformer is not fitted.")

        # Training madhye shiklela mean vaprun input center karto.
        Xc = self.centerer.transform(X)

        # Centered data la component directions var project karto.
        Z = Xc @ self.components_.T
        return Z

    # Reduced representation Z madhun original feature space madhye reconstruction karto.
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:

        # Transformer fitted asne required aahe.
        if self.components_ is None or self.centerer is None or not self.centerer.is_fitted:
            raise NotFittedError(message="Transformer is not fitted.")

        # Reduced representation 2D matrix asavi.
        if Z.ndim != 2:
            raise InputValidation(message="Z must be 2D")

        # Z madhye columns chi sankhya retained components k shi match zali pahije.
        if Z.shape[1] != self.components_.shape[0]:
            raise ShapeMismatchError(message=f"Z shape {Z.shape} does not match k={self.components_.shape[0]}")

        # Reduced representation la principal components vaprun original feature space madhye map karto.
        Xc_reconstructed = Z @ self.components_

        # Centering undo karun original feature scale madhye reconstructed data return karto.
        return self.centerer.inverse_transform(Xc_reconstructed)

    # Ekach method madhye transform + inverse_transform karun reconstruction karto.
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.inverse_transform(self.transform(X))

    # Fitted transformer chi state disk var save karto.
    def save(self, path: str):

        # Transformer punha load karta yava mhanun required learned values ani config store karto.
        state = {
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "singular_values_": self.singular_values_,
            "total_variance_": self.total_variance_,
            "n_features_in_": self.n_features_in_,

            # Configuration values plain dictionary madhye store karto.
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

            # Save kartana use zalele NumPy ani sklearn versions record karto.
            "versions": {
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },

            # Saved state cha format identify karnyasathi schema version.
            "schema_version": 1
        }

        # joblib vaprun state specified path var serialize karto.
        joblib.dump(state, path)

    # Previously saved transformer state load karun transformer object rebuild karto.
    @classmethod
    def load(cls, path: str) -> "DecompositionTransformer":

        # Disk varun saved state read karto.
        state = joblib.load(path)

        # Current library versions ani saved versions compare karto.
        # Difference asel tar warning deto.
        if state["versions"]["numpy"] != np.__version__:
            logger.warning("Numpy version mismatch")
        if state["versions"]["sklearn"] != sklearn.__version__:
            logger.warning("Sklearn version mismatch")

        # Saved configuration varun config object punha create karto.
        config = DecompositionConfig(**state["config"])
        transformer = cls(config)

        # Saved learned PCA attributes restore karto.
        transformer.components_ = state["components_"]
        transformer.explained_variance_ = state["explained_variance_"]
        transformer.singular_values_ = state["singular_values_"]
        transformer.total_variance_ = state["total_variance_"]
        transformer.n_features_in_ = state["n_features_in_"]

        # Centerer object rebuild karun saved mean ani feature count restore karto.
        transformer.centerer = Centerer()
        transformer.centerer.mean_ = state["mean_"]
        transformer.centerer.n_features_in_ = state["n_features_in_"]

        # Reconstructed transformer return karto.
        return transformer