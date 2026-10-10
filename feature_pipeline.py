import logging
from error import InputValidation, ShapeMismatchError, NotFittedError
import numpy as np

class Centerer:
    """Centers data by subtracting the mean of each feature. Output dtype is always float64."""
    def __init__(self):
        self.mean_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    def _validate(self, X: np.ndarray, *, fitting: bool) -> None:
        if not isinstance(X, np.ndarray):
            raise InputValidation(message=f"Expected numpy ndarray, got {type(X)}")
        if X.dtype not in (np.float32, np.float64):
            raise InputValidation(message=f"Dtype must be float32 or float64, got {X.dtype}")
        if X.ndim != 2:
            raise InputValidation(message=f"Expected 2 dimensions, got {X.ndim}")
        n, d = X.shape
        if fitting and n < 2:
            raise InputValidation(message=f"fit requires n >= 2, got {n}")
        if not fitting and n < 1:
            raise InputValidation(message=f"transform requires n >= 1, got {n}")
        if not np.isfinite(X).all():
            raise InputValidation(message="Data contains NaN or infinite values")
        if not fitting:
            if d != self.n_features_in_:
                raise ShapeMismatchError(message=f"expected d={self.n_features_in_}, got d={d}")

    def fit(self, X: np.ndarray) -> "Centerer":
        self._validate(X, fitting=True)
        mean = np.mean(X, axis=0, dtype=np.float64)
        self.mean_ = mean
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise NotFittedError(message="The X is not fitted")
        self._validate(X, fitting=False)
        return (X - self.mean_).astype(np.float64)

    def inverse_transform(self, Xc: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise NotFittedError(message="The X is not fitted")
        self._validate(Xc, fitting=False)
        return (Xc + self.mean_).astype(np.float64)