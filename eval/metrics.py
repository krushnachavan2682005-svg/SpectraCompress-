import numpy as np
from error import InputValidation, ShapeMismatchError

def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    if original.ndim != 2 or reconstructed.ndim != 2:
        raise InputValidation(message="Inputs must be 2D arrays")
    if original.shape != reconstructed.shape:
        raise ShapeMismatchError(message=f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    error = np.sqrt(np.sum((original - reconstructed) ** 2, axis=1))
    return error

def mean_squared_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.ndim != 2 or reconstructed.ndim != 2:
        raise InputValidation(message="Inputs must be 2D arrays")
    if original.shape != reconstructed.shape:
        raise ShapeMismatchError(message="Shape mismatch")
    return float(np.mean(np.sum((original - reconstructed) ** 2, axis=1)))

def variance_explained(eigenvalues: np.ndarray, total_variance: float) -> np.ndarray:
    if total_variance <= 0 or not np.isfinite(total_variance):
        raise InputValidation(message="total_variance must be > 0 and finite")
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0):
        raise InputValidation(message="eigenvalues must be finite and >= 0")
    importance = (eigenvalues / total_variance).astype(np.float64)
    return importance

def cumulative_variance_explained(importance: np.ndarray) -> np.ndarray:
    # returns cumulative sum as a fraction in [0, 1]
    return np.cumsum(importance)

def component_cosine_similarity(W_a: np.ndarray, W_b: np.ndarray) -> np.ndarray:
    if W_a.ndim != 2 or W_b.ndim != 2:
        raise InputValidation(message="Inputs must be 2D arrays")
    if W_a.shape != W_b.shape:
        raise ShapeMismatchError(message="Shapes must be identical")
    
    norm_a = np.linalg.norm(W_a, axis=1)
    norm_b = np.linalg.norm(W_b, axis=1)
    
    if np.any(norm_a == 0) or np.any(norm_b == 0):
        raise InputValidation(message="Zero-norm row detected")
        
    dot_product = np.sum(W_a * W_b, axis=1)
    similarity = np.abs(dot_product / (norm_a * norm_b))
    return similarity

def subspace_distance(W_a: np.ndarray, W_b: np.ndarray) -> float:
    """Computes projector difference ||W_a^T W_a - W_b^T W_b||_F.
    Assumes orthonormal rows; meaningful for equal k."""
    if W_a.ndim != 2 or W_b.ndim != 2:
        raise InputValidation(message="Inputs must be 2D")
    if W_a.shape[1] != W_b.shape[1]:
        raise ShapeMismatchError(message="Input dimensions (d) must be equal")
    
    pca_1 = W_a.T @ W_a
    pca_2 = W_b.T @ W_b
    difference = pca_1 - pca_2
    return float(np.linalg.norm(difference, ord='fro'))

