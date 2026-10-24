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
    """Computes sample-averaged squared L2 reconstruction error.

    Mathematical Definition:
        MSE_sample = (1 / n_samples) * sum_{i=1}^{n_samples} ||x_i - x_hat_i||_2^2
                   = (1 / n_samples) * ||X - X_reconstructed||_F^2

    Interpretation & Units:
        Represents the average squared Euclidean reconstruction error per row/sample.
        Note that this is NOT elementwise MSE. When the number of features (d) is non-zero,
        the elementwise mean squared error is related by:
            MSE_elementwise = MSE_sample / n_features
                            = (1 / (n_samples * n_features)) * ||X - X_reconstructed||_F^2

    Args:
        original: Dense array of shape (n_samples, n_features).
        reconstructed: Dense reconstructed array of shape (n_samples, n_features).

    Returns:
        float: Sample-averaged squared L2 reconstruction error.
    """
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


def sparse_reconstruction_mse(
    original: np.ndarray,
    reconstructed: np.ndarray,
    batch_size: int = 1000
) -> float:
    """Computes sample-averaged squared L2 reconstruction error in row batches.

    Mathematical Definition:
        MSE_sample = (1 / n_samples) * sum_{i=1}^{n_samples} ||x_i - x_hat_i||_2^2
                   = (1 / n_samples) * ||X - X_reconstructed||_F^2

    Interpretation & Units:
        Represents the average squared Euclidean reconstruction error per row/sample.
        Note that this is NOT elementwise MSE. When the number of features (d) is non-zero,
        the elementwise mean squared error is related by:
            MSE_elementwise = MSE_sample / n_features
                            = (1 / (n_samples * n_features)) * ||X - X_reconstructed||_F^2

    Sparse Chunking and Memory Behavior:
        Evaluates reconstruction error in row batches of size `batch_size` (default: 1000).
        For sparse inputs or fitted transformer objects, row batches are densified and
        projected in bounded chunks of shape (batch_size, n_features). This guarantees that
        a full (n_samples, n_features) dense error matrix is never allocated in memory,
        preventing Out-Of-Memory (OOM) errors on large sparse datasets.

    Args:
        original: Original matrix (scipy.sparse matrix or dense ndarray) of shape (n_samples, n_features).
        reconstructed: Reconstructed matrix of shape (n_samples, n_features), or a fitted transformer.
        batch_size: Row batch size for chunked evaluation (default: 1000).

    Returns:
        float: Sample-averaged squared L2 reconstruction error.
    """
    import scipy.sparse as sp

    # Handle case where reconstructed is a fitted transformer object
    if hasattr(reconstructed, "transform") and hasattr(reconstructed, "inverse_transform"):
        n_samples = original.shape[0]
        total_sq_err = 0.0
        for i in range(0, n_samples, batch_size):
            batch_orig = original[i : i + batch_size]
            batch_orig_dense = batch_orig.toarray() if sp.issparse(batch_orig) else np.asarray(batch_orig, dtype=np.float64)
            batch_recon = reconstructed.inverse_transform(reconstructed.transform(batch_orig))
            total_sq_err += float(np.sum((batch_orig_dense - batch_recon) ** 2))
        return float(total_sq_err / n_samples)

    if not hasattr(original, "shape") or not hasattr(reconstructed, "shape"):
        raise InputValidation(message="Inputs must have a shape attribute")

    if len(original.shape) != 2 or len(reconstructed.shape) != 2:
        raise InputValidation(message="Inputs must be 2D arrays or sparse matrices")

    if original.shape != reconstructed.shape:
        raise ShapeMismatchError(message=f"Shape mismatch: {original.shape} vs {reconstructed.shape}")

    n_samples, n_features = original.shape
    total_sq_err = 0.0

    for i in range(0, n_samples, batch_size):
        batch_orig = original[i : i + batch_size]
        batch_recon = reconstructed[i : i + batch_size]

        batch_orig_dense = batch_orig.toarray() if sp.issparse(batch_orig) else np.asarray(batch_orig, dtype=np.float64)
        batch_recon_dense = batch_recon.toarray() if sp.issparse(batch_recon) else np.asarray(batch_recon, dtype=np.float64)

        diff = batch_orig_dense - batch_recon_dense
        total_sq_err += float(np.sum(diff ** 2))

    return float(total_sq_err / n_samples)


