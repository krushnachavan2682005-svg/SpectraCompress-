"""
inference.py - Day 7 Production Inference Service for SpectraCompress.

Serves batch predictions using a cryptographically verified, frozen model artifact:
  - Strict checksum verification on load (arrays_sha256 and manifest_sha256)
  - Dense and CSR sparse support (CSR stays CSR, never densified)
  - Uses the exact same transform() as training (zero train-serving skew)
  - Loaded artifact is never refit
  - Actionable error messages for missing or tampered artifacts
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from serialization import load_transformer, ArtifactIntegrityError, SpectraCompressError
from decomposition import DecompositionTransformer


def _resolve_artifact_prefix(artifact_path: Union[str, Path]) -> Path:
    p = Path(artifact_path)
    if p.is_dir():
        return p / "spectra_compress_model"
    if p.suffix in [".json", ".npz"]:
        return p.with_suffix("")
    return p


def _check_artifact_exists(artifact_path: Union[str, Path]) -> Tuple[Path, Path]:
    """Resolves artifact paths and raises actionable FileNotFoundError if missing."""
    prefix = _resolve_artifact_prefix(artifact_path)
    npz_p = prefix.with_suffix(".npz")
    manifest_p = prefix.with_suffix(".json")

    if not (npz_p.exists() and manifest_p.exists()):
        raise FileNotFoundError(
            f"Artifact missing at '{artifact_path}'. Expected '{npz_p}' and '{manifest_p}'. "
            f"To generate this artifact, run: 'python run_pipeline.py --config final_config.json'"
        )
    return npz_p, manifest_p


def predict_batch(
    X_new: Union[np.ndarray, sp.spmatrix],
    artifact_path: Union[str, Path] = "artifacts/spectra_compress_model",
) -> Dict[str, Any]:
    """
    Executes inference on a batch of new samples using a frozen, checksum-verified artifact.

    Args:
        X_new: Dense 2D numpy.ndarray or Scipy CSR sparse matrix of shape (N, 784).
        artifact_path: Path to directory or base filename of the saved artifact.

    Returns:
        Dict with keys:
          - "embedding": np.ndarray of shape (N, k) with low-dimensional projections
          - "reconstruction_error": np.ndarray of shape (N,) with per-row squared reconstruction error e_i
          - "metadata": dict containing model_version, method, k, arrays_sha256, schema_version

    Raises:
        FileNotFoundError: If the artifact does not exist (with actionable creation instructions).
        ValueError: If input shape, dimensions, or values are invalid (e.g. wrong width).
        ArtifactIntegrityError: If artifact checksums or manifest have been tampered with.
    """
    # 1. Actionable missing-artifact check
    _check_artifact_exists(artifact_path)
    resolved_prefix = _resolve_artifact_prefix(artifact_path)

    # 2. Cryptographic checksum-verified load (strict=True checks manifest_sha256 & arrays_sha256)
    transformer = load_transformer(resolved_prefix, strict=True)

    # Read manifest for metadata
    _, manifest_p = _check_artifact_exists(artifact_path)
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 3. Input validation
    if not (isinstance(X_new, np.ndarray) or sp.issparse(X_new)):
        raise ValueError(f"Expected numpy ndarray or scipy.sparse matrix, got {type(X_new)}")

    if X_new.ndim != 2:
        raise ValueError(f"Expected 2D input array, got shape {X_new.shape}")

    expected_d = transformer.n_features_in_ or transformer.components_.shape[1]
    if X_new.shape[1] != expected_d:
        raise ValueError(f"Wrong input width: expected {expected_d} features, got {X_new.shape[1]}")

    is_sparse = sp.issparse(X_new)
    if is_sparse:
        if X_new.nnz > 0 and not np.all(np.isfinite(X_new.data)):
            raise ValueError("Sparse matrix contains non-finite values (NaN or Inf)")
        if not sp.isspmatrix_csr(X_new):
            X_input = X_new.tocsr()
        else:
            X_input = X_new
    else:
        if not np.all(np.isfinite(X_new)):
            raise ValueError("Input array contains non-finite values (NaN or Inf)")
        X_input = X_new

    # 4. Transform using identical training transform logic (no refit, zero train/serve skew)
    embedding = transformer.transform(X_input)

    # 5. Reconstruct and compute per-row reconstruction error
    # For dense: e_i = ||x_i - \hat{x}_i||^2
    # For sparse: ||x_i - \hat{x}_i||^2 = ||x_i||^2 - 2 <x_i, \hat{x}_i> + ||\hat{x}_i||^2
    # CSR input stays CSR: .toarray() / .todense() is NEVER called on X_input.
    X_hat = transformer.reconstruct(X_input)

    if is_sparse:
        sq_norm_X = np.asarray(X_input.multiply(X_input).sum(axis=1)).ravel()
        dot_X_Xhat = np.asarray(X_input.multiply(X_hat).sum(axis=1)).ravel()
        sq_norm_Xhat = np.sum(X_hat ** 2, axis=1)
        recon_error = np.maximum(0.0, sq_norm_X - 2.0 * dot_X_Xhat + sq_norm_Xhat)
    else:
        recon_error = np.sum((X_input - X_hat) ** 2, axis=1)

    metadata = {
        "model_version": str(manifest.get("schema_version", 1)),
        "method": transformer.config_.method.value,
        "k": int(transformer.config_.k),
        "arrays_sha256": manifest.get("arrays_sha256"),
        "schema_version": manifest.get("schema_version", 1),
    }

    return {
        "embedding": embedding,
        "reconstruction_error": recon_error,
        "metadata": metadata,
    }
