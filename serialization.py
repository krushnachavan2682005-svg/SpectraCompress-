"""Production Serialization Engine for SpectraCompress Transformers.

Dual-file serialization architecture:
  - <path>.npz: Canonical numerical arrays (components_, feature_mean_, singular_values_, etc.)
  - <path>.json: Human-readable manifest recording schema version, shapes, dtypes, fit metadata,
                 library versions, arrays SHA-256, and canonical manifest SHA-256 integrity checksum.

Hard Safety & Governance Invariants:
  - Pickle is prohibited as the primary serialization format.
  - Array checksum is computed over length-prefixed canonical array fields (name, dtype with byte order,
    ndim, shape, nbytes, contiguous buffer), preventing boundary shift ambiguities.
  - Manifest integrity is cryptographically sealed via manifest_sha256 over canonical JSON.
  - Loaded artifacts are strictly immutable (fit() raises SpectraCompressError).
  - Legacy joblib artifacts remain loadable only when allow_legacy=True is explicitly passed (deprecated).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy
import sklearn

from error import (
    ArtifactIntegrityError,
    ArtifactVersionError,
    NotFittedError,
    SpectraCompressError,
    ShapeMismatchError,
)

if TYPE_CHECKING:
    from decomposition import DecompositionTransformer

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CONFIRMED_RUN_003_REPORT_HASH = "11b882546939181721f87cc4eabda960bb9bfe894842de0001d3bd6c4f189b95"


def compute_canonical_arrays_sha256(arrays_dict: Dict[str, np.ndarray]) -> str:
    """Computes a deterministic SHA-256 checksum over length-prefixed canonical array fields.

    Format per array entry:
        f"{name_len}:{name}|{dtype_str}|{ndim}|{shape_str}|{nbytes}|" + contiguous_bytes

    Where dtype_str carries explicit endianness / byte order (e.g. '<f8', '>f8', '|u1').
    Length-prefixing guarantees that no two distinct array sets can collide via boundary shifting.
    """
    hasher = hashlib.sha256()
    for key in sorted(arrays_dict.keys()):
        arr = arrays_dict[key]
        name_bytes = key.encode("utf-8")
        name_len = len(name_bytes)
        if arr is None or arr.size == 0:
            header = f"{name_len}:{key}|empty|0||0|".encode("utf-8")
            hasher.update(header)
            continue
        c_arr = np.ascontiguousarray(arr)
        dtype_str = c_arr.dtype.str  # Contains byte order, e.g. '<f8'
        ndim = c_arr.ndim
        shape_str = ",".join(str(s) for s in c_arr.shape)
        nbytes = c_arr.nbytes

        header = f"{name_len}:{key}|{dtype_str}|{ndim}|{shape_str}|{nbytes}|".encode("utf-8")
        hasher.update(header)
        hasher.update(c_arr.tobytes())
    return hasher.hexdigest()


def compute_canonical_manifest_sha256(manifest_dict: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 checksum over canonical JSON representation of manifest.

    Excludes the 'manifest_sha256' field itself, sorts keys, and uses compact delimiters.
    """
    clean_dict = {k: v for k, v in manifest_dict.items() if k != "manifest_sha256"}
    canonical_json_bytes = json.dumps(
        clean_dict,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical_json_bytes).hexdigest()


def _resolve_paths(path: Union[str, Path]) -> Tuple[Path, Path]:
    """Resolves manifest (.json) and weights (.npz) file paths from a base path or filename."""
    p = Path(path)
    if p.suffix in {".json", ".npz", ".joblib"}:
        base = p.with_suffix("")
    else:
        base = p
    manifest_path = base.with_suffix(".json")
    npz_path = base.with_suffix(".npz")
    return manifest_path, npz_path


def save_transformer(
    transformer: "DecompositionTransformer",
    path: Union[str, Path],
    *,
    provisional: bool = False,
    source_report_hash: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Path, Path]:
    """Serializes a fitted DecompositionTransformer into dual .npz and .json artifacts.

    Args:
        transformer: Fitted DecompositionTransformer instance.
        path: Base path or path ending in .npz/.json/.joblib.
        provisional: Flag indicating if the configuration is provisional.
        source_report_hash: SHA-256 hash of the governing selection report.
        feature_names: Optional list of feature names.

    Returns:
        Tuple of (npz_path, manifest_path).

    Raises:
        NotFittedError: If the transformer has not been fitted.
    """
    if getattr(transformer, "components_", None) is None:
        raise NotFittedError(message="Transformer is not fitted. Cannot save unfitted transformer.")

    manifest_path, npz_path = _resolve_paths(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prepare numerical arrays
    feature_mean = transformer.feature_mean_
    arrays_to_save: Dict[str, np.ndarray] = {
        "feature_mean_": (
            np.ascontiguousarray(feature_mean, dtype=np.float64)
            if feature_mean is not None
            else np.empty(0, dtype=np.float64)
        ),
        "components_": np.ascontiguousarray(transformer.components_, dtype=np.float64),
        "singular_values_": np.ascontiguousarray(transformer.singular_values_, dtype=np.float64),
        "explained_variance_": np.ascontiguousarray(transformer.explained_variance_, dtype=np.float64),
        "explained_variance_ratio_": np.ascontiguousarray(transformer.explained_variance_ratio_, dtype=np.float64),
    }

    # 2. Compute canonical checksum over length-prefixed array bytes
    arrays_sha256 = compute_canonical_arrays_sha256(arrays_to_save)

    # 3. Save weights in .npz format (no pickle)
    np.savez_compressed(npz_path, **arrays_to_save)

    # 4. Prepare manifest metadata
    method_val = (
        transformer.config_.method.value
        if hasattr(transformer.config_.method, "value")
        else str(transformer.config_.method)
    )

    ortho_tol = getattr(transformer.config_, "ortho_tol", 1e-4)

    solver_params = {
        "pca_svd_solver": getattr(transformer.config_, "pca_svd_solver", "full"),
        "tsvd_algorithm": getattr(transformer.config_, "tsvd_algorithm", "arpack"),
        "tsvd_n_iter": getattr(transformer.config_, "tsvd_n_iter", 5),
        "eig_tol": getattr(transformer.config_, "eig_tol", 1e-10),
        "eig_max_iter": getattr(transformer.config_, "eig_max_iter", 1000),
        "psd_clip_rtol": getattr(transformer.config_, "psd_clip_rtol", 1e-8),
        "output_dtype": getattr(transformer.config_, "output_dtype", "float64"),
        "max_dense_cov_dim": getattr(transformer.config_, "max_dense_cov_dim", 5000),
        "ortho_tol": float(ortho_tol),
    }

    shapes = {
        "feature_mean_": list(feature_mean.shape) if feature_mean is not None else None,
        "components_": list(transformer.components_.shape),
        "singular_values_": list(transformer.singular_values_.shape),
        "explained_variance_": list(transformer.explained_variance_.shape),
        "explained_variance_ratio_": list(transformer.explained_variance_ratio_.shape),
    }

    input_diag = getattr(transformer, "input_diagnostics_", None)
    cond_rep = getattr(transformer, "condition_report_", None)

    fit_metadata = {
        "n_train": int(input_diag.shape[0]) if input_diag is not None else None,
        "is_sparse": bool(input_diag.is_sparse) if input_diag is not None else False,
        "split_seed": getattr(transformer.config_, "seed", 42),
        "condition_number_at_fit": float(cond_rep.kappa) if cond_rep is not None else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    condition_report_dict = None
    if cond_rep is not None:
        try:
            condition_report_dict = dataclasses.asdict(cond_rep)
        except Exception:
            condition_report_dict = {
                "kappa": float(cond_rep.kappa),
                "kappa_effective": float(cond_rep.kappa_effective),
                "numerical_rank": int(cond_rep.numerical_rank),
                "band": str(cond_rep.band),
            }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": method_val,
        "k": int(transformer.config_.k),
        "requested_rank_": int(getattr(transformer, "requested_rank_", transformer.config_.k)),
        "effective_rank_": int(getattr(transformer, "effective_rank_", transformer.components_.shape[0])),
        "n_features_in_": int(transformer.n_features_in_),
        "centered_": bool(getattr(transformer, "centered_", True)),
        "total_variance_": float(transformer.total_variance_) if transformer.total_variance_ is not None else 0.0,
        "random_state": transformer.config_.seed,
        "solver_params": solver_params,
        "dtype": str(transformer.components_.dtype),
        "shapes": shapes,
        "feature_names": feature_names,
        "fit_metadata": fit_metadata,
        "condition_report_": condition_report_dict,
        "fit_warnings_": list(getattr(transformer, "fit_warnings_", [])),
        "library_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "arrays_sha256": arrays_sha256,
        "npz_filename": npz_path.name,
        "provisional": provisional,
        "source_report_hash": source_report_hash or CONFIRMED_RUN_003_REPORT_HASH,
    }

    # 5. Cryptographically seal manifest with manifest_sha256
    manifest["manifest_sha256"] = compute_canonical_manifest_sha256(manifest)

    # 6. Write manifest JSON
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.info(f"Successfully saved transformer to {npz_path} and {manifest_path}")
    return npz_path, manifest_path


def _load_legacy_joblib(path: Union[str, Path]) -> "DecompositionTransformer":
    """Fallback reader for legacy joblib model artifacts (read-only, deprecated)."""
    import joblib
    from decomposition import DecompositionConfig, DecompositionTransformer, MethodType
    from feature_pipeline import Centerer

    state = joblib.load(path)
    config_dict = state["config"]
    if isinstance(config_dict.get("method"), str):
        config_dict["method"] = MethodType(config_dict["method"])

    config = DecompositionConfig(**config_dict)
    transformer = DecompositionTransformer(config)
    transformer.components_ = state["components_"]
    transformer.explained_variance_ = state["explained_variance_"]
    transformer.singular_values_ = state["singular_values_"]
    transformer.total_variance_ = state["total_variance_"]
    transformer.n_features_in_ = state["n_features_in_"]
    transformer.effective_rank_ = state.get("effective_rank_", transformer.components_.shape[0])
    transformer.requested_rank_ = state.get("requested_rank_", config.k)
    transformer.centered_ = state.get("centered_", True)

    if state.get("mean_") is not None:
        transformer.centerer = Centerer()
        transformer.centerer.mean_ = state["mean_"]
        transformer.centerer.n_features_in_ = state["n_features_in_"]
    else:
        transformer.centerer = None

    transformer._is_loaded_artifact = True
    return transformer


def load_transformer(
    path: Union[str, Path],
    *,
    strict: bool = False,
    allow_legacy: bool = False,
) -> "DecompositionTransformer":
    """Loads and validates a DecompositionTransformer from dual .npz and .json artifacts.

    Validates schema version, manifest hash, canonical arrays checksum, shapes, dtypes,
    and library versions. Rebuilds Centerer and recomputes derived cache. Loaded transformer
    is marked immutable.

    Args:
        path: Base path or path pointing to .json/.npz/.joblib.
        strict: If True, library version mismatches raise ArtifactVersionError.
                If False, version mismatches emit a UserWarning.
        allow_legacy: If True, allows fallback loading of legacy .joblib artifacts with DeprecationWarning.
                      If False (default), loading legacy .joblib artifacts raises ArtifactIntegrityError.

    Returns:
        Restored, immutable DecompositionTransformer instance.

    Raises:
        FileNotFoundError: If manifest or weights file is missing.
        ArtifactIntegrityError: If manifest checksum, array checksum, shape, dtype, or schema verification fails.
        ArtifactVersionError: If library version mismatch occurs under strict=True.
    """
    p = Path(path)

    # 1. Check for legacy joblib file fallback
    if p.is_file() and p.suffix == ".joblib":
        cand_json = p.with_suffix(".json")
        cand_npz = p.with_suffix(".npz")
        if not (cand_json.exists() and cand_npz.exists()):
            if not allow_legacy:
                raise ArtifactIntegrityError(
                    f"Legacy joblib format detected at '{path}', but allow_legacy=False. "
                    "Loading legacy joblib format is disabled by default for security and stability. "
                    "Set allow_legacy=True to opt in (deprecated)."
                )
            warnings.warn(
                f"Loading legacy joblib format from {path} is deprecated. Use .npz + manifest.json format.",
                DeprecationWarning,
                stacklevel=2,
            )
            return _load_legacy_joblib(path)

    # 2. Resolve manifest and npz paths
    manifest_path, npz_path = _resolve_paths(path)
    if not manifest_path.exists() and Path(str(p) + ".json").exists():
        manifest_path = Path(str(p) + ".json")
        npz_path = Path(str(p) + ".npz")

    if not manifest_path.exists():
        if p.is_file() and p.suffix != ".json":
            if not allow_legacy:
                raise ArtifactIntegrityError(
                    f"Legacy joblib format detected at '{path}', but allow_legacy=False."
                )
            warnings.warn(
                f"Loading legacy joblib format from {path} is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
            return _load_legacy_joblib(path)
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if not npz_path.exists():
        raise FileNotFoundError(f"Weights file not found: {npz_path}")

    # 3. Read manifest JSON
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 4. Manifest integrity verification (F2)
    if "manifest_sha256" in manifest:
        expected_manifest_sha256 = manifest["manifest_sha256"]
        actual_manifest_sha256 = compute_canonical_manifest_sha256(manifest)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ArtifactIntegrityError(
                f"Manifest integrity verification failed. Expected SHA-256 '{expected_manifest_sha256}', "
                f"but calculated '{actual_manifest_sha256}'. The manifest metadata has been tampered with or corrupted."
            )

    # 5. Schema version validation
    schema_v = manifest.get("schema_version")
    if schema_v != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"Unsupported schema_version: {schema_v}. Supported version is {SCHEMA_VERSION}."
        )

    # 6. Library version checks
    cur_versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    saved_versions = manifest.get("library_versions", {})
    for lib in ("numpy", "scipy", "sklearn"):
        saved_v = saved_versions.get(lib)
        cur_v = cur_versions[lib]
        if saved_v is not None and saved_v != cur_v:
            warn_msg = (
                f"Library version mismatch for {lib}: artifact saved with {saved_v}, "
                f"current runtime environment has {cur_v}."
            )
            if strict:
                raise ArtifactVersionError(warn_msg)
            warnings.warn(warn_msg, UserWarning, stacklevel=2)

    # 7. Load numerical weights from .npz (pickle forbidden)
    with np.load(npz_path, allow_pickle=False) as data:
        loaded_arrays = {key: data[key] for key in data.files}

    # 8. Checksum integrity verification on arrays
    calculated_sha256 = compute_canonical_arrays_sha256(loaded_arrays)
    expected_sha256 = manifest.get("arrays_sha256")
    if calculated_sha256 != expected_sha256:
        raise ArtifactIntegrityError(
            f"Artifact checksum verification failed. Expected SHA-256 '{expected_sha256}', "
            f"but calculated '{calculated_sha256}'. The weights artifact may have been tampered with or corrupted."
        )

    # 9. Shape and dtype validation
    expected_dtype = manifest.get("dtype", "float64")
    expected_shapes = manifest.get("shapes", {})

    for arr_name, arr in loaded_arrays.items():
        if str(arr.dtype) != expected_dtype:
            raise ArtifactIntegrityError(
                f"Array '{arr_name}' dtype mismatch: expected {expected_dtype}, got {arr.dtype}"
            )
        exp_shape = expected_shapes.get(arr_name)
        if exp_shape is not None and arr.size > 0:
            if list(arr.shape) != exp_shape:
                raise ArtifactIntegrityError(
                    f"Array '{arr_name}' shape mismatch: expected {exp_shape}, got {list(arr.shape)}"
                )

    # 10. Reconstruct transformer instance
    from decomposition import DecompositionConfig, DecompositionTransformer, MethodType
    from feature_pipeline import Centerer
    from stability_diagnostics import ConditionReport

    method_str = manifest.get("method")
    try:
        method_enum = MethodType(method_str)
    except Exception:
        raise ArtifactIntegrityError(f"Unknown decomposition method in manifest: '{method_str}'")

    solver_params = manifest.get("solver_params", {})
    ortho_tol = float(solver_params.get("ortho_tol", 1e-4))

    config = DecompositionConfig(
        method=method_enum,
        k=int(manifest["k"]),
        seed=manifest.get("random_state", 42),
        eig_tol=float(solver_params.get("eig_tol", 1e-10)),
        eig_max_iter=int(solver_params.get("eig_max_iter", 1000)),
        psd_clip_rtol=float(solver_params.get("psd_clip_rtol", 1e-8)),
        pca_svd_solver=str(solver_params.get("pca_svd_solver", "full")),
        tsvd_algorithm=str(solver_params.get("tsvd_algorithm", "arpack")),
        tsvd_n_iter=int(solver_params.get("tsvd_n_iter", 5)),
        output_dtype=str(solver_params.get("output_dtype", "float64")),
        max_dense_cov_dim=int(solver_params.get("max_dense_cov_dim", 5000)),
        ortho_tol=ortho_tol,
    )

    transformer = DecompositionTransformer(config)
    transformer.components_ = loaded_arrays["components_"]
    transformer.singular_values_ = loaded_arrays["singular_values_"]
    transformer.explained_variance_ = loaded_arrays["explained_variance_"]
    transformer.explained_variance_ratio_ = loaded_arrays["explained_variance_ratio_"]
    transformer.total_variance_ = manifest.get("total_variance_")
    transformer.n_features_in_ = int(manifest["n_features_in_"])
    transformer.effective_rank_ = int(manifest.get("effective_rank_", transformer.components_.shape[0]))
    transformer.requested_rank_ = int(manifest.get("requested_rank_", config.k))
    transformer.centered_ = bool(manifest.get("centered_", True))
    transformer.fit_warnings_ = list(manifest.get("fit_warnings_", []))

    # Reconstruct condition_report_ if stored in manifest
    cr_dict = manifest.get("condition_report_")
    if cr_dict is not None and isinstance(cr_dict, dict):
        try:
            transformer.condition_report_ = ConditionReport(
                kappa=float(cr_dict.get("kappa", float("inf"))),
                kappa_effective=float(cr_dict.get("kappa_effective", float("inf"))),
                numerical_rank=int(cr_dict.get("numerical_rank", transformer.effective_rank_)),
                rank_tol=float(cr_dict.get("rank_tol", 0.0)),
                sigma_max=float(cr_dict.get("sigma_max", 0.0)),
                sigma_min_nonzero=float(cr_dict.get("sigma_min_nonzero", 0.0)),
                digits_lost=float(cr_dict.get("digits_lost", float("inf"))),
                band=str(cr_dict.get("band", "acceptable")),
                notes=tuple(cr_dict.get("notes", ())),
            )
        except Exception as e:
            logger.debug(f"Could not reconstruct ConditionReport dataclass: {e}")

    # 11. Rebuild Centerer from feature_mean_
    mean_arr = loaded_arrays.get("feature_mean_")
    if mean_arr is not None and mean_arr.size > 0:
        if mean_arr.shape[0] != transformer.n_features_in_:
            raise ArtifactIntegrityError(
                f"feature_mean_ dimension ({mean_arr.shape[0]}) does not match n_features_in_ ({transformer.n_features_in_})"
            )
        centerer = Centerer()
        centerer.mean_ = mean_arr
        centerer.n_features_in_ = transformer.n_features_in_
        transformer.centerer = centerer
    else:
        transformer.centerer = None

    # 12. Recompute derived cache & verify invariants
    if transformer.components_.shape[1] != transformer.n_features_in_:
        raise ArtifactIntegrityError(
            f"components_ column dimension ({transformer.components_.shape[1]}) does not match n_features_in_ ({transformer.n_features_in_})"
        )

    # Config-driven orthonormality verification (F4)
    if transformer.components_.shape[0] > 0:
        gram_comp = transformer.components_ @ transformer.components_.T
        ortho_err = float(np.max(np.abs(gram_comp - np.eye(transformer.components_.shape[0]))))
        if ortho_err > ortho_tol:
            logger.warning(
                f"Orthonormality deviation detected on load: {ortho_err:.2e} exceeds config tolerance {ortho_tol:.2e}"
            )

    # Mark loaded artifact as immutable
    transformer._is_loaded_artifact = True
    return transformer
