"""
Cross-Validation Stability Evaluation Module for PCA and Matrix Decompositions (Day 4).

Provides:
- CrossValidationConfig: Configuration for K-fold cross-validation and experiments.
- FoldReport: Detailed diagnostics, conditioning, rank, and held-out metrics for a single fold.
- StabilityReport: Aggregated stability evaluation, pairwise metrics, null-test baseline, and verdict.
- KFoldStabilityRunner: Deterministic, leakage-free runner orchestrating cross-validation stability.
- generate_kfold_indices: Deterministic K-fold index partition generator.
- shuffle_columns_null: Permutation-based null model breaking cross-feature covariance.

Leakage Prevention Contract:
- Preprocessing (Centerer) and decomposition (DecompositionTransformer) are fitted strictly
  on the training partition of each fold.
- Validation data is projected using the training-fold mean; validation rows never contribute
  to feature means, scaling, components, rank decisions, or fit-time diagnostics.
- Sparse inputs are sliced and processed without materializing dense (n, d) matrices.
"""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

from error import (
    InputValidation,
    ShapeMismatchError,
    NotFittedError,
    NonConvergenceError,
    InvalidFoldConfigError,
)
from stability_diagnostics import (
    StabilityConfig,
    ConditionReport,
    InputDiagnostics,
    NonFiniteInputError,
    component_cosine_stability,
    subspace_similarity,
    sparse_reconstruction_mse,
)
from decomposition import (
    DecompositionConfig,
    DecompositionTransformer,
    MethodType,
)

logger = logging.getLogger(__name__)


# =====================================================================
# JSON Serialization Utilities
# =====================================================================

def _to_json_safe(obj: Any) -> Any:
    """Recursively converts objects to strictly RFC 8259-compliant JSON representations.

    Converts:
    - np.generic (np.integer, np.floating, np.bool_) to native Python types
    - np.ndarray to nested list
    - float('inf') to "Infinity", float('-inf') to "-Infinity", float('nan') to "NaN"
    - Dataclasses to dictionaries
    - Tuples and sets to lists
    - Enums to their values
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
        return obj
    if isinstance(obj, np.generic):
        item = obj.item()
        if isinstance(item, float):
            if math.isinf(item):
                return "Infinity" if item > 0 else "-Infinity"
            if math.isnan(item):
                return "NaN"
        return item
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {f.name: _to_json_safe(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]
    return str(obj)


# =====================================================================
# Configuration Dataclass
# =====================================================================

@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for cross-validation stability evaluation.

    Fields:
        n_splits: Number of folds (must be an integer >= 2, default: 5).
        seed: Random seed for deterministic partition generation and experiments (default: 42).
        shuffle: Whether to shuffle sample indices before partitioning (default: True).
        run_null_test: Whether to execute an independent column-shuffled null baseline (default: False).
        fold_size_fractions: Optional strictly increasing fractions in (0, 1] of training data
            to evaluate stability scaling (e.g., (0.25, 0.5, 1.0)).
        min_retained_rank: Minimum rank expected for non-degenerate stability (default: 1).
        subspace_warning_threshold: Threshold below which subspace similarity is considered marginal (default: 0.90).
        subspace_critical_threshold: Threshold below which subspace similarity is considered unstable (default: 0.70).
    """
    n_splits: int = 5
    seed: Optional[int] = 42
    shuffle: bool = True
    run_null_test: bool = False
    fold_size_fractions: Optional[Tuple[float, ...]] = None
    min_retained_rank: int = 1
    subspace_warning_threshold: float = 0.90
    subspace_critical_threshold: float = 0.70

    def __post_init__(self):
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise InvalidFoldConfigError(f"n_splits must be an integer >= 2, got {self.n_splits}")

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
                raise InvalidFoldConfigError(f"seed must be a non-negative integer or None, got {self.seed}")

        if isinstance(self.min_retained_rank, bool) or not isinstance(self.min_retained_rank, int) or self.min_retained_rank < 0:
            raise InvalidFoldConfigError(f"min_retained_rank must be a non-negative integer, got {self.min_retained_rank}")

        for name, val in [
            ("subspace_warning_threshold", self.subspace_warning_threshold),
            ("subspace_critical_threshold", self.subspace_critical_threshold),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val):
                raise InvalidFoldConfigError(f"{name} must be a finite float, got {val}")

        if not (0.0 < self.subspace_critical_threshold < self.subspace_warning_threshold <= 1.0):
            raise InvalidFoldConfigError(
                f"Must satisfy 0 < subspace_critical_threshold ({self.subspace_critical_threshold}) "
                f"< subspace_warning_threshold ({self.subspace_warning_threshold}) <= 1.0"
            )

        if self.fold_size_fractions is not None:
            if not isinstance(self.fold_size_fractions, (list, tuple)):
                raise InvalidFoldConfigError("fold_size_fractions must be a tuple or list of floats in (0, 1]")

            fractions_list = []
            for f in self.fold_size_fractions:
                if not isinstance(f, (int, float)) or isinstance(f, bool) or not np.isfinite(f) or f <= 0.0 or f > 1.0:
                    raise InvalidFoldConfigError(f"Each fold_size_fraction must be in (0.0, 1.0], got {f}")
                fractions_list.append(float(f))

            if len(fractions_list) == 0:
                raise InvalidFoldConfigError("fold_size_fractions cannot be empty when specified")

            # Must be strictly increasing
            for i in range(len(fractions_list) - 1):
                if fractions_list[i] >= fractions_list[i + 1]:
                    raise InvalidFoldConfigError(
                        f"fold_size_fractions must be strictly increasing, got {fractions_list}"
                    )

            object.__setattr__(self, "fold_size_fractions", tuple(fractions_list))


# =====================================================================
# Report Dataclasses
# =====================================================================

@dataclass(frozen=True)
class FoldReport:
    """Detailed evaluation report for an individual cross-validation fold.

    Fields:
        fold: 0-indexed fold index.
        n_train: Number of training samples in this fold.
        n_val: Number of validation samples in this fold.
        effective_rank: Effective numerical rank retained after fit.
        requested_rank: Requested rank k configured for decomposition.
        val_reconstruction_mse: Held-out mean squared reconstruction error on validation fold.
        condition_report: Numerical condition diagnostics of the training fold.
        input_diagnostics: Dimensionality, density, and constant features of training fold.
        mean_component_cosine: Average absolute cosine similarity against all other folds.
        mean_subspace_similarity: Average principal angle subspace similarity against all other folds.
        pairwise_component_cosines: Tuple of cosine similarities against other folds (other_fold, similarity).
        pairwise_subspace_similarities: Tuple of subspace similarities against other folds (other_fold, similarity).
        warnings: Warnings encountered during training or evaluation of this fold.
        solver_status: Convergence status ("converged" or "non_convergence").
        notes: Contextual notes for this fold.
    """
    fold: int
    n_train: int
    n_val: int
    effective_rank: int
    requested_rank: int
    val_reconstruction_mse: float
    condition_report: ConditionReport
    input_diagnostics: InputDiagnostics
    mean_component_cosine: Optional[float] = None
    mean_subspace_similarity: Optional[float] = None
    pairwise_component_cosines: Tuple[Tuple[int, float], ...] = ()
    pairwise_subspace_similarities: Tuple[Tuple[int, float], ...] = ()
    warnings: Tuple[str, ...] = ()
    solver_status: str = "converged"
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_safe(self)


@dataclass(frozen=True)
class StabilityReport:
    """Comprehensive cross-validation stability report with schema versioning.

    Fields:
        schema_version: Report schema identifier (always "day4.v1").
        cv_config: CrossValidationConfig used for this evaluation.
        decomp_config: DecompositionConfig used for this evaluation.
        stability_config: StabilityConfig defining diagnostic thresholds.
        seed: Random seed used for reproducibility.
        fold_reports: Tuple of individual FoldReports across all folds.
        mean_val_reconstruction_mse: Mean held-out reconstruction error across folds.
        std_val_reconstruction_mse: Sample standard deviation of held-out reconstruction error.
        min_val_reconstruction_mse: Minimum held-out reconstruction error observed.
        max_val_reconstruction_mse: Maximum held-out reconstruction error observed.
        mean_subspace_similarity: Aggregate average subspace similarity across all fold pairs.
        std_subspace_similarity: Sample standard deviation of pairwise subspace similarities.
        pairwise_subspace_similarities: All pairwise subspace similarities (fold_i, fold_j, similarity).
        mean_component_cosine: Aggregate average component cosine similarity across all fold pairs.
        std_component_cosine: Sample standard deviation of pairwise component cosine similarities.
        pairwise_component_cosines: All pairwise component cosines (fold_i, fold_j, cosine).
        null_test: Optional results from the independent column-permutation null baseline.
        fold_size_results: Optional results from training-subset scaling experiments.
        warnings: System-wide or aggregation warnings.
        verdict: Final diagnostic stability assessment ("stable", "marginal", or "unstable").
        verdict_notes: Detailed explanations justifying the diagnostic verdict.
    """
    cv_config: CrossValidationConfig
    decomp_config: DecompositionConfig
    stability_config: StabilityConfig
    seed: Optional[int]
    fold_reports: Tuple[FoldReport, ...]
    mean_val_reconstruction_mse: float
    std_val_reconstruction_mse: float
    min_val_reconstruction_mse: float
    max_val_reconstruction_mse: float
    schema_version: str = "day4.v1"
    mean_subspace_similarity: Optional[float] = None
    std_subspace_similarity: Optional[float] = None
    pairwise_subspace_similarities: Tuple[Tuple[int, int, float], ...] = ()
    mean_component_cosine: Optional[float] = None
    std_component_cosine: Optional[float] = None
    pairwise_component_cosines: Tuple[Tuple[int, int, float], ...] = ()
    null_test: Optional[Dict[str, Any]] = None
    fold_size_results: Optional[Tuple[Dict[str, Any], ...]] = None
    warnings: Tuple[str, ...] = ()
    verdict: str = "stable"
    verdict_notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_safe(self)

    def to_json(self, indent: int = 2) -> str:
        """Serializes the report to a strictly compliant RFC 8259 JSON string."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


# =====================================================================
# Partitioning and Null Data Utilities
# =====================================================================

def generate_kfold_indices(
    n_samples: int,
    n_splits: int,
    shuffle: bool = True,
    seed: Optional[int] = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates deterministic, non-overlapping K-fold train/validation index partitions.

    Args:
        n_samples: Total number of rows in the dataset (integer >= 1).
        n_splits: Number of folds (integer >= 2).
        shuffle: Whether to randomly permute indices prior to splitting.
        seed: Random seed for deterministic shuffling.

    Returns:
        List of (train_indices, val_indices) 1D int64 NumPy arrays.
    """
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
        raise InvalidFoldConfigError(f"n_samples must be a positive integer, got {n_samples}")

    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise InvalidFoldConfigError(f"n_splits must be an integer >= 2, got {n_splits}")

    if n_samples < n_splits:
        raise InvalidFoldConfigError(
            f"Cannot create n_splits={n_splits} folds with only n_samples={n_samples} samples."
        )

    if shuffle:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples, dtype=np.int64)

    base_size = n_samples // n_splits
    remainder = n_samples % n_splits
    fold_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_splits)]

    splits = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current : current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size:]])
        splits.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))
        current += size

    return splits


def shuffle_columns_null(
    X: Union[np.ndarray, sp.spmatrix],
    seed: Optional[int] = None
) -> Union[np.ndarray, sp.spmatrix]:
    """Generates a null-hypothesis matrix by independently permuting each feature column.

    Mathematical Properties:
    - Exactly preserves individual feature marginal distributions, means, variances, and ranges.
    - Exactly preserves the sparsity pattern count per column.
    - Completely breaks cross-feature correlations and covariance structure (E[X_j X_k] -> E[X_j] E[X_k]).
    - Never mutates the input matrix X.
    - Sparse inputs are permuted directly in CSC format without densification.

    Args:
        X: 2D NumPy array or SciPy sparse matrix.
        seed: Optional random seed for deterministic column shuffling.

    Returns:
        New matrix of identical type and shape with shuffled feature relationships.
    """
    if not (isinstance(X, np.ndarray) or sp.issparse(X)):
        raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X)}")

    if X.ndim != 2:
        raise InputValidation(message=f"Expected 2D matrix, got ndim={X.ndim}")

    n, d = X.shape
    rng = np.random.RandomState(seed)

    if sp.issparse(X):
        # Permute non-zero entries per column in CSC format without densifying
        X_csc = X.tocsc(copy=True)
        for j in range(d):
            start = X_csc.indptr[j]
            end = X_csc.indptr[j + 1]
            nnz_j = end - start
            if nnz_j > 0:
                # Randomly choose new row positions for explicit values and permute values
                new_indices = np.sort(rng.choice(n, size=nnz_j, replace=False))
                new_data = rng.permutation(X_csc.data[start:end])
                X_csc.indices[start:end] = new_indices
                X_csc.data[start:end] = new_data

        # Return in the original sparse format (CSR or CSC)
        if sp.isspmatrix_csr(X):
            return X_csc.tocsr()
        return X_csc
    else:
        # Dense copy and column-wise permutation
        X_null = X.copy()
        for j in range(d):
            X_null[:, j] = rng.permutation(X_null[:, j])
        return X_null


# =====================================================================
# Stability Verdict Evaluator
# =====================================================================

def evaluate_stability_verdict(
    fold_reports: Sequence[FoldReport],
    mean_subspace_similarity: Optional[float],
    mean_component_cosine: Optional[float],
    stability_config: Optional[StabilityConfig] = None,
    cv_config: Optional[CrossValidationConfig] = None,
) -> Tuple[str, Tuple[str, ...]]:
    """Evaluates cross-validation diagnostic results to produce a comprehensive stability verdict.

    Classification:
    - 'unstable':
      - Any fold exhibits non-convergence or solver failure.
      - Any fold condition report is classified as 'unacceptable'.
      - Any fold collapses to effective rank 0.
      - Mean subspace similarity across folds is below subspace_critical_threshold.
    - 'marginal':
      - Any fold condition report is classified as 'warn'.
      - Any fold retains an effective rank below min_retained_rank (but > 0).
      - Effective ranks vary across folds.
      - Mean subspace similarity across folds is between critical and warning thresholds.
      - Mean subspace similarity is None due to insufficient valid fold comparisons.
    - 'stable':
      - All folds converge with acceptable conditioning, stable ranks >= min_retained_rank,
        and subspace similarity >= subspace_warning_threshold.
    """
    warn_thresh = cv_config.subspace_warning_threshold if cv_config is not None else 0.90
    crit_thresh = cv_config.subspace_critical_threshold if cv_config is not None else 0.70
    min_rank = cv_config.min_retained_rank if cv_config is not None else 1

    notes = []
    is_unstable = False

    # Solver failures
    failed_folds = [fr.fold for fr in fold_reports if fr.solver_status != "converged"]
    if failed_folds:
        notes.append(f"Solver failed or did not converge in fold(s): {failed_folds}.")
        is_unstable = True

    # Zero effective rank collapse
    ranks = [fr.effective_rank for fr in fold_reports]
    if any(r == 0 for r in ranks):
        if all(r == 0 for r in ranks):
            notes.append("All cross-validation folds collapsed to effective rank 0.")
        else:
            zero_folds = [fr.fold for fr in fold_reports if fr.effective_rank == 0]
            notes.append(f"Effective rank collapsed to 0 in fold(s): {zero_folds}.")
        is_unstable = True

    # Condition bands
    unacceptable_folds = [fr.fold for fr in fold_reports if fr.condition_report.band == "unacceptable"]
    if unacceptable_folds:
        notes.append(f"Unacceptable matrix conditioning observed in fold(s): {unacceptable_folds}.")
        is_unstable = True

    # Subspace similarity critical threshold
    if mean_subspace_similarity is not None and mean_subspace_similarity < crit_thresh:
        notes.append(
            f"Mean subspace similarity ({mean_subspace_similarity:.3f}) across folds is below critical threshold {crit_thresh:.2f}."
        )
        is_unstable = True

    if is_unstable:
        return "unstable", tuple(notes)

    is_marginal = False

    # Warning condition bands
    warn_folds = [fr.fold for fr in fold_reports if fr.condition_report.band == "warn"]
    if warn_folds:
        notes.append(f"Condition number warning threshold exceeded in fold(s): {warn_folds}.")
        is_marginal = True

    # Rank mismatch across folds
    if len(set(ranks)) > 1:
        notes.append(f"Effective ranks varied across folds: {ranks}.")
        is_marginal = True

    # Effective rank below min_retained_rank (non-zero ranks below configured minimum)
    below_min_folds = [fr.fold for fr in fold_reports if 0 < fr.effective_rank < min_rank]
    if below_min_folds:
        notes.append(
            f"Effective rank below min_retained_rank ({min_rank}) in fold(s): {below_min_folds} "
            f"with ranks {[fr.effective_rank for fr in fold_reports if 0 < fr.effective_rank < min_rank]}."
        )
        is_marginal = True

    # Subspace similarity warning threshold
    if mean_subspace_similarity is not None and crit_thresh <= mean_subspace_similarity < warn_thresh:
        notes.append(
            f"Mean subspace similarity ({mean_subspace_similarity:.3f}) across folds is marginal (< {warn_thresh:.2f})."
        )
        is_marginal = True

    # Missing / insufficient valid comparisons
    if mean_subspace_similarity is None:
        notes.append(
            "Subspace stability could not be evaluated due to insufficient valid fold comparisons."
        )
        is_marginal = True

    if is_marginal:
        return "marginal", tuple(notes)

    notes.append(
        "Cross-validation stability criteria fully satisfied with robust subspace alignment and acceptable conditioning."
    )
    return "stable", tuple(notes)


# =====================================================================
# KFoldStabilityRunner
# =====================================================================

class KFoldStabilityRunner:
    """Orchestrates leakage-free K-fold cross-validation stability evaluation.

    Guarantees:
    - Preprocessing and decomposition are fitted purely on training partitions.
    - Validation data is projected using the training-fold mean; validation rows never leak.
    - Sparse matrices are processed without global dense conversions.
    - Evaluates held-out reconstruction error, condition numbers, and rank consistency.
    - Computes pairwise component cosine stability (sign-flip invariant) and subspace similarities.
    - Optional column-shuffled null test and training-size sensitivity experiments.
    """

    def __init__(
        self,
        decomp_config: Optional[DecompositionConfig] = None,
        cv_config: Optional[CrossValidationConfig] = None,
        stability_config: Optional[StabilityConfig] = None,
        k: Optional[int] = None,
        method: Union[MethodType, str] = MethodType.from_scratch_covariance,
        n_splits: int = 5,
        seed: Optional[int] = 42,
    ):
        # Resolve decomposition configuration
        if decomp_config is not None:
            if k is not None and k != decomp_config.k:
                # Override k if explicitly provided
                self.decomp_config = DecompositionConfig(
                    method=decomp_config.method,
                    k=k,
                    seed=decomp_config.seed,
                    eig_tol=decomp_config.eig_tol,
                    eig_max_iter=decomp_config.eig_max_iter,
                    psd_clip_rtol=decomp_config.psd_clip_rtol,
                    pca_svd_solver=decomp_config.pca_svd_solver,
                    tsvd_algorithm=decomp_config.tsvd_algorithm,
                    tsvd_n_iter=decomp_config.tsvd_n_iter,
                    output_dtype=decomp_config.output_dtype,
                    max_dense_cov_dim=decomp_config.max_dense_cov_dim,
                )
            else:
                self.decomp_config = decomp_config
        else:
            self.decomp_config = DecompositionConfig(
                method=method,
                k=k if k is not None else 10,
                seed=seed,
            )

        # Resolve stability configuration
        self.stability_config = stability_config if stability_config is not None else StabilityConfig()

        # Resolve cross-validation configuration
        if cv_config is not None:
            self.cv_config = cv_config
        else:
            self.cv_config = CrossValidationConfig(
                n_splits=n_splits,
                seed=seed,
            )

    def _validate_input_matrix(self, X: Union[np.ndarray, sp.spmatrix]) -> Tuple[int, int]:
        """Validates input matrix properties and dimensions."""
        if not (isinstance(X, np.ndarray) or sp.issparse(X)):
            raise InputValidation(message=f"Expected numpy ndarray or scipy.sparse matrix, got {type(X)}")

        if X.ndim != 2:
            raise InputValidation(message=f"Expected 2D matrix, got ndim={X.ndim}")

        n, d = X.shape
        if n < 1 or d < 1:
            raise InputValidation(message=f"Matrix dimensions must be positive, got {X.shape}")

        if sp.issparse(X):
            if X.nnz > 0 and not np.isfinite(X.data).all():
                raise NonFiniteInputError(message="Sparse data contains NaN or infinite values")
        else:
            if not np.isfinite(X).all():
                raise NonFiniteInputError(message="Dense data contains NaN or infinite values")

        if n < self.cv_config.n_splits:
            raise InvalidFoldConfigError(
                f"Number of samples ({n}) is less than requested n_splits ({self.cv_config.n_splits})."
            )

        return n, d

    def run(self, X: Union[np.ndarray, sp.spmatrix]) -> StabilityReport:
        """Executes full leakage-free cross-validation stability evaluation.

        Args:
            X: 2D NumPy array or SciPy sparse matrix of shape (n_samples, n_features).

        Returns:
            StabilityReport containing all fold reports, aggregate statistics, and diagnostic verdict.
        """
        n, d = self._validate_input_matrix(X)
        k_req = self.decomp_config.k

        # -------------------------------------------------------------
        # 1. Generate Deterministic K-Fold Partitions
        # -------------------------------------------------------------
        splits = generate_kfold_indices(
            n_samples=n,
            n_splits=self.cv_config.n_splits,
            shuffle=self.cv_config.shuffle,
            seed=self.cv_config.seed
        )

        fold_reports_list: List[FoldReport] = []
        fold_components: List[np.ndarray] = []
        fold_effective_ranks: List[int] = []
        system_warnings: List[str] = []

        # -------------------------------------------------------------
        # 2. Iterate Across Folds (Leakage-Safe Pipeline)
        # -------------------------------------------------------------
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            n_train = len(train_idx)
            n_val = len(val_idx)

            if n_train <= k_req:
                raise InvalidFoldConfigError(
                    f"Fold {fold_idx}: training sample count ({n_train}) is too small for requested rank "
                    f"k={k_req} (requires n >= k + 1 = {k_req + 1})."
                )

            # Row slicing without densifying sparse inputs
            X_train = X[train_idx]
            X_val = X[val_idx]

            # Independent transformer per fold
            transformer = DecompositionTransformer(
                config=self.decomp_config,
                stability_config=self.stability_config
            )

            solver_status = "converged"
            fold_warnings: List[str] = []

            try:
                transformer.fit(X_train)
                fold_warnings.extend(transformer.fit_warnings_)
            except NonConvergenceError as e:
                solver_status = "non_convergence"
                msg = f"Fold {fold_idx}: Solver did not converge: {e}"
                logger.warning(msg)
                fold_warnings.append(msg)
                system_warnings.append(msg)

            eff_rank = transformer.effective_rank_ if transformer.effective_rank_ is not None else 0
            fold_effective_ranks.append(eff_rank)
            components = transformer.components_ if transformer.components_ is not None else np.empty((0, d))
            fold_components.append(components)

            # Held-out reconstruction error evaluated against training-centered validation data
            # sparse_reconstruction_mse processes rows in bounded batches without materializing full dense (n, d)
            val_mse = sparse_reconstruction_mse(X_val, transformer)

            fold_report = FoldReport(
                fold=fold_idx,
                n_train=n_train,
                n_val=n_val,
                effective_rank=eff_rank,
                requested_rank=k_req,
                val_reconstruction_mse=val_mse,
                condition_report=transformer.condition_report_,
                input_diagnostics=transformer.input_diagnostics_,
                warnings=tuple(fold_warnings),
                solver_status=solver_status
            )
            fold_reports_list.append(fold_report)

        # -------------------------------------------------------------
        # 3. Compute Pairwise Cross-Fold Stability Metrics
        # -------------------------------------------------------------
        n_folds = len(fold_reports_list)
        pairwise_subspace: List[Tuple[int, int, float]] = []
        pairwise_cosine: List[Tuple[int, int, float]] = []

        # Lists of (other_fold, similarity) tuples for per-fold averages
        per_fold_subspace: List[List[Tuple[int, float]]] = [[] for _ in range(n_folds)]
        per_fold_cosine: List[List[Tuple[int, float]]] = [[] for _ in range(n_folds)]

        for i in range(n_folds):
            W_i = fold_components[i]
            r_i = fold_effective_ranks[i]

            for j in range(i + 1, n_folds):
                W_j = fold_components[j]
                r_j = fold_effective_ranks[j]

                # Subspace similarity (only valid if both folds retain effective rank > 0)
                if r_i > 0 and r_j > 0:
                    s_sim = subspace_similarity(W_i, W_j)
                    pairwise_subspace.append((i, j, s_sim))
                    per_fold_subspace[i].append((j, s_sim))
                    per_fold_subspace[j].append((i, s_sim))
                else:
                    warn_msg = (
                        f"Pairwise subspace comparison omitted between fold {i} (rank {r_i}) "
                        f"and fold {j} (rank {r_j}) due to rank zero."
                    )
                    system_warnings.append(warn_msg)

                # Component cosine similarity (account for sign flips and rank mismatch)
                if r_i > 0 and r_j > 0:
                    r_common = min(r_i, r_j)
                    if r_i != r_j:
                        warn_msg = (
                            f"Rank mismatch between fold {i} (rank {r_i}) and fold {j} (rank {r_j}); "
                            f"truncated to common rank {r_common} for cosine similarity."
                        )
                        system_warnings.append(warn_msg)

                    cosines = component_cosine_stability(W_i[:r_common], W_j[:r_common])
                    mean_cos = float(np.mean(cosines))
                    pairwise_cosine.append((i, j, mean_cos))
                    per_fold_cosine[i].append((j, mean_cos))
                    per_fold_cosine[j].append((i, mean_cos))
                else:
                    if r_i != r_j:
                        warn_msg = f"Rank-0 mismatch between fold {i} (rank {r_i}) and fold {j} (rank {r_j})."
                        system_warnings.append(warn_msg)

        # Update per-fold reports with cross-fold stability averages
        updated_fold_reports: List[FoldReport] = []
        for i, fr in enumerate(fold_reports_list):
            mean_sub = float(np.mean([s for _, s in per_fold_subspace[i]])) if per_fold_subspace[i] else None
            mean_cos = float(np.mean([c for _, c in per_fold_cosine[i]])) if per_fold_cosine[i] else None

            pw_sub = tuple(sorted(per_fold_subspace[i], key=lambda x: x[0]))
            pw_cos = tuple(sorted(per_fold_cosine[i], key=lambda x: x[0]))

            updated_fold_reports.append(
                FoldReport(
                    fold=fr.fold,
                    n_train=fr.n_train,
                    n_val=fr.n_val,
                    effective_rank=fr.effective_rank,
                    requested_rank=fr.requested_rank,
                    val_reconstruction_mse=fr.val_reconstruction_mse,
                    condition_report=fr.condition_report,
                    input_diagnostics=fr.input_diagnostics,
                    mean_component_cosine=mean_cos,
                    mean_subspace_similarity=mean_sub,
                    pairwise_component_cosines=pw_cos,
                    pairwise_subspace_similarities=pw_sub,
                    warnings=fr.warnings,
                    solver_status=fr.solver_status,
                    notes=fr.notes
                )
            )

        # -------------------------------------------------------------
        # 4. Compute Aggregate Statistics
        # -------------------------------------------------------------
        val_mses = [fr.val_reconstruction_mse for fr in updated_fold_reports]
        mean_val_mse = float(np.mean(val_mses))
        std_val_mse = float(np.std(val_mses, ddof=1)) if len(val_mses) > 1 else 0.0
        min_val_mse = float(np.min(val_mses))
        max_val_mse = float(np.max(val_mses))

        sub_vals = [s for _, _, s in pairwise_subspace]
        mean_subspace = float(np.mean(sub_vals)) if sub_vals else None
        std_subspace = float(np.std(sub_vals, ddof=1)) if len(sub_vals) > 1 else (0.0 if sub_vals else None)

        cos_vals = [c for _, _, c in pairwise_cosine]
        mean_cosine = float(np.mean(cos_vals)) if cos_vals else None
        std_cosine = float(np.std(cos_vals, ddof=1)) if len(cos_vals) > 1 else (0.0 if cos_vals else None)

        # -------------------------------------------------------------
        # 5. Optional Null Test (Independent Column Permutation)
        # -------------------------------------------------------------
        null_test_results = None
        if self.cv_config.run_null_test:
            # Deterministic seed stream derived from cv_config.seed
            null_seed = (self.cv_config.seed + 10007) % (2**31 - 1) if self.cv_config.seed is not None else 12345
            X_null = shuffle_columns_null(X, seed=null_seed)

            # Execute cross-validation on null data with run_null_test=False to prevent recursion
            null_runner = KFoldStabilityRunner(
                decomp_config=self.decomp_config,
                cv_config=CrossValidationConfig(
                    n_splits=self.cv_config.n_splits,
                    seed=self.cv_config.seed,
                    shuffle=self.cv_config.shuffle,
                    run_null_test=False
                ),
                stability_config=self.stability_config
            )
            null_report = null_runner.run(X_null)

            subspace_delta = (
                float(mean_subspace - null_report.mean_subspace_similarity)
                if (mean_subspace is not None and null_report.mean_subspace_similarity is not None)
                else None
            )
            mse_ratio = (
                float(mean_val_mse / null_report.mean_val_reconstruction_mse)
                if null_report.mean_val_reconstruction_mse > 0
                else None
            )

            null_test_results = {
                "procedure": (
                    "Independent column permutation breaking feature covariance while preserving "
                    "marginal distributions, individual variances, and matrix sparsity."
                ),
                "null_seed": null_seed,
                "null_mean_val_reconstruction_mse": null_report.mean_val_reconstruction_mse,
                "null_mean_subspace_similarity": null_report.mean_subspace_similarity,
                "null_mean_component_cosine": null_report.mean_component_cosine,
                "subspace_similarity_delta": subspace_delta,
                "reconstruction_mse_ratio": mse_ratio,
                "interpretation": (
                    "A positive subspace_similarity_delta and reconstruction_mse_ratio < 1.0 indicate "
                    "that real components capture genuine multi-feature covariance rather than noise."
                )
            }

        # -------------------------------------------------------------
        # 6. Optional Stability-vs-Fold-Size Experiment
        # -------------------------------------------------------------
        fold_size_results_list = None
        if self.cv_config.fold_size_fractions is not None:
            fold_size_results_list = []
            for frac in self.cv_config.fold_size_fractions:
                frac_mses = []
                frac_components = []
                frac_ranks = []

                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    n_subset = int(np.floor(len(train_idx) * frac))
                    if n_subset < 2:
                        raise InvalidFoldConfigError(
                            f"Training subset size ({n_subset}) for fraction {frac} is less than 2."
                        )
                    if n_subset <= k_req:
                        raise InvalidFoldConfigError(
                            f"Training subset size ({n_subset}) for fraction {frac} is too small for requested rank "
                            f"k={k_req} (requires n >= k + 1 = {k_req + 1})."
                        )

                    # Strictly nested subset selection
                    subset_train_idx = train_idx[:n_subset]
                    X_train_sub = X[subset_train_idx]
                    X_val_full = X[val_idx]

                    sub_trans = DecompositionTransformer(
                        config=self.decomp_config,
                        stability_config=self.stability_config
                    )
                    sub_trans.fit(X_train_sub)
                    frac_mses.append(sparse_reconstruction_mse(X_val_full, sub_trans))
                    frac_components.append(sub_trans.components_ if sub_trans.components_ is not None else np.empty((0, d)))
                    frac_ranks.append(sub_trans.effective_rank_ if sub_trans.effective_rank_ is not None else 0)

                # Pairwise subspace similarities for this fraction (positive ranks only)
                frac_subspaces = []
                for fi in range(len(splits)):
                    for fj in range(fi + 1, len(splits)):
                        if frac_ranks[fi] > 0 and frac_ranks[fj] > 0:
                            frac_subspaces.append(subspace_similarity(frac_components[fi], frac_components[fj]))

                fold_size_results_list.append({
                    "fraction": frac,
                    "effective_sample_size": int(np.floor(len(splits[0][0]) * frac)),
                    "mean_effective_rank": float(np.mean(frac_ranks)),
                    "mean_val_reconstruction_mse": float(np.mean(frac_mses)),
                    "mean_subspace_similarity": float(np.mean(frac_subspaces)) if frac_subspaces else None,
                })

        # -------------------------------------------------------------
        # 7. Final Diagnostic Stability Verdict
        # -------------------------------------------------------------
        verdict, verdict_notes = evaluate_stability_verdict(
            fold_reports=updated_fold_reports,
            mean_subspace_similarity=mean_subspace,
            mean_component_cosine=mean_cosine,
            stability_config=self.stability_config,
            cv_config=self.cv_config
        )

        return StabilityReport(
            schema_version="day4.v1",
            cv_config=self.cv_config,
            decomp_config=self.decomp_config,
            stability_config=self.stability_config,
            seed=self.cv_config.seed,
            fold_reports=tuple(updated_fold_reports),
            mean_val_reconstruction_mse=mean_val_mse,
            std_val_reconstruction_mse=std_val_mse,
            min_val_reconstruction_mse=min_val_mse,
            max_val_reconstruction_mse=max_val_mse,
            mean_subspace_similarity=mean_subspace,
            std_subspace_similarity=std_subspace,
            pairwise_subspace_similarities=tuple(pairwise_subspace),
            mean_component_cosine=mean_cosine,
            std_component_cosine=std_cosine,
            pairwise_component_cosines=tuple(pairwise_cosine),
            null_test=null_test_results,
            fold_size_results=tuple(fold_size_results_list) if fold_size_results_list is not None else None,
            warnings=tuple(system_warnings),
            verdict=verdict,
            verdict_notes=verdict_notes
        )


__all__ = [
    "CrossValidationConfig",
    "FoldReport",
    "StabilityReport",
    "KFoldStabilityRunner",
    "generate_kfold_indices",
    "shuffle_columns_null",
    "evaluate_stability_verdict",
    "InvalidFoldConfigError",
]
