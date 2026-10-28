"""
Day 5 Error Analysis Module for SpectraCompress.

Provides:
- SliceDiagnosisConfig: Configuration thresholds for root-cause slice gap diagnosis.
- per_row_errors: Absolute squared Euclidean error and relative normalized error per row.
- make_slice_labels: Leak-free train-only slice assignment for categorical and numeric features.
- slice_reconstruction_error: Stratified slice-level evaluation with bootstrap confidence intervals for both absolute and relative error.
- worst_n_rows: Diagnostic inspection of the worst reconstructed rows sorted by e or r, with feature errors and base-rate caveat.
- diagnose_slice_gap: Root-cause diagnosis classifying slice gaps as no_gap, scale_effect, outlier_driven, or systematic.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from error import (
    SpectraCompressError,
    InputValidation,
    ShapeMismatchError,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration Dataclass
# =====================================================================

@dataclass(frozen=True)
class SliceDiagnosisConfig:
    """Thresholds for slice-level reconstruction gap diagnosis.

    Fields:
        ratio_e_threshold: Upper threshold on ratio_e (slice mean_e / overall mean_e).
        ratio_r_high: Upper threshold on ratio_r (slice mean_r / overall mean_r).
        ratio_r_low: Lower threshold on ratio_r for normal variance band.
        min_size: Minimum slice size for high confidence.
    """
    ratio_e_threshold: float = 1.15
    ratio_r_high: float = 1.15
    ratio_r_low: float = 0.87
    min_size: int = 30


# =====================================================================
# Row-Level Error Metric
# =====================================================================

def per_row_errors(
    X: np.ndarray,
    X_hat: np.ndarray,
    mu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes absolute squared error and relative normalized error per row.

    Mathematical Definition:
        e_i = ||x_i - x_hat_i||_2^2 = sum_j (x_ij - x_hat_ij)^2
        r_i = e_i / ||x_i - mu||_2^2

    Guards:
        Guards against zero / near-zero denominator (||x_i - mu||_2^2 <= 1e-12)
        by assigning r_i = 0.0, avoiding NaN or Inf.

    Args:
        X: Original 2D array of shape (n_samples, n_features).
        X_hat: Reconstructed 2D array of shape (n_samples, n_features).
        mu: 1D training mean vector of shape (n_features,).

    Returns:
        (e, r): Tuple of two 1D float64 arrays, each of shape (n_samples,).
    """
    if not isinstance(X, np.ndarray) or not isinstance(X_hat, np.ndarray) or not isinstance(mu, np.ndarray):
        raise InputValidation(message="X, X_hat, and mu must be numpy ndarrays")

    if X.ndim != 2 or X_hat.ndim != 2:
        raise InputValidation(message=f"X and X_hat must be 2D arrays. Got {X.ndim} and {X_hat.ndim}")

    if X.shape != X_hat.shape:
        raise ShapeMismatchError(message=f"X and X_hat shapes must match: {X.shape} vs {X_hat.shape}")

    n, d = X.shape
    if mu.ndim != 1 or mu.shape[0] != d:
        raise ShapeMismatchError(message=f"mu must be 1D of length d={d}, got shape {mu.shape}")

    if not np.isfinite(X).all() or not np.isfinite(X_hat).all() or not np.isfinite(mu).all():
        raise InputValidation(message="Inputs contain non-finite values (NaN or Inf)")

    # Absolute squared Euclidean reconstruction error per row
    e = np.sum((X - X_hat) ** 2, axis=1, dtype=np.float64)

    # Squared distance from training mean per row
    denom = np.sum((X - mu) ** 2, axis=1, dtype=np.float64)

    # Guard against ~0 denominator
    r = np.zeros_like(e, dtype=np.float64)
    nonzero_mask = denom > 1e-12
    r[nonzero_mask] = e[nonzero_mask] / denom[nonzero_mask]

    return e, r


# =====================================================================
# Slicing Labels
# =====================================================================

def make_slice_labels(
    col_train: Union[np.ndarray, pd.Series, Sequence[Any]],
    col_val: Union[np.ndarray, pd.Series, Sequence[Any]],
    quantiles: Union[int, Sequence[float]] = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates slice labels using train-only statistics.

    Policy:
        - If numeric: Computes quantile bin edges strictly on col_train. Applies
          these edges to col_val. Values falling outside the train range or NaN
          are mapped to '__unseen__'.
        - If categorical / string / object: Categories observed in col_train are
          preserved. Any category appearing in col_val that was not in col_train
          is labeled '__unseen__'.

    Args:
        col_train: Slicing column from training data.
        col_val: Slicing column from validation data.
        quantiles: Number of quantile buckets (e.g. 4 for quartiles) or sequence
            of quantile cutoffs in [0, 1].

    Returns:
        (labels_train, labels_val): Tuple of 1D string ndarrays.
    """
    arr_train = np.asarray(col_train)
    arr_val = np.asarray(col_val)

    if arr_train.ndim != 1 or arr_val.ndim != 1:
        raise InputValidation(message="Slicing columns must be 1D arrays or series")

    if len(arr_train) == 0:
        raise InputValidation(message="col_train cannot be empty")

    is_numeric = np.issubdtype(arr_train.dtype, np.number) and not isinstance(arr_train[0], (bool, np.bool_))

    if is_numeric:
        if not np.isfinite(arr_train).all():
            raise InputValidation(message="col_train contains NaN or Inf values")

        if isinstance(quantiles, int):
            if quantiles < 1:
                raise InputValidation(message="quantiles must be >= 1")
            q_probs = np.linspace(0.0, 1.0, quantiles + 1)
        else:
            q_probs = np.asarray(quantiles, dtype=np.float64)
            if not np.all((q_probs >= 0.0) & (q_probs <= 1.0)):
                raise InputValidation(message="Quantile cutoffs must be in [0, 1]")
            if not np.all(np.diff(q_probs) > 0):
                raise InputValidation(message="Quantile cutoffs must be strictly increasing")

        edges = np.quantile(arr_train, q_probs)
        edges = np.unique(edges)
        if len(edges) < 2:
            labels_train = np.full(len(arr_train), "Q0_constant", dtype=object)
            labels_val = np.where(arr_val == edges[0], "Q0_constant", "__unseen__")
            return labels_train.astype(str), labels_val.astype(str)

        n_bins = len(edges) - 1
        bin_names = [f"Q{i+1}_[{edges[i]:.2f}, {edges[i+1]:.2f}]" for i in range(n_bins)]

        train_idx = np.digitize(arr_train, edges[:-1], right=False) - 1
        train_idx = np.clip(train_idx, 0, n_bins - 1)
        labels_train = np.array([bin_names[i] for i in train_idx], dtype=object)

        val_labels_list = []
        for val in arr_val:
            if not np.isfinite(val) or val < edges[0] or val > edges[-1]:
                val_labels_list.append("__unseen__")
            else:
                idx = np.digitize([val], edges[:-1], right=False)[0] - 1
                idx = np.clip(idx, 0, n_bins - 1)
                val_labels_list.append(bin_names[idx])
        labels_val = np.array(val_labels_list, dtype=object)

        return labels_train.astype(str), labels_val.astype(str)

    else:
        train_cats = set(arr_train.tolist())
        labels_train = np.array([str(x) for x in arr_train], dtype=str)

        labels_val_list = []
        for x in arr_val:
            if x in train_cats:
                labels_val_list.append(str(x))
            else:
                labels_val_list.append("__unseen__")
        labels_val = np.array(labels_val_list, dtype=str)

        return labels_train, labels_val


# =====================================================================
# Stratified Slice Evaluation
# =====================================================================

def slice_reconstruction_error(
    X: np.ndarray,
    X_hat: np.ndarray,
    slice_labels: Sequence[str],
    *,
    mu: np.ndarray,
    min_size: int = 30,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluates reconstruction performance across slices with bootstrap CIs for both e and r.

    Columns:
        [slice, n, mean_e, median_e, trimmed_mean_e, mean_r, ci_low, ci_high,
         mean_r_ci_low, mean_r_ci_high, ratio_e, ratio_r, ratio_r_after_trim, low_confidence]

    Args:
        X: Original 2D array.
        X_hat: Reconstructed 2D array.
        slice_labels: 1D sequence of slice names corresponding to rows in X.
        mu: 1D training mean vector.
        min_size: Sample size threshold below which low_confidence is True.
        n_boot: Number of bootstrap iterations for confidence intervals.
        seed: Random seed for deterministic bootstrap sampling.

    Returns:
        pd.DataFrame: Table summarizing slice-level reconstruction metrics.
    """
    labels = np.asarray(slice_labels, dtype=str)
    if len(labels) != len(X):
        raise ShapeMismatchError(message=f"slice_labels length {len(labels)} does not match X rows {len(X)}")

    e, r = per_row_errors(X, X_hat, mu)
    overall_mean_e = float(np.mean(e))
    overall_mean_r = float(np.mean(r))

    unique_slices = sorted(list(set(labels)))
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for s_name in unique_slices:
        mask = labels == s_name
        n_slice = int(np.sum(mask))

        if n_slice == 0:
            rows.append({
                "slice": s_name,
                "n": 0,
                "mean_e": np.nan,
                "median_e": np.nan,
                "trimmed_mean_e": np.nan,
                "mean_r": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "mean_r_ci_low": np.nan,
                "mean_r_ci_high": np.nan,
                "ratio_e": np.nan,
                "ratio_r": np.nan,
                "ratio_r_after_trim": np.nan,
                "low_confidence": True,
            })
            continue

        e_s = e[mask]
        r_s = r[mask]

        mean_e = float(np.mean(e_s))
        median_e = float(np.median(e_s))
        trimmed_e = float(stats.trim_mean(e_s, proportiontocut=0.05)) if n_slice >= 10 else mean_e
        mean_r = float(np.mean(r_s))
        low_confidence = bool(n_slice < min_size)

        # Bootstrap 95% confidence intervals for mean_e and mean_r
        if n_slice >= 2:
            boot_means_e = np.empty(n_boot, dtype=np.float64)
            boot_means_r = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                idx_b = rng.choice(n_slice, size=n_slice, replace=True)
                boot_means_e[b] = np.mean(e_s[idx_b])
                boot_means_r[b] = np.mean(r_s[idx_b])
            ci_low = float(np.percentile(boot_means_e, 2.5))
            ci_high = float(np.percentile(boot_means_e, 97.5))
            mean_r_ci_low = float(np.percentile(boot_means_r, 2.5))
            mean_r_ci_high = float(np.percentile(boot_means_r, 97.5))
        else:
            ci_low = mean_e
            ci_high = mean_e
            mean_r_ci_low = mean_r
            mean_r_ci_high = mean_r

        # Ratios against overall dataset metrics
        ratio_e = float(mean_e / (overall_mean_e + 1e-12))
        ratio_r = float(mean_r / (overall_mean_r + 1e-12))

        # Relative error after removing top 1% worst rows by error within slice
        if n_slice >= 10:
            n_cut = max(1, int(np.ceil(0.01 * n_slice)))
            keep_idx = np.argsort(e_s)[:-n_cut]
            mean_r_trimmed = float(np.mean(r_s[keep_idx]))
            ratio_r_after_trim = float(mean_r_trimmed / (overall_mean_r + 1e-12))
        else:
            ratio_r_after_trim = ratio_r

        rows.append({
            "slice": s_name,
            "n": n_slice,
            "mean_e": mean_e,
            "median_e": median_e,
            "trimmed_mean_e": trimmed_e,
            "mean_r": mean_r,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_r_ci_low": mean_r_ci_low,
            "mean_r_ci_high": mean_r_ci_high,
            "ratio_e": ratio_e,
            "ratio_r": ratio_r,
            "ratio_r_after_trim": ratio_r_after_trim,
            "low_confidence": low_confidence,
        })

    return pd.DataFrame(rows)


# =====================================================================
# Worst-N Rows Diagnostic
# =====================================================================

def worst_n_rows(
    X: np.ndarray,
    X_hat: np.ndarray,
    slice_labels: Sequence[str],
    n: int = 20,
    mu: Optional[np.ndarray] = None,
    sort_by: str = "e",
) -> pd.DataFrame:
    """Extracts the top-N worst reconstructed rows and dominant feature errors.

    Columns:
        [row_index, e, r, norm_x_minus_mu, slice, top_features, slice_share_top_n, caveat]

    Args:
        X: Original 2D array.
        X_hat: Reconstructed 2D array.
        slice_labels: Sequence of slice identifiers for each row.
        n: Number of worst rows to return (default: 20).
        mu: 1D training mean vector.
        sort_by: Ranking metric ('e' for absolute error, 'r' for relative error).

    Returns:
        pd.DataFrame: Table with the top-N worst rows and error contributions.
    """
    labels = np.asarray(slice_labels, dtype=str)
    if mu is None:
        mu = np.mean(X, axis=0)

    e, r = per_row_errors(X, X_hat, mu)
    norm_mu = np.sqrt(np.sum((X - mu) ** 2, axis=1, dtype=np.float64))

    n_samples = len(X)
    n_worst = min(n, n_samples)
    if n_worst < 1:
        raise InputValidation(message="n must be >= 1")

    # Select ranking metric
    if sort_by == "e":
        score = e
    elif sort_by == "r":
        score = r
    else:
        raise InputValidation(message=f"sort_by must be 'e' or 'r', got '{sort_by}'")

    worst_idx = np.argsort(score)[-n_worst:][::-1]

    top_slices = labels[worst_idx]
    slice_counts = collections.Counter(top_slices)
    slice_shares = {s: count / n_worst for s, count in slice_counts.items()}
    caveat_text = "N is small; compare against the base rate (about 10% per slice) and do not read this as proof."

    rows: List[Dict[str, Any]] = []
    for idx in worst_idx:
        row_e = float(e[idx])
        row_r = float(r[idx])
        row_norm = float(norm_mu[idx])
        s_val = str(labels[idx])
        s_share = float(slice_shares[s_val])

        feat_errs = (X[idx] - X_hat[idx]) ** 2
        top_3_feat_idx = np.argsort(feat_errs)[-3:][::-1]
        top_features = [
            {"feature_idx": int(f_idx), "sq_error": float(feat_errs[f_idx])}
            for f_idx in top_3_feat_idx
        ]

        rows.append({
            "row_index": int(idx),
            "e": row_e,
            "r": row_r,
            "norm_x_minus_mu": row_norm,
            "slice": s_val,
            "top_features": top_features,
            "slice_share_top_n": s_share,
            "caveat": caveat_text,
        })

    return pd.DataFrame(rows)


# =====================================================================
# Root-Cause Slice Gap Diagnosis
# =====================================================================

def diagnose_slice_gap(
    slice_summary: pd.DataFrame,
    X: np.ndarray,
    X_hat: np.ndarray,
    slice_labels: Sequence[str],
    mu: np.ndarray,
    config: Optional[SliceDiagnosisConfig] = None,
) -> Dict[str, str]:
    """Diagnoses the underlying mechanism of reconstruction gaps for each slice.

    Evaluation Order (R2 Specification):
        1. no_gap: ratio_e < ratio_e_threshold (1.15) and ratio_r within [ratio_r_low, ratio_r_high],
           or both ratio_e and ratio_r are below thresholds (not elevated).
        2. scale_effect: ratio_e >= ratio_e_threshold but ratio_r < ratio_r_high,
           OR ratio_r's CI includes overall mean_r (mean_r_ci_low <= overall_mean_r <= mean_r_ci_high).
        3. outlier_driven: ratio_r >= ratio_r_high but the gap vanishes (ratio_r_after_trim < ratio_r_high)
           after removing the top 1% of rows.
        4. systematic: ratio_r >= ratio_r_high, CI low bound of mean_r > overall mean_r,
           AND the gap survives top-1% removal (ratio_r_after_trim >= ratio_r_high).
        - Small slices (low_confidence=True) get their label plus a caveat flag,
          and are never labelled 'systematic' on their own.

    Args:
        slice_summary: DataFrame produced by slice_reconstruction_error.
        X: Original 2D array.
        X_hat: Reconstructed 2D array.
        slice_labels: Sequence of slice identifiers for each row.
        mu: 1D training mean vector.
        config: SliceDiagnosisConfig with threshold parameters.

    Returns:
        Dict[str, str]: Mapping from slice_name to diagnosis category.
    """
    if config is None:
        config = SliceDiagnosisConfig()

    e, r = per_row_errors(X, X_hat, mu)
    overall_mean_r = float(np.mean(r))

    diagnoses: Dict[str, str] = {}

    for _, row in slice_summary.iterrows():
        s_name = str(row["slice"])
        n_slice = int(row["n"])

        if n_slice == 0 or np.isnan(row["mean_e"]):
            diagnoses[s_name] = "empty_slice"
            continue

        ratio_e = float(row["ratio_e"])
        ratio_r = float(row["ratio_r"])
        mean_r_ci_low = float(row["mean_r_ci_low"])
        mean_r_ci_high = float(row["mean_r_ci_high"])
        ratio_r_after_trim = float(row["ratio_r_after_trim"])
        is_low_confidence = bool(row["low_confidence"])
        ci_includes_overall = bool(mean_r_ci_low <= overall_mean_r <= mean_r_ci_high)

        # 1. no_gap
        # Ratio e is not elevated, and ratio r is within balanced band or below
        if ratio_e < config.ratio_e_threshold and ratio_r <= config.ratio_r_high:
            diag = "no_gap"

        # 2. scale_effect (pure magnitude scaling)
        # Absolute error is elevated, but relative error is NOT elevated
        elif ratio_e >= config.ratio_e_threshold and ratio_r < config.ratio_r_high:
            diag = "scale_effect"

        # 3. outlier_driven
        # Relative error is elevated, but the gap vanishes after removing the top 1% worst rows
        elif ratio_r >= config.ratio_r_high and ratio_r_after_trim < config.ratio_r_high:
            diag = "outlier_driven"

        # 4. systematic
        # Relative error is elevated, CI lower bound strictly exceeds overall mean, AND gap survives trim
        elif ratio_r >= config.ratio_r_high and mean_r_ci_low > overall_mean_r and ratio_r_after_trim >= config.ratio_r_high:
            if is_low_confidence:
                diag = "low_confidence_caveat"
            else:
                diag = "systematic"

        # 5. scale_effect (CI includes overall mean_r)
        elif ci_includes_overall:
            diag = "scale_effect"

        else:
            diag = "no_gap"

        # Append caveat note if low confidence
        if is_low_confidence and diag != "low_confidence_caveat":
            diag = f"{diag} (low_confidence_caveat)"

        diagnoses[s_name] = diag

    return diagnoses
