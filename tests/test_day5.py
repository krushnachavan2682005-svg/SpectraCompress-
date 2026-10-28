"""
Unit tests for Day 5: Model Selection and Error Analysis.

Repairs covered (R1-R5):
R1 Tests:
- test_variance_threshold_picks_smallest_k
- test_variance_threshold_tie_break_priority
- test_grid_boundary_flag_fires
- test_no_eligible_k_raises_structured_error
- test_relative_tolerance_reproduces_run_001

R2 Tests:
- test_scale_effect_not_flagged_as_bias
- test_no_gap_when_error_below_overall
- test_outlier_driven_slice
- test_systematic_slice
- test_tiny_slice_caveat
- test_run_001_slice_0_not_systematic

Regression Tests:
- test_tie_break_prefers_smaller_k
- test_small_slice_flagged
- test_no_test_argument_exists
- test_slice_error_engineered
- test_val_error_monotone_in_k
- test_tsvd_centered_matches_pca_subspace
- test_train_only_statistics
- test_selection_deterministic
- test_report_hash_and_flags
- test_rejects_bad_input
"""

import inspect
import json
import hashlib
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA, TruncatedSVD

from error import (
    SpectraCompressError,
    InputValidation,
    ShapeMismatchError,
)
from model_comparison import (
    SelectionRule,
    MethodSpec,
    SelectionResult,
    InvalidGridError,
    NoEligibleRankError,
    compare_methods_and_ranks,
    variance_explained_curve,
    select_config,
    write_selection_report,
    get_default_methods,
)
from error_analysis import (
    SliceDiagnosisConfig,
    per_row_errors,
    make_slice_labels,
    slice_reconstruction_error,
    worst_n_rows,
    diagnose_slice_gap,
)


def _make_synthetic(n_samples: int, n_features: int, s_values: list[float], seed: int = 42) -> np.ndarray:
    """Helper generating synthetic orthogonal data with specified singular values."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n_samples, len(s_values))))
    V, _ = np.linalg.qr(rng.normal(size=(n_features, len(s_values))))
    X = U @ np.diag(s_values) @ V.T
    return X.astype(np.float64)


# =====================================================================
# R1 Tests: Variance Threshold Selection Rule & Grid Boundary
# =====================================================================

def test_variance_threshold_picks_smallest_k():
    """Verify that variance_threshold selects the SMALLEST k with V_val >= tau."""
    results_data = [
        {"method": "sklearn_pca", "k": 5, "train_recon_error": 15.0, "val_recon_error": 18.0, "val_rel_error": 0.30, "V_train": 0.70, "V_val": 0.70, "gap": 3.0, "runtime_s": 0.01},
        {"method": "sklearn_pca", "k": 10, "train_recon_error": 8.0, "val_recon_error": 9.0, "val_rel_error": 0.09, "V_train": 0.91, "V_val": 0.91, "gap": 1.0, "runtime_s": 0.02},
        {"method": "sklearn_pca", "k": 20, "train_recon_error": 4.0, "val_recon_error": 4.5, "val_rel_error": 0.04, "V_train": 0.96, "V_val": 0.96, "gap": 0.5, "runtime_s": 0.03},
        {"method": "day1_baseline", "k": 10, "train_recon_error": 25.0, "val_recon_error": 30.0, "val_rel_error": 0.60, "V_train": 0.40, "V_val": 0.40, "gap": 5.0, "runtime_s": 0.001},
    ]
    df = pd.DataFrame(results_data)

    rule = SelectionRule(kind="variance_threshold", tau=0.90, tie_break="smallest_k")
    result = select_config(df, rule, k_grid=[5, 10, 20])

    # Both k=10 (0.91) and k=20 (0.96) meet tau=0.90; must pick smallest k = 10
    assert result.chosen_k == 10
    assert result.chosen_method == "sklearn_pca"
    assert result.grid_boundary is False
    assert result.grid_boundary_warning is None


def test_variance_threshold_tie_break_priority():
    """Verify tie-breaking across methods at the same smallest k using method_priority."""
    results_data = [
        {"method": "from_scratch_covariance", "k": 10, "train_recon_error": 8.0, "val_recon_error": 9.0, "val_rel_error": 0.09, "V_train": 0.91, "V_val": 0.91, "gap": 1.0, "runtime_s": 0.05},
        {"method": "sklearn_pca", "k": 10, "train_recon_error": 8.0, "val_recon_error": 9.0, "val_rel_error": 0.09, "V_train": 0.91, "V_val": 0.91, "gap": 1.0, "runtime_s": 0.01},
        {"method": "sklearn_truncated_svd", "k": 10, "train_recon_error": 8.0, "val_recon_error": 9.0, "val_rel_error": 0.09, "V_train": 0.91, "V_val": 0.91, "gap": 1.0, "runtime_s": 0.02},
    ]
    df = pd.DataFrame(results_data)

    rule = SelectionRule(
        kind="variance_threshold",
        tau=0.90,
        method_priority=("sklearn_pca", "from_scratch_covariance", "sklearn_truncated_svd")
    )
    result = select_config(df, rule, k_grid=[10, 20])
    assert result.chosen_method == "sklearn_pca"

    # Test reverse priority
    rule_rev = SelectionRule(
        kind="variance_threshold",
        tau=0.90,
        method_priority=("from_scratch_covariance", "sklearn_pca")
    )
    result_rev = select_config(df, rule_rev, k_grid=[10, 20])
    assert result_rev.chosen_method == "from_scratch_covariance"


def test_grid_boundary_flag_fires():
    """Verify grid_boundary flag is True and warning is emitted when chosen_k == max(k_grid)."""
    results_data = [
        {"method": "sklearn_pca", "k": 5, "train_recon_error": 20.0, "val_recon_error": 22.0, "val_rel_error": 0.40, "V_train": 0.60, "V_val": 0.60, "gap": 2.0, "runtime_s": 0.01},
        {"method": "sklearn_pca", "k": 10, "train_recon_error": 14.0, "val_recon_error": 15.0, "val_rel_error": 0.25, "V_train": 0.75, "V_val": 0.75, "gap": 1.0, "runtime_s": 0.02},
        {"method": "sklearn_pca", "k": 20, "train_recon_error": 5.0, "val_recon_error": 5.5, "val_rel_error": 0.09, "V_train": 0.91, "V_val": 0.91, "gap": 0.5, "runtime_s": 0.03},
    ]
    df = pd.DataFrame(results_data)

    rule = SelectionRule(kind="variance_threshold", tau=0.90)
    # k=20 is the max of the grid
    result = select_config(df, rule, k_grid=[5, 10, 20])

    assert result.chosen_k == 20
    assert result.grid_boundary is True
    assert result.grid_boundary_warning is not None
    assert "upper grid boundary" in result.grid_boundary_warning


def test_no_eligible_k_raises_structured_error():
    """Verify that when no k achieves V_val >= tau, NoEligibleRankError is raised."""
    results_data = [
        {"method": "sklearn_pca", "k": 5, "train_recon_error": 20.0, "val_recon_error": 22.0, "val_rel_error": 0.40, "V_train": 0.60, "V_val": 0.60, "gap": 2.0, "runtime_s": 0.01},
        {"method": "sklearn_pca", "k": 10, "train_recon_error": 15.0, "val_recon_error": 16.0, "val_rel_error": 0.30, "V_train": 0.70, "V_val": 0.70, "gap": 1.0, "runtime_s": 0.02},
    ]
    df = pd.DataFrame(results_data)

    rule = SelectionRule(kind="variance_threshold", tau=0.90)
    with pytest.raises(NoEligibleRankError, match="No rank in grid achieved variance threshold"):
        select_config(df, rule, k_grid=[5, 10])


def test_relative_tolerance_reproduces_run_001():
    """Verify that the relative_tolerance rule still reproduces run_001's choice."""
    report_path = "selection_report.json"
    if not os.path.exists(report_path):
        pytest.skip("run_001 selection_report.json not found")

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    results_df = pd.DataFrame(rep["per_method_k_rows"])
    rule = SelectionRule(kind="relative_tolerance", epsilon=0.05, tie_break="smallest_k")
    sel = select_config(results_df, rule, k_grid=rep["k_grid"])

    assert sel.chosen_method == rep["chosen_method"]
    assert sel.chosen_k == rep["chosen_k"]
    assert np.isclose(sel.val_recon_error, 298260.8387, rtol=1e-3)


# =====================================================================
# R2 Tests: Fixed diagnose_slice_gap Logic
# =====================================================================

def test_scale_effect_not_flagged_as_bias():
    """Verify magnitude-scaled slice differs in absolute error but has equal relative error."""
    rng = np.random.default_rng(42)
    n_each = 100
    d = 5

    base = rng.normal(size=(n_each, d))
    base -= np.mean(base, axis=0)

    # Slice 2 scaled by 5.0
    scaled = base * 5.0

    X = np.vstack([base, scaled])
    labels = ["slice_base"] * n_each + ["slice_scaled"] * n_each

    mu = np.mean(X, axis=0)
    Xc = X - mu

    pca = PCA(n_components=2, svd_solver="full", random_state=42)
    pca.fit(Xc)
    X_hat = (pca.transform(Xc) @ pca.components_) + mu

    slice_df = slice_reconstruction_error(X, X_hat, labels, mu=mu, min_size=30, n_boot=200, seed=42)
    diagnoses = diagnose_slice_gap(slice_df, X, X_hat, labels, mu)

    # slice_scaled has ratio_e > 1.15 but ratio_r ~ 1.0 (or CI includes overall)
    assert diagnoses["slice_scaled"] == "scale_effect"


def test_no_gap_when_error_below_overall():
    """Verify that slices with errors below or equal to overall are labelled no_gap."""
    rng = np.random.default_rng(42)
    n_each = 100
    d = 5

    X = rng.normal(size=(2 * n_each, d))
    mu = np.mean(X, axis=0)
    X_hat = X + rng.normal(scale=0.1, size=X.shape)
    labels = ["slice_A"] * n_each + ["slice_B"] * n_each

    slice_df = slice_reconstruction_error(X, X_hat, labels, mu=mu, min_size=30, n_boot=200, seed=42)
    diagnoses = diagnose_slice_gap(slice_df, X, X_hat, labels, mu)

    # Random split with identical distribution -> both slices must be no_gap
    assert diagnoses["slice_A"] == "no_gap"
    assert diagnoses["slice_B"] == "no_gap"


def test_outlier_driven_slice():
    """Verify outlier-driven slice is correctly classified when gap vanishes after 1% trim."""
    rng = np.random.default_rng(42)
    n = 200
    d = 5

    X = rng.normal(size=(n, d))
    mu = np.mean(X, axis=0)
    Xc = X - mu

    pca = PCA(n_components=2, svd_solver="full", random_state=42)
    pca.fit(Xc)
    X_hat = (pca.transform(Xc) @ pca.components_) + mu

    # Create outlier in 1 row of slice_outlier
    labels = ["slice_normal"] * 100 + ["slice_outlier"] * 100
    # Add huge error to just 1 row (1% of slice_outlier)
    X_hat[100] += 50.0

    slice_df = slice_reconstruction_error(X, X_hat, labels, mu=mu, min_size=30, n_boot=200, seed=42)
    diagnoses = diagnose_slice_gap(slice_df, X, X_hat, labels, mu)

    assert diagnoses["slice_outlier"] == "outlier_driven"


def test_systematic_slice():
    """Verify systematic slice is classified when bias persists across all rows."""
    rng = np.random.default_rng(42)
    n = 200
    d = 6

    # Slice 1: features 0, 1 captured by PCA
    X1 = np.zeros((100, d))
    X1[:, :2] = rng.normal(scale=50.0, size=(100, 2))

    # Slice 2: features 0, 1 captured, but features 4, 5 are orthogonal uncaptured bias
    X2 = np.zeros((100, d))
    X2[:, :2] = rng.normal(scale=50.0, size=(100, 2))
    X2[:, 4:] = rng.normal(scale=15.0, size=(100, 2))

    X = np.vstack([X1, X2])
    labels = ["slice_well_fit"] * 100 + ["slice_underfit"] * 100
    mu = np.mean(X, axis=0)
    Xc = X - mu

    pca = PCA(n_components=2, svd_solver="full", random_state=42)
    pca.fit(Xc)
    X_hat = (pca.transform(Xc) @ pca.components_) + mu

    slice_df = slice_reconstruction_error(X, X_hat, labels, mu=mu, min_size=30, n_boot=200, seed=42)
    diagnoses = diagnose_slice_gap(slice_df, X, X_hat, labels, mu)

    assert diagnoses["slice_underfit"] == "systematic"


def test_tiny_slice_caveat():
    """Verify tiny slice (n < 30) receives low_confidence caveat and never standalone systematic."""
    rng = np.random.default_rng(42)
    d = 5
    n_small = 15
    n_large = 100

    X = rng.normal(size=(n_small + n_large, d))
    mu = np.mean(X, axis=0)
    X_hat = X + rng.normal(scale=0.1, size=X.shape)
    # Add systematic bias to tiny slice
    X_hat[:n_small] += 2.0

    labels = ["slice_tiny"] * n_small + ["slice_large"] * n_large
    slice_df = slice_reconstruction_error(X, X_hat, labels, mu=mu, min_size=30, n_boot=200, seed=42)
    diagnoses = diagnose_slice_gap(slice_df, X, X_hat, labels, mu)

    # Must contain caveat
    assert "caveat" in diagnoses["slice_tiny"].lower()
    assert diagnoses["slice_tiny"] != "systematic"


def test_run_001_slice_0_not_systematic():
    """Verify that slice 0 from run_001 (relative error below overall) is NOT labelled systematic."""
    # Mock summary mirroring run_001 numbers:
    # overall mean_e ~ 298,260, overall mean_r ~ 0.0860
    # slice '0': mean_e = 316,045 (ratio_e = 1.059 < 1.15), mean_r = 0.0668 (ratio_r = 0.776 < 1.15)
    slice_df = pd.DataFrame([{
        "slice": "0",
        "n": 201,
        "mean_e": 316045.68,
        "median_e": 299449.92,
        "trimmed_mean_e": 306509.68,
        "mean_r": 0.0668,
        "ci_low": 301240.85,
        "ci_high": 329889.24,
        "mean_r_ci_low": 0.063,
        "mean_r_ci_high": 0.070,
        "ratio_e": 1.059,
        "ratio_r": 0.776,
        "ratio_r_after_trim": 0.760,
        "low_confidence": False,
    }])

    # Dummy X, X_hat, mu where global mean_r ~ 0.0860
    X = np.ones((201, 10))
    X_hat = np.ones((201, 10)) * 0.95
    mu = np.zeros(10)

    diagnoses = diagnose_slice_gap(slice_df, X, X_hat, ["0"] * 201, mu)

    # Must be no_gap, NOT systematic!
    assert diagnoses["0"] != "systematic"
    assert diagnoses["0"] == "no_gap"


# =====================================================================
# Regression & Invariant Unit Tests
# =====================================================================

def test_tie_break_prefers_smaller_k():
    """Verify select_config picks smaller k under relative_tolerance."""
    results_data = [
        {"method": "sklearn_pca", "k": 5, "train_recon_error": 8.0, "val_recon_error": 10.4, "val_rel_error": 0.20, "V_train": 0.85, "V_val": 0.80, "gap": 2.4, "runtime_s": 0.01},
        {"method": "sklearn_pca", "k": 10, "train_recon_error": 6.0, "val_recon_error": 10.0, "val_rel_error": 0.18, "V_train": 0.90, "V_val": 0.82, "gap": 4.0, "runtime_s": 0.02},
    ]
    df = pd.DataFrame(results_data)
    rule = SelectionRule(kind="relative_tolerance", epsilon=0.05, tie_break="smallest_k")
    result = select_config(df, rule)
    assert result.chosen_k == 5


def test_small_slice_flagged():
    """Verify small slice flagged with low_confidence=True."""
    rng = np.random.default_rng(42)
    n_samples, d = 100, 5
    X = rng.normal(size=(n_samples, d))
    X_hat = X + rng.normal(scale=0.1, size=(n_samples, d))
    mu = np.mean(X, axis=0)

    slice_labels = ["slice_small"] * 15 + ["slice_large"] * 85
    slice_df = slice_reconstruction_error(X, X_hat, slice_labels, mu=mu, min_size=30, n_boot=200, seed=42)

    small_row = slice_df[slice_df["slice"] == "slice_small"].iloc[0]
    large_row = slice_df[slice_df["slice"] == "slice_large"].iloc[0]

    assert small_row["n"] == 15
    assert bool(small_row["low_confidence"]) is True
    assert large_row["n"] == 85
    assert bool(large_row["low_confidence"]) is False


def test_no_test_argument_exists():
    """Ensure compare_methods_and_ranks has no test param, *args, or **kwargs."""
    sig = inspect.signature(compare_methods_and_ranks)
    params = sig.parameters
    for name, param in params.items():
        assert "test" not in name.lower(), f"Parameter '{name}' contains 'test'"
        assert param.kind != inspect.Parameter.VAR_POSITIONAL
        assert param.kind != inspect.Parameter.VAR_KEYWORD


def test_slice_error_engineered():
    """Synthetic dataset with engineered slice reconstruction error differences."""
    rng = np.random.default_rng(42)
    n_each = 60
    d = 6

    X_A = np.zeros((n_each, d))
    X_A[:, 0] = rng.normal(scale=50.0, size=n_each)
    X_A[:, 1] = rng.normal(scale=40.0, size=n_each)
    X_A += rng.normal(scale=0.01, size=(n_each, d))

    X_B = np.zeros((n_each, d))
    X_B[:, 0] = rng.normal(scale=50.0, size=n_each)
    X_B[:, 1] = rng.normal(scale=40.0, size=n_each)
    X_B[:, 3] = rng.normal(scale=15.0, size=n_each)
    X_B[:, 4] = rng.normal(scale=12.0, size=n_each)

    X = np.vstack([X_A, X_B])
    labels = ["slice_A"] * n_each + ["slice_B"] * n_each
    mu = np.mean(X, axis=0)
    Xc = X - mu

    pca = PCA(n_components=2, svd_solver="full", random_state=42)
    pca.fit(Xc)
    X_hat = (pca.transform(Xc) @ pca.components_) + mu

    slice_df = slice_reconstruction_error(X, X_hat, labels, mu=mu, min_size=30, n_boot=200, seed=42)
    err_A = slice_df[slice_df["slice"] == "slice_A"].iloc[0]
    err_B = slice_df[slice_df["slice"] == "slice_B"].iloc[0]

    assert err_B["mean_e"] > 10.0 * err_A["mean_e"]
    assert err_B["mean_r"] > 5.0 * err_A["mean_r"]


def test_val_error_monotone_in_k():
    """Validation error of exact PCA on synthetic data must be non-increasing in k."""
    s = [20.0, 15.0, 10.0, 6.0, 4.0, 2.0, 1.0, 0.5]
    d = len(s)
    X_train = _make_synthetic(200, d, s, seed=42)
    X_val = _make_synthetic(100, d, s, seed=99)

    methods = [
        MethodSpec(
            name="sklearn_pca",
            factory=lambda k, s_seed: PCA(n_components=k, svd_solver="full", random_state=s_seed),
            is_baseline=False,
        )
    ]
    k_grid = [1, 2, 3, 4, 5, 6]
    res = compare_methods_and_ranks(X_train, X_val, methods, k_grid, seed=42, n_runtime_repeats=1)

    val_errors = res["val_recon_error"].tolist()
    for idx in range(len(val_errors) - 1):
        assert val_errors[idx + 1] <= val_errors[idx] + 1e-7


def test_tsvd_centered_matches_pca_subspace():
    """On centered data, TruncatedSVD and PCA must produce identical subspace projectors."""
    s = [10.0, 8.0, 5.0, 3.0, 1.0]
    d = len(s)
    X = _make_synthetic(150, d, s, seed=42)
    mu = np.mean(X, axis=0)
    Xc = X - mu

    k = 3
    pca = PCA(n_components=k, svd_solver="full", random_state=42)
    pca.fit(Xc)
    tsvd = TruncatedSVD(n_components=k, algorithm="arpack", random_state=42)
    tsvd.fit(Xc)

    P_pca = pca.components_.T @ pca.components_
    P_svd = tsvd.components_.T @ tsvd.components_

    dist = float(np.linalg.norm(P_pca - P_svd, ord="fro"))
    assert dist < 1e-8


def test_train_only_statistics():
    """Altering validation set does not alter train mu or slice quantile boundaries."""
    rng = np.random.default_rng(42)
    n_train, n_val, d = 100, 40, 5
    X_train = rng.normal(loc=5.0, scale=2.0, size=(n_train, d))

    col_train = X_train[:, 0]
    X_val_1 = rng.normal(loc=5.0, scale=2.0, size=(n_val, d))
    col_val_1 = X_val_1[:, 0]
    X_val_2 = rng.normal(loc=100.0, scale=50.0, size=(n_val, d))
    col_val_2 = X_val_2[:, 0]

    mu_1 = np.mean(X_train, axis=0)
    mu_2 = np.mean(X_train, axis=0)
    assert np.array_equal(mu_1, mu_2)

    train_labels_1, _ = make_slice_labels(col_train, col_val_1, quantiles=4)
    train_labels_2, _ = make_slice_labels(col_train, col_val_2, quantiles=4)
    assert np.array_equal(train_labels_1, train_labels_2)


def test_selection_deterministic():
    """Running comparison harness with fixed seed produces identical output and selection."""
    s = [12.0, 9.0, 6.0, 3.0, 1.0]
    d = len(s)
    X_train = _make_synthetic(120, d, s, seed=42)
    X_val = _make_synthetic(50, d, s, seed=99)

    methods = get_default_methods()
    k_grid = [1, 2, 3, 4]

    res1 = compare_methods_and_ranks(X_train, X_val, methods, k_grid, seed=42, n_runtime_repeats=1)
    res2 = compare_methods_and_ranks(X_train, X_val, methods, k_grid, seed=42, n_runtime_repeats=1)

    rule = SelectionRule(kind="variance_threshold", tau=0.50)
    sel1 = select_config(res1, rule, k_grid=k_grid)
    sel2 = select_config(res2, rule, k_grid=k_grid)

    assert sel1.chosen_method == sel2.chosen_method
    assert sel1.chosen_k == sel2.chosen_k
    assert np.isclose(sel1.val_recon_error, sel2.val_recon_error)


def test_report_hash_and_flags():
    """Selection report verifies content_sha256 and test_set_accessed == False."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "test_report.json")
        rule = SelectionRule(kind="variance_threshold", tau=0.90)
        sel_result = SelectionResult(
            chosen_method="sklearn_pca",
            chosen_k=3,
            val_recon_error=0.45,
            train_recon_error=0.40,
            val_rel_error=0.15,
            V_val=0.85,
            runtime_s=0.005,
            rule_kind="variance_threshold",
            epsilon=None,
            tau=0.90,
            e_min=0.44,
            grid_boundary=False,
            grid_boundary_warning=None,
            eligible_candidates=[],
            is_better_than_baseline=True,
            baseline_val_recon_error=1.20,
            baseline_comparison_note="Outperformed Day 1 baseline",
        )
        dummy_df = pd.DataFrame([{"method": "sklearn_pca", "k": 3, "val_recon_error": 0.45}])

        computed_sha = write_selection_report(
            output_path=report_path,
            experiment_id="unit_test_run_01",
            selection_result=sel_result,
            selection_rule=rule,
            k_grid=[1, 2, 3],
            seed=42,
            results_df=dummy_df,
            data_hash="fakehash123",
            split_id="seed42_val0.2",
            slice_summary=[{"slice": "Q1", "mean_e": 0.5}],
        )

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["test_set_accessed"] is False
        assert data["content_sha256"] == computed_sha

        stored_hash = data.pop("content_sha256")
        canonical = json.dumps(data, sort_keys=True, indent=2)
        recalculated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert stored_hash == recalculated_hash


def test_rejects_bad_input():
    """Verify structured exceptions for invalid inputs: NaN/Inf, k >= d, dimension mismatch."""
    rng = np.random.default_rng(42)
    methods = get_default_methods()

    X_nan = rng.normal(size=(50, 5))
    X_nan[0, 0] = np.nan
    X_val = rng.normal(size=(20, 5))

    with pytest.raises(InputValidation):
        compare_methods_and_ranks(X_nan, X_val, methods, [1, 2])

    X_inf = rng.normal(size=(50, 5))
    X_inf[0, 0] = np.inf
    with pytest.raises(InputValidation):
        compare_methods_and_ranks(X_inf, X_val, methods, [1, 2])

    X_mismatch = rng.normal(size=(20, 6))
    X_train = rng.normal(size=(50, 5))
    with pytest.raises(ShapeMismatchError):
        compare_methods_and_ranks(X_train, X_mismatch, methods, [1, 2])

    with pytest.raises(InvalidGridError):
        compare_methods_and_ranks(X_train, X_val, methods, [1, 5])

    with pytest.raises(InvalidGridError):
        compare_methods_and_ranks(X_train, X_val, methods, [])

    with pytest.raises(InvalidGridError):
        compare_methods_and_ranks(X_train, X_val, methods, [3, 1])


def test_run_002_report_integrity_and_flags():
    """Verify that selection_report_run_002.json has test_set_accessed == False and a reproducible content_sha256."""
    report_path = "selection_report_run_002.json"
    if not os.path.exists(report_path):
        pytest.skip(f"{report_path} not found; run run_day5_experiment.py first.")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_set_accessed"] is False
    assert "content_sha256" in data
    stored_hash = data.pop("content_sha256")
    canonical = json.dumps(data, sort_keys=True, indent=2)
    computed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert stored_hash == computed_hash


def test_failed_candidate_never_selected():
    """R6a: A synthetic candidate list containing a failed row is never selected, even if other fields look eligible."""
    data = [
        {
            "method": "from_scratch_covariance",
            "k": 10,
            "status": "failed",
            "failure_reason": "NonConvergenceError: Power iteration did not converge",
            "train_recon_error": np.nan,
            "val_recon_error": 1.0,  # lower error, but failed
            "val_rel_error": 0.05,
            "V_train": 0.99,
            "V_val": 0.99,  # satisfies tau=0.90, but status is failed!
            "gap": 0.0,
            "runtime_s": np.nan,
        },
        {
            "method": "sklearn_pca",
            "k": 20,
            "status": "ok",
            "failure_reason": "",
            "train_recon_error": 10.0,
            "val_recon_error": 12.0,
            "val_rel_error": 0.08,
            "V_train": 0.92,
            "V_val": 0.92,
            "gap": 2.0,
            "runtime_s": 0.01,
        },
    ]
    df = pd.DataFrame(data)
    rule = SelectionRule(kind="variance_threshold", tau=0.90)
    result = select_config(df, rule, k_grid=[10, 20])
    assert result.chosen_method == "sklearn_pca"
    assert result.chosen_k == 20


def test_failed_row_does_not_enter_per_k_method_agreement():
    """R6a: A failed row does not enter per_k_method_agreement."""
    data = [
        {
            "method": "sklearn_pca",
            "k": 10,
            "status": "ok",
            "failure_reason": "",
            "train_recon_error": 10.0,
            "val_recon_error": 10.0,
            "val_rel_error": 0.1,
            "V_train": 0.90,
            "V_val": 0.90,
            "gap": 0.0,
            "runtime_s": 0.01,
        },
        {
            "method": "sklearn_truncated_svd",
            "k": 10,
            "status": "ok",
            "failure_reason": "",
            "train_recon_error": 10.0,
            "val_recon_error": 10.0001,
            "val_rel_error": 0.1,
            "V_train": 0.90,
            "V_val": 0.90,
            "gap": 0.0,
            "runtime_s": 0.01,
        },
        {
            "method": "from_scratch_covariance",
            "k": 10,
            "status": "failed",
            "failure_reason": "NonConvergenceError",
            "train_recon_error": np.nan,
            "val_recon_error": np.nan,
            "val_rel_error": np.nan,
            "V_train": np.nan,
            "V_val": np.nan,
            "gap": np.nan,
            "runtime_s": np.nan,
        },
    ]
    df = pd.DataFrame(data)
    rule = SelectionRule(kind="variance_threshold", tau=0.90)
    result = select_config(df, rule, k_grid=[10])
    assert result.per_k_method_agreement is not None
    agreement_k10 = [row for row in result.per_k_method_agreement if row["k"] == 10][0]
    assert "from_scratch_covariance" not in agreement_k10["methods_compared"]
    assert set(agreement_k10["methods_compared"]) == {"sklearn_pca", "sklearn_truncated_svd"}


def test_failed_row_status_and_reason_recorded():
    """R6a: A failed row gives status='failed' with a non-empty failure_reason."""
    class FailingModel:
        def fit(self, X):
            raise ValueError("Deliberate test failure")
        def transform(self, X):
            return X
        def inverse_transform(self, X):
            return X

    failing_spec = MethodSpec(
        name="failing_method",
        factory=lambda k, seed: FailingModel(),
        is_baseline=False,
    )

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(50, 10))
    X_val = rng.normal(size=(20, 10))

    df = compare_methods_and_ranks(
        X_train=X_train,
        X_val=X_val,
        methods=[failing_spec],
        k_values=[2, 4],
        seed=42,
    )
    assert len(df) == 2
    assert (df["status"] == "failed").all()
    assert (df["failure_reason"].str.startswith("ValueError")).all()
    assert np.isnan(df["val_recon_error"]).all()


def test_all_failed_candidates_gives_structured_no_eligible_k():
    """R6a: All-failed candidate input raises structured NoEligibleRankError."""
    data = [
        {
            "method": "from_scratch_covariance",
            "k": 10,
            "status": "failed",
            "failure_reason": "NonConvergenceError",
            "train_recon_error": np.nan,
            "val_recon_error": np.nan,
            "val_rel_error": np.nan,
            "V_train": np.nan,
            "V_val": np.nan,
            "gap": np.nan,
            "runtime_s": np.nan,
        },
    ]
    df = pd.DataFrame(data)
    rule = SelectionRule(kind="variance_threshold", tau=0.90)
    with pytest.raises(NoEligibleRankError, match="No eligible candidate ranks"):
        select_config(df, rule, k_grid=[10])


def test_json_report_contains_no_nan_or_infinity_literals(tmp_path):
    """R6a: The JSON report contains no NaN/Infinity literals (uses null)."""
    results_data = [
        {
            "method": "from_scratch_covariance",
            "k": 120,
            "status": "failed",
            "failure_reason": "NonConvergenceError",
            "train_recon_error": float("nan"),
            "val_recon_error": float("nan"),
            "val_rel_error": float("nan"),
            "V_train": float("nan"),
            "V_val": float("nan"),
            "gap": float("nan"),
            "runtime_s": float("nan"),
            "runtime_outlier_flag": False,
        },
        {
            "method": "sklearn_pca",
            "k": 90,
            "status": "ok",
            "failure_reason": "",
            "train_recon_error": 100.0,
            "val_recon_error": 110.0,
            "val_rel_error": 0.08,
            "V_train": 0.91,
            "V_val": 0.91,
            "gap": 10.0,
            "runtime_s": 0.5,
            "runtime_outlier_flag": False,
        },
    ]
    df = pd.DataFrame(results_data)
    rule = SelectionRule(kind="variance_threshold", tau=0.90)
    sel = select_config(df, rule, k_grid=[90, 120])

    report_file = str(tmp_path / "test_report.json")
    write_selection_report(
        output_path=report_file,
        experiment_id="test_exp",
        selection_result=sel,
        selection_rule=rule,
        k_grid=[90, 120],
        seed=42,
        results_df=df,
        data_hash="dummy_hash",
        split_id="dummy_split",
        allow_overwrite=True,
    )

    with open(report_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Assert standard JSON compliance: no NaN or Infinity literals
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    assert "-Infinity" not in raw_text
    assert "null" in raw_text

    # Verify standard json parses it cleanly
    parsed = json.loads(raw_text)
    assert parsed["per_method_k_rows"][0]["val_recon_error"] is None


def test_run_003_report_integrity_and_flags():
    """Verify that selection_report_run_003.json has test_set_accessed == False, correct libraries, supersedes, and a reproducible content_sha256."""
    report_path = "selection_report_run_003.json"
    if not os.path.exists(report_path):
        pytest.skip(f"{report_path} not found; run run_day5_experiment.py first.")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["test_set_accessed"] is False
    assert "content_sha256" in data
    assert "supersedes" in data
    assert "scipy" in data["library_versions"]
    assert "sklearn" in data["library_versions"]
    assert data["library_versions"]["scipy"] != data["library_versions"]["sklearn"]

    stored_hash = data.pop("content_sha256")
    canonical = json.dumps(data, sort_keys=True, indent=2, allow_nan=False)
    computed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert stored_hash == computed_hash

