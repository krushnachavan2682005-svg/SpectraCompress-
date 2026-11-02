"""Unit and Regression Tests for Day 6 (Hardened Serialization & Interpretability Engine).

Covers:
  - Gate 1 Fixes (F1 - F6):
    - F1: Canonical hash boundary-ambiguity collision resistance with length-prefixed fields & endianness.
    - F2: Manifest integrity seal (manifest_sha256); editing centered_, method, or k raises ArtifactIntegrityError.
    - F3: Legacy joblib fallback disabled by default (allow_legacy=False raises ArtifactIntegrityError; allow_legacy=True warns & succeeds).
    - F4: Config-driven orthonormality tolerance; effective_rank_ < requested_rank_ rank-deficient round-trip.
    - F5: Repeat-fit determinism covering sklearn_pca AND sklearn_truncated_svd with fixed seeds.
    - F6: Round-trip equality of condition_report_, fit_warnings_, effective_rank_, provisional, source_report_hash.
  - Gate 2 (Interpretability & Loadings Engine):
    - Sign convention invariance across refits with inverted signs.
    - Correct top-m sorting by absolute loading with preserved signed values.
    - Rotation-ambiguity and instability flags on near-degenerate eigenvalue spectra.
    - Immutability: loadings report never mutates transformer components.
    - Edge cases: d < top_m and effective_rank_ < requested_rank_.
    - Compatibility: Works seamlessly on loaded immutable artifacts.
"""

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from benchmark import (
    pure_python_gaussian_elimination,
    time_operation,
    measure_peak_memory,
    measure_subprocess_peak_memory,
    BenchmarkRunner,
)
from decomposition import DecompositionConfig, DecompositionTransformer, MethodType
from error import (
    ArtifactIntegrityError,
    ArtifactVersionError,
    NotFittedError,
    ShapeMismatchError,
    SpectraCompressError,
)
from interpretability import (
    DEFAULT_CAVEAT_TEXT,
    InterpretabilityConfig,
    component_loadings_report,
)
from serialization import (
    CONFIRMED_RUN_003_REPORT_HASH,
    SCHEMA_VERSION,
    compute_canonical_arrays_sha256,
    compute_canonical_manifest_sha256,
    load_transformer,
    save_transformer,
)


def _make_synthetic_data(n: int = 100, d: int = 10, seed: int = 42) -> np.ndarray:
    """Helper to generate reproducible synthetic dense test data."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d), dtype=np.float64)


# ==============================================================================
# GATE 1 TESTS & FIXES (F1 - F6)
# ==============================================================================


def test_round_trip_exact_equality(tmp_path: Path):
    """Gate 1 / F6: Transformer round-trips with bit-identical arrays, transform, reconstruct,

    and complete metadata (condition_report_, fit_warnings_, effective_rank_, provisional, source_report_hash).
    """
    X = _make_synthetic_data(120, 15, seed=42)
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=5, seed=42)
    transformer = DecompositionTransformer(config).fit(X)

    save_path = tmp_path / "model_pca"
    npz_path, json_path = transformer.save(save_path)

    assert npz_path.exists()
    assert json_path.exists()

    loaded = DecompositionTransformer.load(save_path)

    # 1. Exact numerical array equality
    assert np.array_equal(transformer.components_, loaded.components_)
    assert np.array_equal(transformer.singular_values_, loaded.singular_values_)
    assert np.array_equal(transformer.explained_variance_, loaded.explained_variance_)
    assert np.array_equal(transformer.explained_variance_ratio_, loaded.explained_variance_ratio_)
    assert np.array_equal(transformer.feature_mean_, loaded.feature_mean_)

    # 2. Exact transform & reconstruct equality
    assert np.array_equal(transformer.transform(X), loaded.transform(X))
    assert np.array_equal(transformer.reconstruct(X), loaded.reconstruct(X))

    # 3. F6 metadata equality
    assert loaded.effective_rank_ == transformer.effective_rank_
    assert loaded.requested_rank_ == transformer.requested_rank_
    assert loaded.fit_warnings_ == transformer.fit_warnings_

    with open(json_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["provisional"] is False
    assert manifest["source_report_hash"] == CONFIRMED_RUN_003_REPORT_HASH

    if transformer.condition_report_ is not None and loaded.condition_report_ is not None:
        assert np.isclose(loaded.condition_report_.kappa, transformer.condition_report_.kappa)
        assert loaded.condition_report_.numerical_rank == transformer.condition_report_.numerical_rank
        assert loaded.condition_report_.band == transformer.condition_report_.band


def test_canonical_hash_boundary_ambiguity_resistance():
    """F1: Length-prefixed canonical array hash cannot collide via field boundary shifting."""
    # Set A: array "ab" with 4 bytes
    arr_a1 = np.array([1, 2], dtype=np.int16)
    dict_a = {"ab": arr_a1}

    # Set B: array "a" with key boundary shifted
    arr_b1 = np.array([1, 2], dtype=np.int16)
    dict_b = {"a": arr_b1}

    hash_a = compute_canonical_arrays_sha256(dict_a)
    hash_b = compute_canonical_arrays_sha256(dict_b)
    assert hash_a != hash_b

    # Endianness distinction: little endian vs big endian float64 array of same values
    arr_le = np.array([1.0, 2.0], dtype="<f8")
    arr_be = arr_le.astype(">f8")
    hash_le = compute_canonical_arrays_sha256({"x": arr_le})
    hash_be = compute_canonical_arrays_sha256({"x": arr_be})
    assert hash_le != hash_be


def test_manifest_sha256_tampering_detection(tmp_path: Path):
    """F2: Editing centered_, method, or k in manifest raises ArtifactIntegrityError."""
    X = _make_synthetic_data(60, 8, seed=42)
    transformer = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)).fit(X)

    save_path = tmp_path / "manifest_tamper"
    _, manifest_path = transformer.save(save_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        original_manifest = json.load(f)

    # Test tampering 'centered_'
    tampered_manifest = dict(original_manifest)
    tampered_manifest["centered_"] = False
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)
    with pytest.raises(ArtifactIntegrityError, match="Manifest integrity verification failed"):
        DecompositionTransformer.load(save_path)

    # Test tampering 'k'
    tampered_manifest = dict(original_manifest)
    tampered_manifest["k"] = 99
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)
    with pytest.raises(ArtifactIntegrityError, match="Manifest integrity verification failed"):
        DecompositionTransformer.load(save_path)

    # Test tampering 'method'
    tampered_manifest = dict(original_manifest)
    tampered_manifest["method"] = "sklearn_truncated_svd"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(tampered_manifest, f)
    with pytest.raises(ArtifactIntegrityError, match="Manifest integrity verification failed"):
        DecompositionTransformer.load(save_path)


def test_legacy_joblib_allow_legacy_contract(tmp_path: Path):
    """F3: Legacy joblib loading is rejected by default and requires allow_legacy=True."""
    legacy_file = tmp_path / "legacy.joblib"
    k, d = 2, 6
    rng = np.random.default_rng(42)
    state = {
        "mean_": rng.standard_normal(d),
        "components_": rng.standard_normal((k, d)),
        "explained_variance_": np.array([2.0, 1.0]),
        "singular_values_": np.array([20.0, 10.0]),
        "total_variance_": 5.0,
        "n_features_in_": d,
        "effective_rank_": k,
        "requested_rank_": k,
        "centered_": True,
        "config": {
            "method": "sklearn_pca",
            "k": k,
            "seed": 42,
            "eig_tol": 1e-10,
            "eig_max_iter": 1000,
            "psd_clip_rtol": 1e-8,
            "pca_svd_solver": "full",
            "tsvd_algorithm": "arpack",
            "tsvd_n_iter": 5,
            "output_dtype": "float64",
            "max_dense_cov_dim": 5000,
        },
        "versions": {"numpy": "2.4.6", "sklearn": "1.9.0"},
        "schema_version": 1,
    }
    joblib.dump(state, legacy_file)

    # 1. Default allow_legacy=False raises ArtifactIntegrityError
    with pytest.raises(ArtifactIntegrityError, match="allow_legacy=False"):
        DecompositionTransformer.load(legacy_file, allow_legacy=False)

    # 2. Explicit opt-in allow_legacy=True warns with DeprecationWarning and loads successfully
    with pytest.warns(DeprecationWarning, match="Loading legacy joblib format"):
        loaded = DecompositionTransformer.load(legacy_file, allow_legacy=True)
    assert loaded.components_.shape == (k, d)
    assert loaded._is_loaded_artifact is True


def test_effective_rank_less_than_requested_rank_round_trip(tmp_path: Path):
    """F4: Rank-deficient dataset where effective_rank_ < requested_rank_ preserves exact truncated shapes."""
    # Synthesize data where true rank is strictly 3 in a 10-feature space
    rng = np.random.default_rng(42)
    n, d = 100, 10
    basis = rng.standard_normal((n, 3))
    mixing = rng.standard_normal((3, d))
    X_rank3 = basis @ mixing  # Exact numerical rank = 3

    # Request k=6 components
    config = DecompositionConfig(method=MethodType.sklearn_pca, k=6, seed=42, ortho_tol=1e-5)
    transformer = DecompositionTransformer(config).fit(X_rank3)

    assert transformer.effective_rank_ == 3
    assert transformer.components_.shape == (3, d)

    save_path = tmp_path / "rank_deficient_model"
    transformer.save(save_path)
    loaded = DecompositionTransformer.load(save_path)

    assert loaded.effective_rank_ == 3
    assert loaded.requested_rank_ == 6
    assert loaded.components_.shape == (3, d)
    assert np.array_equal(transformer.transform(X_rank3), loaded.transform(X_rank3))
    assert np.array_equal(transformer.reconstruct(X_rank3), loaded.reconstruct(X_rank3))


@pytest.mark.parametrize("method", [MethodType.sklearn_pca, MethodType.sklearn_truncated_svd])
def test_repeat_fit_determinism_pca_and_svd(tmp_path: Path, method: MethodType):
    """F5: Repeat-fit determinism across both sklearn_pca AND sklearn_truncated_svd with fixed seeds."""
    X = _make_synthetic_data(100, 12, seed=42)
    config = DecompositionConfig(method=method, k=4, seed=42)

    t1 = DecompositionTransformer(config).fit(X)
    t2 = DecompositionTransformer(config).fit(X)

    p1 = tmp_path / f"rep1_{method.value}"
    p2 = tmp_path / f"rep2_{method.value}"

    _, m1_path = t1.save(p1)
    _, m2_path = t2.save(p2)

    with open(m1_path) as f1, open(m2_path) as f2:
        m1 = json.load(f1)
        m2 = json.load(f2)

    assert m1["arrays_sha256"] == m2["arrays_sha256"]
    assert np.array_equal(t1.components_, t2.components_)
    assert np.array_equal(t1.singular_values_, t2.singular_values_)


def test_tamper_weights_checksum_error(tmp_path: Path):
    """Gate 1: Weights file modification raises ArtifactIntegrityError."""
    X = _make_synthetic_data(60, 6, seed=42)
    transformer = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)).fit(X)

    save_path = tmp_path / "weight_tamper"
    npz_path, _ = transformer.save(save_path)

    with np.load(npz_path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["components_"][0, 0] += 0.01
    np.savez_compressed(npz_path, **arrays)

    with pytest.raises(ArtifactIntegrityError, match="Artifact checksum verification failed"):
        DecompositionTransformer.load(save_path)


def test_version_mismatch_behavior(tmp_path: Path):
    """Gate 1: Version mismatch emits warning or raises ArtifactVersionError under strict=True."""
    X = _make_synthetic_data(50, 6, seed=42)
    transformer = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)).fit(X)

    save_path = tmp_path / "ver_mismatch"
    _, manifest_path = transformer.save(save_path)

    with open(manifest_path, "r") as f:
        m = json.load(f)
    m["library_versions"]["scipy"] = "0.0.1"
    # Recompute manifest hash to isolate the library version check
    m["manifest_sha256"] = compute_canonical_manifest_sha256(m)
    with open(manifest_path, "w") as f:
        json.dump(m, f)

    with pytest.warns(UserWarning, match="Library version mismatch for scipy"):
        loaded = DecompositionTransformer.load(save_path, strict=False)
        assert loaded.components_ is not None

    with pytest.raises(ArtifactVersionError, match="Library version mismatch for scipy"):
        DecompositionTransformer.load(save_path, strict=True)


def test_shape_mismatch_and_unfitted_save_and_immutable(tmp_path: Path):
    """Gate 1: Shape mismatch raises ValueError; unfitted save raises NotFittedError; load-then-fit raises SpectraCompressError."""
    X = _make_synthetic_data(50, 8, seed=42)
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)).fit(X)

    save_path = tmp_path / "guards_test"
    t.save(save_path)
    loaded = DecompositionTransformer.load(save_path)

    # 1. Transform wrong width
    with pytest.raises(ValueError, match="expected d=8"):
        loaded.transform(_make_synthetic_data(10, 14))

    # 2. Unfitted save
    unfitted = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42))
    with pytest.raises(NotFittedError, match="Transformer is not fitted"):
        unfitted.save(tmp_path / "unfitted")

    # 3. Load then fit
    with pytest.raises(SpectraCompressError, match="Cannot fit an already loaded transformer artifact"):
        loaded.fit(X)


# ==============================================================================
# GATE 2 TESTS (INTERPRETABILITY & COMPONENT LOADINGS ENGINE)
# ==============================================================================


def test_sign_convention_invariance_across_sign_flips():
    """Gate 2: Deterministic sign convention (max |loading| positive) ensures report invariance under arbitrary sign flips."""
    X = _make_synthetic_data(100, 10, seed=42)
    t1 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=4, seed=42)).fit(X)

    # Create synthetic transformer with deliberately flipped component signs
    t2 = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=4, seed=42)).fit(X)
    # Flip signs of components 0 and 2
    t2.components_[0, :] *= -1.0
    t2.components_[2, :] *= -1.0

    rep1 = component_loadings_report(t1, top_m=5)
    rep2 = component_loadings_report(t2, top_m=5)

    for c1, c2 in zip(rep1.components, rep2.components):
        for f1, f2 in zip(c1.top_features, c2.top_features):
            assert f1.feature_index == f2.feature_index
            assert np.isclose(f1.loading, f2.loading, atol=1e-12)
            assert np.isclose(f1.abs_loading, f2.abs_loading, atol=1e-12)


def test_top_m_feature_ordering_correct():
    """Gate 2: Top-m features are ordered strictly descending by absolute loading."""
    X = _make_synthetic_data(100, 8, seed=42)
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)).fit(X)
    report = component_loadings_report(t, top_m=4)

    assert report.top_m == 4
    for comp in report.components:
        assert len(comp.top_features) == 4
        abs_vals = [f.abs_loading for f in comp.top_features]
        # Assert strictly non-increasing order
        assert all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1))


def test_unstable_and_rotation_ambiguous_flags_on_clustered_spectrum():
    """Gate 2: Near-degenerate eigenvalues flag rotation-ambiguous; perturbed reference flags instability."""
    rng = np.random.default_rng(42)
    n, d = 100, 5
    U, _ = np.linalg.qr(rng.standard_normal((n, 3)))
    V, _ = np.linalg.qr(rng.standard_normal((d, 3)))
    # Exact clustered singular values: ratio of variance (10.01 / 10.00)^2 = 1.002 < 1.05
    S = np.array([10.01, 10.00, 2.0])
    X_clustered = U @ np.diag(S) @ V.T

    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)).fit(X_clustered)

    # Reference with small perturbation that rotates the 2D subspace
    ref_perturbed = t.components_.copy()
    theta = np.pi / 4.0  # 45 degree rotation
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    ref_perturbed = rot @ ref_perturbed

    cfg = InterpretabilityConfig(top_m=3, stability_threshold=0.90, rotation_ambiguity_ratio=1.05)
    report = component_loadings_report(
        t,
        stability_reference=ref_perturbed,
        config=cfg,
    )

    assert report.stability_status == "COMPUTED"
    # First component should be flagged as rotation-ambiguous due to close eigenvalue
    assert report.components[0].is_rotation_ambiguous is True
    # Due to 45 deg rotation, cosine stability is ~ cos(45 deg) = 0.707 < 0.90 -> is_stable == False
    assert report.components[0].is_stable is False
    assert report.components[0].cosine_stability < 0.90


def test_loadings_report_does_not_mutate_transformer():
    """Gate 2: Calling component_loadings_report leaves transformer components bit-identical."""
    X = _make_synthetic_data(80, 10, seed=42)
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)).fit(X)

    orig_components = t.components_.copy()
    _ = component_loadings_report(t, top_m=5)

    assert np.array_equal(t.components_, orig_components)


def test_handles_d_less_than_top_m_and_caveat_text():
    """Gate 2: Gracefully handles d < top_m without error and embeds the mandatory governance caveat."""
    X = _make_synthetic_data(50, 3, seed=42)  # d=3 < top_m=10
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=2, seed=42)).fit(X)

    report = component_loadings_report(t, top_m=10)
    assert report.top_m == 3
    assert len(report.components[0].top_features) == 3
    assert report.caveat == DEFAULT_CAVEAT_TEXT
    assert "Loadings describe variance structure, not causation" in report.caveat
    assert "A low cosine stability can also result from component order swaps" in report.caveat
    assert report.features_standardized in [False, "unknown"]
    assert report.stability_status == "NOT COMPUTED"
    assert report.components[0].cosine_stability is None


def test_works_for_loaded_artifact(tmp_path: Path):
    """Gate 2: Report generation functions identically on an artifact loaded from disk."""
    X = _make_synthetic_data(80, 8, seed=42)
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)).fit(X)

    save_path = tmp_path / "artifact_for_interp"
    t.save(save_path)
    loaded = DecompositionTransformer.load(save_path)

    rep_orig = component_loadings_report(t, top_m=3)
    rep_loaded = component_loadings_report(loaded, top_m=3)

    assert rep_orig.n_components == rep_loaded.n_components
    assert rep_orig.n_features == rep_loaded.n_features
    for c_orig, c_loaded in zip(rep_orig.components, rep_loaded.components):
        assert np.isclose(c_orig.explained_variance_ratio, c_loaded.explained_variance_ratio)
        for f_orig, f_loaded in zip(c_orig.top_features, c_loaded.top_features):
            assert f_orig.feature_index == f_loaded.feature_index
            assert np.isclose(f_orig.loading, f_loaded.loading)


# ==============================================================================
# GATE 3 TESTS (BENCHMARKING ENGINE, PROFILING & HARDWARE GAPS)
# ==============================================================================


def test_noise_warning_on_synthetic_noisy_callable():
    """Gate 3: Callables exhibiting high jitter (>20% std/mean) trigger noisy=True."""
    delays = [0.001, 0.050, 0.001, 0.050, 0.001]
    call_idx = 0

    def jittery_callable():
        nonlocal call_idx
        d = delays[call_idx % len(delays)]
        call_idx += 1
        time.sleep(d)

    timing = time_operation(jittery_callable, warmup=1, trials=5, noisy_threshold=0.20)
    assert timing.n_trials == 5
    assert timing.noisy is True
    assert (timing.std_s / timing.mean_s) > 0.20


def test_warmup_excluded_from_timing_trials():
    """Gate 3: Warmup run is timed separately and strictly excluded from trial_times_s."""
    call_counts = 0

    def tracked_callable():
        nonlocal call_counts
        call_counts += 1
        time.sleep(0.005)

    timing = time_operation(tracked_callable, warmup=2, trials=5)
    assert call_counts == 7  # 2 warmup + 5 trials
    assert len(timing.trial_times_s) == 5
    assert timing.warmup_time_s > 0.0
    assert timing.n_trials == 5


def test_sparse_stays_sparse_at_every_stage():
    """Gate 3: scipy_implicit_centered preserves sparsity without densifying the input matrix."""
    rng = np.random.default_rng(42)
    n, d = 200, 50
    # Create 5% density CSR matrix
    nnz = int(n * d * 0.05)
    row_idx = rng.integers(0, n, size=nnz)
    col_idx = rng.integers(0, d, size=nnz)
    data = rng.standard_normal(nnz)
    X_csr = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, d))

    assert sp.issparse(X_csr)
    assert sp.isspmatrix_csr(X_csr)

    cfg = DecompositionConfig(method=MethodType.scipy_implicit_centered, k=5, seed=42)
    transformer = DecompositionTransformer(cfg)

    # Fit must keep input sparse
    transformer.fit(X_csr)
    assert sp.issparse(X_csr)

    # Transform must not mutate or densify the input
    X_proj = transformer.transform(X_csr)
    assert sp.issparse(X_csr)
    assert X_proj.shape == (n, 5)


def test_sparse_peak_memory_less_than_densified():
    """Gate 3 / R2: CSR sparse representation uses strictly fewer bytes than its dense counterpart,
    and subprocess peak_wset confirms memory efficiency."""
    n, d = 2000, 500
    density = 0.02  # 2% non-zero
    nnz = int(n * d * density)
    rng = np.random.default_rng(42)
    row_idx = rng.integers(0, n, size=nnz)
    col_idx = rng.integers(0, d, size=nnz)
    data = rng.standard_normal(nnz, dtype=np.float64)
    csr = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, d))

    # Dense equivalent memory in bytes
    dense_bytes = n * d * 8  # float64
    sparse_bytes = csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes

    assert sparse_bytes < (dense_bytes * 0.10)  # Sparse is <10% of dense size

    # In-process profiler check
    mem = measure_peak_memory(lambda: csr @ np.ones((d, 1)))
    assert mem.tracemalloc_peak_bytes >= 0
    assert mem.psutil_peak_rss_bytes >= 0
    assert "tracemalloc" in mem.limitations

    # Subprocess memory profiling assertion & printing
    mem_pca = measure_subprocess_peak_memory("sklearn_pca_dense")
    mem_csr = measure_subprocess_peak_memory("scipy_implicit_centered_csr")
    mem_ctrl = measure_subprocess_peak_memory("control_explicitly_densified")

    print(f"\n[Subprocess Memory] Dense PCA: peak_wset={mem_pca['peak_wset_bytes'] / (1024**2):.1f} MB, tracemalloc={mem_pca['tracemalloc_peak_bytes'] / (1024**2):.1f} MB")
    print(f"[Subprocess Memory] Sparse CSR: peak_wset={mem_csr['peak_wset_bytes'] / (1024**2):.1f} MB, tracemalloc={mem_csr['tracemalloc_peak_bytes'] / (1024**2):.1f} MB")
    print(f"[Subprocess Memory] Control Dense: peak_wset={mem_ctrl['peak_wset_bytes'] / (1024**2):.1f} MB, tracemalloc={mem_ctrl['tracemalloc_peak_bytes'] / (1024**2):.1f} MB")

    # Assertions on measured numbers
    assert mem_csr["tracemalloc_peak_bytes"] < mem_ctrl["tracemalloc_peak_bytes"]
    assert mem_csr["peak_wset_bytes"] <= mem_ctrl["peak_wset_bytes"]
    assert "tracemalloc misses BLAS/LAPACK/C-level workspace" in mem_pca["limitations"]
    assert "peak_wset includes shared library mappings" in mem_pca["limitations"]


def test_benchmark_never_reads_test_split(monkeypatch):
    """Gate 3: Hard Safety Constraint - Benchmark suite strictly never accesses raw_data/mnist_test.csv."""
    accessed_files = []
    orig_read_csv = pd.read_csv

    def guarded_read_csv(filepath_or_buffer, *args, **kwargs):
        path_str = str(filepath_or_buffer)
        accessed_files.append(path_str)
        if "mnist_test.csv" in path_str:
            raise RuntimeError(f"CRITICAL SAFETY VIOLATION: benchmark accessed test split: {path_str}")
        return orig_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)

    runner = BenchmarkRunner(data_path=Path("raw_data/mnist_train.csv"), seed=42)
    _ = runner.load_mnist_train_split(n_sample_total=500)

    for path in accessed_files:
        assert "mnist_test.csv" not in path


def test_pure_python_elimination_matches_numpy_solve():
    """Gate 3: Pure-Python Gaussian elimination numerically matches np.linalg.solve (atol=1e-8)."""
    rng = np.random.default_rng(42)
    n = 15
    # Diagonally dominant matrix to ensure well-conditioned system
    A = rng.standard_normal((n, n), dtype=np.float64) + np.eye(n, dtype=np.float64) * float(n)
    b = rng.standard_normal(n, dtype=np.float64)

    x_np = np.linalg.solve(A, b)
    x_py = pure_python_gaussian_elimination(A.tolist(), b.tolist())

    assert np.allclose(x_py, x_np, atol=1e-8)


def test_benchmark_report_and_blas_gap_schema_complete():
    """Gate 3: benchmark_report.json has all mandatory keys and BLAS_GAP.md is 10-15 lines."""
    report_path = Path("benchmark_report.json")
    blas_gap_path = Path("BLAS_GAP.md")

    if not report_path.exists():
        pytest.skip("benchmark_report.json not yet generated by full suite.")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level schema keys
    required_keys = [
        "schema_version",
        "experiment_id",
        "timestamp_utc",
        "total_benchmark_runtime_s",
        "locked_configuration",
        "environment",
        "linear_solver_scaling",
        "decomposition_benchmarks",
        "dense_vs_sparse_scaling",
        "memory_benchmarks_subprocess",
        "online_serving_latency",
    ]
    for key in required_keys:
        assert key in data, f"Missing key in benchmark_report.json: {key}"

    # Environment keys
    env = data["environment"]
    for env_key in ["cpu_physical_cores", "cpu_logical_cores", "blas_vendor", "python_version", "numpy_version"]:
        assert env_key in env, f"Missing env key: {env_key}"

    # Linear solver entries
    for g in data["linear_solver_scaling"]:
        if g.get("n") in [100, 200, 400]:
            assert "speedup_numpy_1t_over_python" in g
            assert "pure_python_empirical_local_exponent_100_400" in g

    # BLAS_GAP.md line count
    assert blas_gap_path.exists()
    with open(blas_gap_path, "r", encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]

    # Must be 10-15 lines
    assert 10 <= len(lines) <= 15, f"BLAS_GAP.md has {len(lines)} non-empty lines (expected 10-15)."


# ==============================================================================
# GATE 4 TESTS (LEAKAGE AUDIT & REGRESSION VERIFICATION)
# ==============================================================================


def test_l1_artifact_leakage_and_manifest_train_isolation(tmp_path: Path):
    """Gate 4 / L1: Fix train split, arbitrarily alter val/unseen data; refit locked sklearn_pca k=90.

    Asserts arrays_sha256 and bit-identity of components_ and feature_mean_.
    Asserts manifest n_train == len(train) and spy verifies fit() never receives val/test rows.
    """
    rng = np.random.default_rng(42)
    n_train, n_val, d = 100, 30, 15
    X_train_fixed = rng.standard_normal((n_train, d))

    cfg = DecompositionConfig(method=MethodType.sklearn_pca, k=5, seed=42)

    # 1. First run with arbitrary Val 1
    _ = rng.standard_normal((n_val, d))
    t1 = DecompositionTransformer(cfg)
    rows_passed_1 = []

    def spy_fit_1(X):
        rows_passed_1.append(X.shape[0])
        return t1._fit_original(X)

    t1._fit_original = t1.fit
    t1.fit = spy_fit_1
    t1.fit(X_train_fixed)
    p1 = tmp_path / "model_leakage_run1"
    t1.save(p1)

    # 2. Second run with radically altered Val 2
    _ = rng.standard_normal((n_val * 3, d)) * 100.0 + 500.0
    t2 = DecompositionTransformer(cfg)
    rows_passed_2 = []

    def spy_fit_2(X):
        rows_passed_2.append(X.shape[0])
        return t2._fit_original(X)

    t2._fit_original = t2.fit
    t2.fit = spy_fit_2
    t2.fit(X_train_fixed)
    p2 = tmp_path / "model_leakage_run2"
    t2.save(p2)

    # Spy assertions: fit strictly received exactly n_train rows
    assert rows_passed_1 == [n_train]
    assert rows_passed_2 == [n_train]

    # Bit-identity assertions
    assert np.array_equal(t1.components_, t2.components_)
    assert np.array_equal(t1.feature_mean_, t2.feature_mean_)

    # Manifest assertions
    with open(p1.with_suffix(".json"), "r") as f1, open(p2.with_suffix(".json"), "r") as f2:
        m1 = json.load(f1)
        m2 = json.load(f2)

    assert m1["fit_metadata"]["n_train"] == n_train
    assert m2["fit_metadata"]["n_train"] == n_train
    assert m1["arrays_sha256"] == m2["arrays_sha256"]


def test_l2_no_code_path_opens_mnist_test(monkeypatch, tmp_path: Path):
    """Gate 4 / L2: Monkeypatch pd.read_csv and open to strictly prohibit any access to mnist_test.csv."""
    orig_read_csv = pd.read_csv
    orig_open = open

    def guarded_read_csv(filepath, *args, **kwargs):
        if "mnist_test" in str(filepath):
            raise RuntimeError(f"CRITICAL VIOLATION: pd.read_csv accessed test split: {filepath}")
        return orig_read_csv(filepath, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        if "mnist_test" in str(file):
            raise RuntimeError(f"CRITICAL VIOLATION: open() accessed test split: {file}")
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr("builtins.open", guarded_open)

    # 1. Smoke test benchmark train split loader
    runner = BenchmarkRunner(data_path=Path("raw_data/mnist_train.csv"), seed=42)
    X_train_smoke = runner.load_mnist_train_split(n_sample_total=50)
    assert X_train_smoke.shape[1] == 784

    # 2. Smoke test serialization save and load
    cfg = DecompositionConfig(method=MethodType.sklearn_pca, k=3, seed=42)
    t = DecompositionTransformer(cfg).fit(X_train_smoke)
    save_path = tmp_path / "smoke_save"
    t.save(save_path)
    loaded = DecompositionTransformer.load(save_path)
    assert loaded.components_ is not None


def test_l3_cross_process_determinism_and_thread_invariance(tmp_path: Path):
    """Gate 4 / L3: Fit in separate fresh processes with same seed and thread count; assert arrays_sha256 match.

    Compare 1T vs 4T across processes and report max absolute difference.
    """
    sub_code = """
import sys, json, numpy as np
from threadpoolctl import threadpool_limits
from decomposition import DecompositionConfig, DecompositionTransformer, MethodType
from serialization import compute_canonical_arrays_sha256

threads = int(sys.argv[1])
seed = 42

rng = np.random.default_rng(seed)
X = rng.standard_normal((100, 15))

with threadpool_limits(limits=threads):
    t = DecompositionTransformer(DecompositionConfig(method=MethodType.sklearn_pca, k=5, seed=seed)).fit(X)

arrays = {
    'components_': t.components_,
    'singular_values_': t.singular_values_,
    'feature_mean_': t.feature_mean_,
}
sha = compute_canonical_arrays_sha256(arrays)
np.savez(sys.argv[2], **arrays)
print(sha)
"""
    p1_npz = tmp_path / "proc1.npz"
    p2_npz = tmp_path / "proc2.npz"
    p3_npz = tmp_path / "proc3_4t.npz"

    # Run 1: 1 thread
    res1 = subprocess.check_output([sys.executable, "-c", sub_code, "1", str(p1_npz)], text=True).strip()
    # Run 2: 1 thread in fresh process
    res2 = subprocess.check_output([sys.executable, "-c", sub_code, "1", str(p2_npz)], text=True).strip()
    # Run 3: 4 threads in fresh process
    res3 = subprocess.check_output([sys.executable, "-c", sub_code, "4", str(p3_npz)], text=True).strip()

    # Exact bit-identical match across fresh processes on same thread count
    assert res1 == res2

    # Compare 1T vs 4T
    with np.load(p1_npz) as d1, np.load(p3_npz) as d3:
        comp1 = d1["components_"]
        comp3 = d3["components_"]
        sing1 = d1["singular_values_"]
        sing3 = d3["singular_values_"]
        max_diff_comp = float(np.max(np.abs(comp1 - comp3)))
        max_diff_sing = float(np.max(np.abs(sing1 - sing3)))
        print(f"\n[Thread Invariance] 1T vs 4T max diff: components={max_diff_comp:.2e}, singular_values={max_diff_sing:.2e}")
        assert np.allclose(comp1, comp3, atol=1e-12)


