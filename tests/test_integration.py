import json
import hashlib
import tempfile
from pathlib import Path
import numpy as np
import pytest

from split_strategy import RowSplitter
from data.data_contracts import DataContract
from data.data_validator import SchemaValidator
from error import InputValidation
from run_pipeline import SpectraCompressPipeline, TestSplitAuditHook
from serialization import load_transformer


def compute_canonical_config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if k != "config_hash"}
    canonical_bytes = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def test_safe_split_zero_validation_rows_quirk():
    """
    A4 Test: Documents known edge-case in RowSplitter.safe_split.
    When int(val_ratio * total_rows) == 0, -validation_rows evaluates to -0 == 0.
    Python slicing indices[-0:] equals indices[0:] (all rows to val),
    and indices[:-0] equals indices[:0] (empty array to train).
    Documented in docs/known_issues.md (Issue 8).
    """
    data = np.arange(100)
    train_idx, val_idx = RowSplitter.safe_split(data, val_ratio=0.0, seed=42)
    assert len(train_idx) == 0
    assert len(val_idx) == 100
    assert np.array_equal(np.sort(val_idx), np.arange(100))

    # Also check small sample where int(val_ratio * total_rows) == 0
    small_data = np.arange(4)
    train_idx_small, val_idx_small = RowSplitter.safe_split(small_data, val_ratio=0.2, seed=42)
    assert len(train_idx_small) == 0
    assert len(val_idx_small) == 4


def test_config_hash_matches_locked_report():
    """
    Gate 1: Verifies that final_config.json exactly matches the locked Run 003 selection report.
    """
    config_path = Path("final_config.json")
    assert config_path.exists(), "final_config.json does not exist."

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_003_path = Path("selection_report_run_003.json")
    assert run_003_path.exists(), "selection_report_run_003.json missing."

    with open(run_003_path, "r", encoding="utf-8") as f:
        report_003 = json.load(f)

    assert cfg["method"] == report_003["chosen_method"] == "sklearn_pca"
    assert cfg["k"] == report_003["chosen_k"] == 90
    assert cfg["seed"] == report_003["seed"] == 42
    assert cfg["split_seed"] == 42
    assert cfg["solver_seed"] == 42
    assert cfg["bootstrap_seed"] == 42
    assert cfg["benchmark_seed"] == 42
    assert cfg["source_report_sha256"] == report_003["content_sha256"] == "11b882546939181721f87cc4eabda960bb9bfe894842de0001d3bd6c4f189b95"

    expected_hash = compute_canonical_config_hash(cfg)
    assert cfg["config_hash"] == expected_hash, f"Hash mismatch: {cfg['config_hash']} vs {expected_hash}"


def test_pass_criterion_pre_registered():
    """
    Gate 1: Verifies that reports/pass_criterion.json is pre-registered,
    contains the derived tau and SE from validation data, links to config_hash,
    and states that tau does not account for distribution shift.
    """
    crit_path = Path("reports/pass_criterion.json")
    assert crit_path.exists(), "pass_criterion.json must be pre-registered on disk."

    with open(crit_path, "r", encoding="utf-8") as f:
        crit = json.load(f)

    with open("final_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert crit["config_hash"] == cfg["config_hash"]
    assert crit["tau"] == 0.05
    assert np.isclose(crit["val_mse"], 335548.67510134965, rtol=1e-5)
    assert np.isclose(crit["val_se"], 3149.690341335804, rtol=1e-5)
    assert crit["n_val"] == 2000
    assert "distribution shift" in crit["pre_registration_notes"].lower()


def test_malformed_batch_rejected_at_validator():
    """
    Gate 2: Malformed batch (NaN, Inf, wrong width) rejected at the validation stage
    with clear error messages originating strictly from SchemaValidator (not decomposition).
    """
    contract = DataContract(
        expected_shape=(10, 784),
        dtype=np.dtype("float64"),
        valid_range=(0, 255),
    )

    # 1. NaN rejection
    data_nan = np.zeros((10, 784), dtype=np.float64)
    data_nan[2, 50] = np.nan
    with pytest.raises(InputValidation) as exc_nan:
        SchemaValidator.validate_schema(data_nan, contract)
    assert "Nan" in exc_nan.value.message or "NaN" in exc_nan.value.message
    assert exc_nan.type is InputValidation

    # 2. Inf rejection
    data_inf = np.zeros((10, 784), dtype=np.float64)
    data_inf[4, 100] = np.inf
    with pytest.raises(InputValidation) as exc_inf:
        SchemaValidator.validate_schema(data_inf, contract)
    assert exc_inf.type is InputValidation

    # 3. Wrong width (e.g. 500 features instead of 784)
    data_wrong_width = np.zeros((10, 500), dtype=np.float64)
    with pytest.raises(InputValidation) as exc_width:
        SchemaValidator.validate_schema(data_wrong_width, contract)
    assert "Shape Mismatch" in exc_width.value.message
    assert exc_width.type is InputValidation


def test_pipeline_artifact_round_trip():
    """
    Gate 2: End-to-end artifact save and load round-trip using pipeline.
    Ensures that loaded artifact computes identical transformations and reconstructions.
    """
    with open("final_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_cfg = dict(cfg)
        test_cfg["artifact_dir"] = str(Path(tmp_dir) / "test_artifact")

        pipeline = SpectraCompressPipeline(test_cfg)
        X, labels = pipeline.load_and_validate()
        pipeline.split_data(X, labels)
        pipeline.fit()

        npz_p, manifest_p = pipeline.save_artifact()
        assert npz_p.exists()
        assert manifest_p.exists()

        # Load back
        loaded = load_transformer(test_cfg["artifact_dir"])
        assert loaded.components_ is not None

        # Verify exact reconstruction equality
        X_sample = pipeline.X_val[:50]
        orig_recon = pipeline.transformer.reconstruct(X_sample)
        loaded_recon = loaded.reconstruct(X_sample)
        np.testing.assert_allclose(orig_recon, loaded_recon, rtol=1e-12, atol=1e-12)


def test_validation_evaluation_determinism():
    """
    Gate 2: Determinism of validation evaluation across two independent pipeline runs
    (exact float equality under same seed and threads).
    """
    with open("final_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pipe1 = SpectraCompressPipeline(cfg)
    X1, l1 = pipe1.load_and_validate()
    pipe1.split_data(X1, l1)
    pipe1.fit()
    m1 = pipe1.evaluate_validation()

    pipe2 = SpectraCompressPipeline(cfg)
    X2, l2 = pipe2.load_and_validate()
    pipe2.split_data(X2, l2)
    pipe2.fit()
    m2 = pipe2.evaluate_validation()

    assert m1["val_mse"] == m2["val_mse"], f"Val MSE non-deterministic: {m1['val_mse']} vs {m2['val_mse']}"
    assert m1["val_std"] == m2["val_std"]
    assert m1["val_se"] == m2["val_se"]
    assert m1["val_variance_explained"] == m2["val_variance_explained"]


def test_validation_results_match_day5_expectations():
    """
    Gate 2: Validation results equal Day 5 Run 003 expectations within numerical precision.
    Flags any discrepancy if present.
    """
    with open("final_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Expected from Run 003
    expected_val_mse = 335548.67510134965
    expected_v_val = 0.9032187240037692

    pipe = SpectraCompressPipeline(cfg)
    X, labels = pipe.load_and_validate()
    pipe.split_data(X, labels)
    pipe.fit()
    m = pipe.evaluate_validation()

    diff_mse = abs(m["val_mse"] - expected_val_mse)
    diff_v = abs(m["val_variance_explained"] - expected_v_val)

    assert np.isclose(m["val_mse"], expected_val_mse, rtol=1e-7), (
        f"Validation MSE mismatch: {m['val_mse']} vs expected {expected_val_mse} (diff={diff_mse})"
    )
    assert np.isclose(m["val_variance_explained"], expected_v_val, rtol=1e-4), (
        f"Validation variance explained mismatch: {m['val_variance_explained']} vs expected {expected_v_val} (diff={diff_v})"
    )


def test_quarantined_test_split_untouched_in_gate2():
    """
    Gate 2: Confirms that running the pipeline in default mode strictly NEVER touches
    the quarantined test split (access count remains 0).
    """
    TestSplitAuditHook.reset()
    assert TestSplitAuditHook.get_count() == 0

    with open("final_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_cfg = dict(cfg)
        test_cfg["artifact_dir"] = str(Path(tmp_dir) / "gate2_artifact")
        pipe = SpectraCompressPipeline(test_cfg)
        summary = pipe.run(final_eval=False)

        assert summary["test_metrics"] is None
        assert TestSplitAuditHook.get_count() == 0, "Quarantined test split was accessed during default run!"
