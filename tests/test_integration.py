import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

from split_strategy import RowSplitter


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
    Asserts:
      - method == 'sklearn_pca'
      - k == 90
      - seed == 42
      - source_report_sha256 == 11b88254...
      - canonical config_hash is verified
    """
    config_path = Path("final_config.json")
    assert config_path.exists(), "final_config.json does not exist."

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    run_003_path = Path("selection_report_run_003.json")
    assert run_003_path.exists(), "selection_report_run_003.json missing."

    with open(run_003_path, "r", encoding="utf-8") as f:
        report_003 = json.load(f)

    # Core locked parameters
    assert cfg["method"] == report_003["chosen_method"] == "sklearn_pca"
    assert cfg["k"] == report_003["chosen_k"] == 90
    assert cfg["seed"] == report_003["seed"] == 42
    assert cfg["split_seed"] == 42
    assert cfg["solver_seed"] == 42
    assert cfg["bootstrap_seed"] == 42
    assert cfg["benchmark_seed"] == 42
    assert cfg["source_report_sha256"] == report_003["content_sha256"] == "11b882546939181721f87cc4eabda960bb9bfe894842de0001d3bd6c4f189b95"

    # Verify canonical hash integrity
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

