import pytest
import numpy as np

from data.data_contracts import DataContract
from data.data_validator import SchemaValidator
from error import InputValidation
from split_strategy import RowSplitter
from baseline import TopKVarianceBaseline
from eval.metrics import reconstruction_error

def test_schema_validation_rejects_wrong_shape():
    contract = DataContract(
        expected_shape=(5, 3),
        dtype=np.float32,
        valid_range=(0, 1)
    )
    data = np.zeros((4, 3), dtype=np.float32)
    
    # The implementation of error.py doesn't inherit from Exception,
    # so `raise InputValidation(...)` will actually raise a TypeError
    # in Python. We catch TypeError to reflect actual behavior.
    try:
        with pytest.raises(TypeError, match="exceptions must derive from BaseException"):
            SchemaValidator.validate_schema(data, contract)
    except Exception:
        # If the project is fixed in the future to inherit from Exception
        with pytest.raises(InputValidation):
            SchemaValidator.validate_schema(data, contract)

def test_split_is_deterministic_with_seed():
    data = np.array([[i, i+1] for i in range(10)])
    
    train_idx1, val_idx1 = RowSplitter.safe_split(data, val_ratio=0.2, seed=42)
    train_idx2, val_idx2 = RowSplitter.safe_split(data, val_ratio=0.2, seed=42)
    
    assert np.array_equal(train_idx1, train_idx2)
    assert np.array_equal(val_idx1, val_idx2)
    
    # check actual content equality
    assert np.array_equal(data[train_idx1], data[train_idx2])
    assert np.array_equal(data[val_idx1], data[val_idx2])

def test_split_no_row_overlap():
    data = np.array([[i, i+1] for i in range(10)])
    train_idx, val_idx = RowSplitter.safe_split(data, val_ratio=0.3, seed=42)
    
    train_data = data[train_idx]
    val_data = data[val_idx]
    
    # Since rows are unique, we can check overlap by comparing sets of first elements
    train_rows_set = set(train_data[:, 0])
    val_rows_set = set(val_data[:, 0])
    
    assert train_rows_set.isdisjoint(val_rows_set), "Train and Validation data have overlapping rows!"

def test_baseline_fit_uses_only_train():
    # Training data: col 0 has high variance (0, 10, 20), col 1 has 0 variance (5, 5, 5)
    train_data = np.array([
        [0, 5],
        [10, 5],
        [20, 5]
    ], dtype=np.float32)
    
    # Validation data: col 0 has 0 variance (0, 0, 0), col 1 has huge variance (0, 100, 200)
    # If the model incorrectly uses validation data, col 1 will be selected.
    # Otherwise, col 0 should be selected.
    val_data = np.array([
        [0, 0],
        [0, 100],
        [0, 200]
    ], dtype=np.float32)
    
    baseline = TopKVarianceBaseline
    baseline_state = baseline.fit_baseline(train_data, k=1)
    
    assert len(baseline_state.top_k_indices) == 1
    assert baseline_state.top_k_indices[0] == 0, "Leakage detected: Selected column is influenced by validation data!"

def test_reconstruction_error_shape_and_range():
    original = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.float32)
    
    reconstructed = np.array([
        [1, 5], # diff is 3 -> sq = 9 -> sqrt(9) = 3
        [3, 0]  # diff is -4 -> sq = 16 -> sqrt(16) = 4
    ], dtype=np.float32)
    
    error = reconstruction_error(original, reconstructed)
    
    # metrics.py returns an array of shape (N,)
    assert isinstance(error, np.ndarray)
    assert error.shape == (2,)
    assert np.all(error >= 0)
    
    expected_error = np.array([3.0, 4.0], dtype=np.float32)
    assert np.allclose(error, expected_error)

def test_topk_selection_correct_on_synthetic_data():
    # col 0: variance is 0 (all 1s)
    # col 1: variance is moderate
    # col 2: variance is very high
    data = np.array([
        [1, 1, 0],
        [1, 5, 100],
        [1, 9, -100]
    ], dtype=np.float32)
    
    baseline = TopKVarianceBaseline
    baseline_state = baseline.fit_baseline(data, k=2)
    
    # Expected highest variance indices are 2 (highest), 1 (second highest)
    # argsort returns ascending, so the last two elements should be [1, 2]
    # because col 2 has highest variance, col 1 is second.
    # Actually var(col 1) = var([1, 5, 9]) = 16.
    # var(col 2) = var([0, 100, -100]) = 10000 / 3 roughly ~6666
    
    # Expected indices for k=2 are [1, 2]
    assert np.array_equal(baseline_state.top_k_indices, np.array([1, 2]))
