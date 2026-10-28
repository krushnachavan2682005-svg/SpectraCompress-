# SpectraCompress Known Issues & Backlog (Day 5 Audit)

This document catalogs technical debt, defects, and latent issues identified during the Day 5 audit of SpectraCompress. In accordance with the governance rule prohibiting modifications to Day 1–4 modules during Day 5, all Day 1–4 issues are scheduled for remediation in the Day 6 backlog as isolated commits.

---

### Issue 1: 1-D Array Double-Indexing in `TopKVarianceBaseline.transform`
- **File**: `baseline.py`
- **Line**: 22
- **Symptom**: Calling `TopKVarianceBaseline().transform(data, state)` raises `IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed`.
- **Evidence**:
  ```python
  # baseline.py:21-22
  top_k_columns = baseline_state.top_k_indices
  return top_k_columns[:, top_k_columns]  # top_k_columns is 1D shape (k,)
  ```
  Attempting 2D indexing on a 1-D numpy array fails immediately with an `IndexError`.
- **Why Day 1 Tests Missed It**:
  In `tests/test_day1.py`, unit tests (`test_baseline_fit_uses_only_train`, `test_topk_selection_correct_on_synthetic_data`) only invoke `TopKVarianceBaseline.fit_baseline(...)` to inspect `BaselineState.top_k_indices`. No test ever called `TopKVarianceBaseline.transform(...)`.
- **Impact on Day 5**: Zero impact. Day 5 model comparison implements an isolated, self-contained `BaselineAdapter` class in `model_comparison.py:270` that correctly performs column extraction via `data[:, self.top_k_indices_]` and inverse transforms by padding non-selected columns with train column means.
- **Proposed Fix**: Change line 22 of `baseline.py` to `return data[:, top_k_columns]`. Add a corresponding unit test in `tests/test_day1.py` asserting proper shape `(N, k)` on transform.
- **Target Resolution**: Day 6 backlog (separate commit).

---

### Issue 2: Incorrect Library Version Key Mapping (`scipy` mapped to `sklearn.__version__`)
- **File**: `model_comparison.py`
- **Line**: 842 (in run_001 / run_002)
- **Symptom**: In the JSON report metadata, the key `"scipy"` recorded the scikit-learn version string (`"1.9.0"`) instead of SciPy's version, and `"sklearn"` was absent.
- **Evidence**:
  In `selection_report_run_002.json`:
  ```json
  "library_versions": {
    "numpy": "2.4.6",
    "pandas": "3.0.2",
    "platform": "Windows-10-10.0.26300-SP0",
    "python": "3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]",
    "scipy": "1.9.0"
  }
  ```
  `sklearn.__version__` is `"1.9.0"`, whereas `scipy.__version__` is `"1.14.1"`.
- **Impact on Day 5**:
  The recorded metadata in `selection_report.json` and `selection_report_run_002.json` contains this typographical dictionary assignment. In strict accordance with the immutability constraint, `selection_report_run_002.json` and its SHA-256 hash (`8fab482e...0934`) are **NOT** modified.
- **Proposed Fix**: Fixed in `model_comparison.py` (Day 5 editable file) by importing `scipy` and explicitly mapping both `"scipy": scipy.__version__` and `"sklearn": sklearn.__version__`. The corrected metadata appears in `selection_report_run_003.json`.
- **Target Resolution**: Completed in Day 5 Run 003.

---

### Issue 3: Zero-Validation Slice Overflow in `RowSplitter.safe_split`
- **File**: `split_strategy.py`
- **Line**: 23–24
- **Symptom**: When `val_ratio = 0.0`, `safe_split` returns 0 training rows and all rows in the validation split.
- **Evidence**:
  ```python
  validation_rows = int(val_ratio * total_rows)  # int(0.0) == 0
  val_indices = indices[-validation_rows:]       # indices[-0:] == indices[0:] (all rows!)
  train_indices = indices[:-validation_rows]     # indices[:-0] == indices[:0] (empty array!)
  ```
  Because in Python slice notation `-0` equals `0`, `indices[-0:]` is interpreted as `indices[0:]`, yielding the full index array instead of an empty slice.
- **Impact on Day 5**: Zero impact. Day 5 strictly specifies `val_ratio = 0.20`, yielding `validation_rows = 1600 > 0`, where negative indexing functions as intended.
- **Proposed Fix**: Add an explicit guard:
  ```python
  if validation_rows == 0:
      return indices, np.empty(0, dtype=int)
  ```
- **Target Resolution**: Day 6 backlog (separate commit).

---

### Issue 4: Python Type Objects Used as Default Arguments in `SpectraCompressError` Hierarchy
- **File**: `error.py`
- **Lines**: 2, 8, 12, 16, 20, 24, 28
- **Symptom**: Exception constructor signatures default to `message=str, code=int`, assigning the built-in type objects `<class 'str'>` and `<class 'int'>` rather than string and integer instances.
- **Evidence**:
  ```python
  class SpectraCompressError(Exception):
      def __init__(self, message=str, code=int):
          self.message = message
          self.code = code
  ```
  Calling `InputValidation()` without parameters results in `err.message` being `<class 'str'>` and `err.code` being `<class 'int'>`.
- **Impact on Day 5**: Minimal. Day 5 code always initializes custom exceptions with explicit string and integer arguments (e.g. `message="...", code=400`). However, unhandled exceptions without explicit messages lack human-readable error descriptions.
- **Proposed Fix**: Change defaults to `message: str = "", code: int = 500`.
- **Target Resolution**: Day 6 backlog (separate commit).

---

### Issue 5: Undefined Variable `ci_includes_overall` During Run 001
- **File**: `error_analysis.py`
- **Line**: 500 (historical)
- **Symptom**: `NameError: name 'ci_includes_overall' is not defined` when evaluating slice diagnoses.
- **Evidence**:
  In original Day 5 implementation of `error_analysis.py`, `ci_includes_overall` was referenced on line 500 without prior definition.
- **Impact on Day 5**:
  Because of this defect, slice labels produced in Run 001 cannot be considered trustworthy. The defect was resolved in `error_analysis.py` prior to Run 002 by properly deriving `ci_includes_overall = bool(mean_r_ci_low <= overall_mean_r <= mean_r_ci_high)`. Run 002 and Run 003 slice diagnostics supersede Run 001.
- **Proposed Fix**: Already corrected in `error_analysis.py`.
- **Target Resolution**: Completed in Day 5.

---

### Issue 6: References to `raw_data/mnist_test.csv` in Day 2 Source and Test Files
- **File & Lines**:
  - `pca_evaluation_report.py:17`
  - `tests/test_day2.py:49`
- **Symptom**: Direct references to the test dataset file path `"raw_data/mnist_test.csv"`.
- **Evidence**:
  ```python
  # pca_evaluation_report.py:17
  test_path = "raw_data/mnist_test.csv"

  # tests/test_day2.py:49
  dataset_path = "raw_data/mnist_test.csv"
  ```
- **Execution in Day 5**:
  Neither file runs in any Day 5 execution path:
  1. `pca_evaluation_report.py` is a Day 2 standalone inspection script that is never imported or invoked by Day 5 code.
  2. `tests/test_day2.py` is a regression test in the test suite that skips if `mnist_test.csv` is absent (or reads 50 lines for basic shape validation in Day 2). It is completely disjoint from Day 5 selection logic.
- **Policy Invariant**: The test set `raw_data/mnist_test.csv` is strictly quarantined and may ONLY be accessed once by the final evaluation script in Day 7.
- **Target Resolution**: Refactor Day 2 files in Day 6 backlog to point to validation/synthetic data.

---

### Issue 7: Verification of Zero Test-Set Infiltration in Day 5 Code Paths
- **Files Audited**:
  - `model_comparison.py`
  - `error_analysis.py`
  - `run_day5_experiment.py`
  - `tests/test_day5.py`
- **Evidence**:
  Recursive pattern matching for `"mnist_test"` across all Day 5 source and test files yields **zero matches**:
  ```
  model_comparison.py: 0 matches
  error_analysis.py: 0 matches
  run_day5_experiment.py: 0 matches
  tests/test_day5.py: 0 matches
  ```
  The invariant `test_set_accessed == False` is programmatically asserted and verified.
