Project Name:- SpectraCompress

## Day 5: Selection & Error Analysis

The Day 5 increment implements a rigorous method and rank selection framework, validation-only configuration selection, and stratified slice/worst-case error analysis for the SpectraCompress dimensionality reduction engine.

### Key Modules

1. **Model & Rank Comparison Harness (`model_comparison.py`)**:
   - `compare_methods_and_ranks`: Evaluates candidate dimensionality reduction methods across a sorted rank grid $k$.
   - **Centering Policy**: Computes training mean $\mu$ strictly from $X_{\text{train}}$ once. Centers both $X_{\text{train}}$ and $X_{\text{val}}$ with that identical $\mu$ before dispatching to all methods (including TruncatedSVD and Day 1 baseline), resolving the uncentered TruncatedSVD problem while maintaining equal effective dimensionality. Adds $\mu$ back when reconstructing.
   - **Output Metrics**: Computes train and validation squared reconstruction error ($e_i = ||x_i - \hat{x}_i||_2^2$), relative error ($r_i = e_i / ||x_i - \mu||_2^2$), variance explained ($V = 1 - \sum e_i / \sum ||x_i - \mu||_2^2$), generalization gap ($E_{\text{val}} - E_{\text{train}}$), and median runtime.
   - **Monotonicity & Drift Diagnostics**: Flags non-monotonic validation error across rank $k$, checks train/validation duplicate-row overlap, and detects distribution drift on active features.
   - **Descriptive Variance Curve (`variance_explained_curve`)**: Diagnostic reporting of marginal gains, second differences (curvature), elbow candidates, and 90%/95% variance thresholds. (Report-only; does not govern automated selection).

2. **Validation-Only Selection (`select_config`)**:
   - Pure function implementing the relative tolerance rule:
     $$E \le (1 + \epsilon) \cdot E_{\min}$$
   - Tie-breaking prefers the smallest rank $k$ (maximum compression and model parsimony).
   - Method ties are broken deterministically using `method_priority`.
   - Compares the chosen configuration directly against the Day 1 baseline at equal rank $k$.

3. **Stratified Slice & Root-Cause Error Analysis (`error_analysis.py`)**:
   - `per_row_errors`: Computes sample-wise squared Euclidean error $e_i$ and relative error $r_i$ with zero-norm denominator protection.
   - `make_slice_labels`: Leak-free assignment of categorical slices and train-only quantile bucket edges applied to validation data.
   - `slice_reconstruction_error`: Computes stratified mean, median, 5% trimmed mean, relative error, and 95% bootstrap confidence intervals, flagging small slices ($n < 30$).
   - `worst_n_rows`: Identifies top-N worst reconstructed rows, feature-level error contributions, and slice proportions.
   - `diagnose_slice_gap`: Categorizes slice performance differentials into:
     - `scale_effect`: Elevated absolute error driven by large row norm $||x - \mu||$, with non-elevated relative error.
     - `outlier_driven`: Mean error driven by extreme tail outliers, where removing the top 1% drops mean error substantially toward the median.
     - `systematic`: Subspace underfitting where both mean and median relative errors remain persistently high.

4. **Tamper-Evident Reporting (`selection_report.json`)**:
   - Programmatically generated canonical JSON report tracking library versions, git commit, experiment configuration, baseline comparison, and slice breakdowns, verified with a SHA-256 integrity hash (`content_sha256`) and strict zero test-set access (`test_set_accessed == false`).

### Day 5 Run 002 Policy Amendment & Diagnostic Repairs (R1–R5)

Run 001 was formally rejected due to boundary discretization degeneracy under `relative_tolerance` (`docs/run_001_rejection.md`). Run 002 incorporates the following repairs:
- **R1: Locked Variance Threshold Selection**: Replaced degenerate relative tolerance rule with locked `variance_threshold` $\tau = 0.90$ (with $\tau = 0.95$ sensitivity analysis) over an expanded rank grid $k \in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]$. Automatically flags `grid_boundary` when $k = \max(k_{\text{grid}})$ and raises `NoEligibleRankError` if no rank reaches threshold.
- **R2: Slice Gap Diagnosis Fixes**: Added bootstrap 95% confidence intervals on relative error $r$, ratio metrics ($ratio_e, ratio_r, ratio_r^{\text{trimmed}}$), and an ordered decision cascade that distinguishes `no_gap`, `scale_effect`, `outlier_driven`, and `systematic` under `SliceDiagnosisConfig`.
- **R3: Method Equivalence & Solver Scaling Rationale**: Added `per_k_method_agreement` capturing cross-method discrepancy, runtime outlier detection flags, and documented solver scaling characteristics (sequential power iteration with Hotelling deflation scaling as $O(k \cdot I_{\max} \cdot d^2)$ versus joint SVD).
- **R4: Dual Worst-N Inspection**: Extracted top-20 worst reconstructed samples by absolute error $e$ and relative error $r$, top-3 pixel error contributions, slice proportion caveat, and visual 8-sample reconstruction comparison artifact (`docs/worst_rows_reconstruction.png`).
- **R5: Auditability & Zero-Leakage Invariant**: Logged duplicate rows (0 overlap) and feature mean/variance drift checks, added a structured limitations section, preserved Run 001 report integrity (`ad21e870...`), and emitted `selection_report_run_002.json` (`8fab482e...`) with strict `test_set_accessed == false`.

### Day 5 Run 003 Governance & Failure Handling Repairs (R6a–R6c)

Run 003 incorporates robustness repairs addressing candidate failures, method agreement evidence, and metadata integrity:
- **R6a: Robust Candidate Failure Handling**: Explicit `status` ("ok" | "failed") and `failure_reason` fields. Non-finite values or caught exceptions are recorded as failed candidates with numeric fields set to null (NaN in DataFrame). Failed candidates are strictly excluded from selection, method agreement, monotonicity checks, and outlier detection. Root cause of `from_scratch_covariance` failure at $k \ge 120$ diagnosed via `scripts/diagnose_from_scratch_failure.py` (spectral gap narrowing at rank 105 causing power iteration convergence exhaustion at `linalg_core.py:37`).
- **R6b: Method Agreement Evidence & Empirical Rationale**: Tracked `methods_compared` and relative validation difference across valid methods; validated method equivalence within $1.12 \times 10^{-7}$ for all $k \le 100$. Explicitly noted runtime rank order instability between runs at $k=100$.
- **R6c: Known Issues & Technical Debt Register (`docs/known_issues.md`)**: Cataloged 7 audited items across Day 1–4 modules for resolution in Day 6 backlog, and corrected `library_versions` metadata in `selection_report_run_003.json` to accurately distinguish SciPy and scikit-learn versions while maintaining zero test-set access.