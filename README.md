# SpectraCompress

**SpectraCompress** is a high-performance, mathematically rigorous linear dimensionality reduction and compression engine tailored for high-dimensional image spectra (MNIST 784-pixel digit representations).

The system decomposes $d = 784$ dimensional vectors into a compact $k = 90$ dimensional subspace via Principal Component Analysis (PCA), achieving an **$8.7\times$ dimensional reduction** while capturing **$90.46\%$ of dataset variance** with rigorous data leakage boundaries, tamper-evident cryptographic serialization, and high-throughput zero-densification sparse inference.

---

## 1. Architecture and Locked Configuration

The production model configuration is cryptographically locked in `final_config.json` (`config_hash`: `0448cc7ad697ba4a602eb1472191704d347d7bc9cb4b55dfec2304d752e5ae90`), derived from Day 5 Model Selection Run 003 (`content_sha256`: `11b882546939181721f87cc4eabda960bb9bfe894842de0001d3bd6c4f189b95`):

- **Algorithm**: `sklearn_pca` (LAPACK `gesdd` divide-and-conquer SVD on mean-centered $\mathbf{X}_c$)
- **Subspace Rank ($k$)**: `90` components
- **Seed Determinism**: `split_seed=42`, `solver_seed=42`, `bootstrap_seed=42`, `benchmark_seed=42`
- **Serialization Format**: Dual `.npz` arrays + canonical JSON manifest with SHA-256 seals (zero `pickle` dependencies)

---

## 2. Measured Benchmark Performance (Day 7)

All performance numbers are measured empirically on the host environment and recorded in `reports/final_evaluation.json` and `reports/performance_check.json`:

### 2.1 Model Quality & Generalization
- **Holdout Test Set Reconstruction MSE**: **$327,890.76$** ($\text{SE} = 1,324.36$)
- **Validation Set Reconstruction MSE**: **$335,548.68$** ($\text{SE} = 3,149.69$)
- **Relative Generalization Gap**: **$2.28\%$** ($\le \tau = 5.00\%$ pre-registered pass criterion)
- **Generalization Assessment**: Consistent with validation; test MSE is $2.3\%$ lower (about $2.2$ combined SE), not claimed as better.
- **Test Variance Explained ($R^2$)**: **$90.46\%$** ($0.9046$)
- **Baseline Test MSE (`TopKVarianceBaseline`, $k=90$)**: **$2,321,904.29$** ($7.08\times$ higher error than SpectraCompress)

### 2.2 Latency & Throughput (OpenBLAS / Intel Host)
- **Single-Row Serving Latency (Dense)**: **$20.0\ \mu\text{s}$** p50 ($0.020\text{ ms}$)
- **Single-Row Serving Latency (Sparse CSR)**: **$90.0\ \mu\text{s}$** p50 ($0.090\text{ ms}$)
- **End-to-End Batch Serving (100 rows)**: **$9.98\text{ ms}$**
- **End-to-End Batch Serving (1,000 rows)**: **$27.24\text{ ms}$**
- **Decomposition Fit Time ($8000 \times 784$, 1 Thread)**: **$1.449\text{ s}$** ($20.36\text{ nominal GFLOPS}$)
- **Decomposition Fit Time ($8000 \times 784$, 4 Threads)**: **$1.264\text{ s}$** ($23.35\text{ nominal GFLOPS}$)
- **Full Dataset Transform ($8000 \times 784$, 8 Threads)**: **$66.10\text{ ms}$** (BLAS GEMM)

---

## 3. Implemented Components & CLI Usage

### 3.1 Pipeline Execution (`run_pipeline.py`)
Run end-to-end data ingestion, validation, model fitting, evaluation, and artifact generation:
```bash
# Standard training, validation evaluation, and artifact persistence
python run_pipeline.py --config final_config.json

# Production audit run with holdout test evaluation (single-touch quarantined)
python run_pipeline.py --config final_config.json --final-eval
```

### 3.2 Production Inference API (`inference.py`)
Standalone, stateless batch prediction designed for production serving with zero train-serving skew:
```python
import numpy as np
from inference import predict_batch

# X_new can be dense np.ndarray (N, 784) or scipy.sparse.csr_matrix (N, 784)
X_new = np.random.randn(10, 784)
result = predict_batch(X_new, artifact_path="artifacts/spectra_compress_model")

embeddings = result["embeddings"]            # Shape (10, 90)
reconstructions = result["reconstructions"]  # Shape (10, 784)
errors = result["reconstruction_errors"]      # Shape (10,) per-row squared error
```
- **Zero-Densification Invariant**: Sparse CSR inputs are multiplied directly against projection weights without ever converting to dense arrays via `.toarray()` or `.todense()`.

### 3.3 Hardened Serialization (`serialization.py`)
- Saves transformer weights to `arrays.npz` and metadata to `manifest.json`.
- Validates canonical array SHA-256 and manifest checksums on load.
- Rejects refitting loaded models (`is_fitted_` immutability guard).

---

## 4. Test Suite and Boundary Verification

The test suite comprises **252 tests** executed via pytest:
- **233 legacy tests (Days 1–6)** passing unmodified.
- **19 Day 7 integration tests** covering single-touch quarantine, schema validation, crash recovery, sparse mathematical equivalence, and serving parity.

Run the test suite:
```bash
python -m pytest
```

---

## 5. Known Limitations & Scope Declarations

- **Component Loadings**: Loadings correspond to spatial pixel positions describing stroke patterns across digits; cosine stability across cross-validation folds is marked **NOT COMPUTED**; loadings must not be given a causal interpretation (they describe linear variance structure only).
- **Constant Features**: 118 edge pixels in MNIST have empirical variance $\le 10^{-12}$ and contribute zero information to the subspace.
- **Linear-Only Subspace**: Principal components describe optimal orthogonal linear projections; non-linear manifold geometry is not modeled.
- **Dataset Scale**: Evaluated at 10,000-sample scale; cross-machine hardware reproducibility across architectures, float32 precision, and large-$d$ latency crossovers are designated **NOT RUN**.
