# ADR 001: Decomposition Method and Subspace Rank Choice for SpectraCompress

- **Status**: Decided / Accepted
- **Date**: 2026-11-04 (Day 7 Release)
- **Deciders**: ML Engineering Pair
- **Consulted**: `selection_report_run_003.json` (`content_sha256`: `11b882546939181721f87cc4eabda960bb9bfe894842de0001d3bd6c4f189b95`)
- **Supersedes**: Run 001 relative-tolerance selection (`ad21e870...`), Run 002 selection (`8fab482e...`)

---

## 1. Context and Problem Statement

SpectraCompress requires a robust, reproducible, and computationally efficient linear dimensionality reduction engine to compress high-dimensional handwritten digit image data ($d = 784$ pixels) while preserving sufficient variance for reconstruction and downstream representation.

The selection must balance:
1. **Reconstruction fidelity**: Minimizing mean squared Euclidean error ($||x - \hat{x}||_2^2$).
2. **Subspace parsimony**: Minimizing the target subspace rank $k$ to optimize storage and downstream inference latency.
3. **Numerical stability & scalability**: Preventing solver divergence, rotation ambiguity, and numerical rank deficiency.
4. **Leakage & isolation guarantees**: Guaranteeing that training and validation statistics remain strictly partitioned and that test data is never snooped or referenced during model selection.

---

## 2. Candidate Methods Evaluated

Four candidate decomposition methods were benchmarked across a rank grid $k \in \{5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300\}$:

1. **`sklearn_pca`**: Scikit-learn PCA utilizing LAPACK `gesdd` divide-and-conquer SVD on mean-centered dense matrices $\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}$.
2. **`from_scratch_covariance`**: In-house covariance matrix computation $\mathbf{C} = \frac{1}{N-1}\mathbf{X}_c^T\mathbf{X}_c$ solved via sequential power iteration and Hotelling deflation (`linalg_core.py`).
3. **`sklearn_truncated_svd`**: Truncated SVD without implicit mean centering.
4. **`TopKVarianceBaseline` (Day 1)**: Heuristic baseline retaining the $k$ features with largest marginal sample variance.

---

## 3. Decision Rule

The pre-registered model selection rule codified in Day 5 Run 003:
$$\min k \quad \text{subject to} \quad \text{VarExpl}_{\text{val}}(k) \ge 0.90 \quad \text{and} \quad \text{status} = \text{"ok"}$$
Ties between methods at equal rank $k$ are broken deterministically by `method_priority` (`sklearn_pca` > `sklearn_truncated_svd` > `from_scratch_covariance`).

---

## 4. Empirical Evaluation & Comparison

| Method | Chosen $k$ | Validation MSE | Validation VarExpl | Baseline Val MSE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`sklearn_pca`** | **90** | **335,548.68** | **90.32%** | **2,350,016.32** | **PASS / SELECTED** |
| `from_scratch_covariance` | 90 | 335,548.68 | 90.32% | 2,350,016.32 | PASS (Fails at $k \ge 120$) |
| `sklearn_truncated_svd` (centered) | 90 | 335,548.68 | 90.32% | 2,350,016.32 | PASS |
| `TopKVarianceBaseline` | 90 | 2,350,016.32 | 34.19% | 2,350,016.32 | REJECTED (Under-retains variance) |

### Key Findings:
- **Subspace Quality**: At $k=90$, `sklearn_pca` achieves **90.32%** variance explained on validation data, exceeding the 90.0% threshold with the smallest eligible rank. On official holdout test data (`raw_data/mnist_test.csv`), it achieves **90.46%** variance explained ($\text{MSE}_{\text{test}} = 327,890.76$).
- **Superiority over Baseline**: `sklearn_pca` reconstruction error is **7.08x lower** than the Top-K variance baseline ($327,890.76$ vs $2,321,904.29$).
- **Convergence Failure of Scratch Solver**: At $k \ge 120$, `from_scratch_covariance` suffers spectral gap collapse ($\lambda_{104} \approx \lambda_{105}$), exhausting the 1000-iteration power iteration budget and failing with `SpectraCompressNonConvergence`.
- **Uncentered Truncated SVD Flaw**: Uncentered SVD directs its 1st component along the global data mean $\boldsymbol{\mu}$ rather than the dominant variance eigenvector, wasting rank capacity on uncentered image pixels.

---

## 5. Architectural & Implementation Rationale

1. **LAPACK SVD vs Sparse Iterative Solver**:
   - While `scipy_implicit_centered` (ARPACK `svds`) fits faster on sparse data ($0.74\text{ s}$ vs $1.45\text{ s}$), for dense $8000 \times 784$ floating-point data, LAPACK `gesdd` provides deterministic orthogonal eigenvectors, exact singular values, and zero convergence exceptions.
2. **Inference Non-Densification Support**:
   - Although PCA is fit on dense matrices, `inference.py` supports streaming CSR sparse inputs by projecting directly via implicit mean multiplication: $\mathbf{Z} = \mathbf{X}_{\text{sparse}}\mathbf{V}_k - \boldsymbol{1}(\boldsymbol{\mu}^T\mathbf{V}_k)$. This eliminates memory blow-up without requiring dense instantiation.
3. **Hardened Serialization**:
   - The selected model weights are serialized into compressed `.npz` arrays accompanied by a cryptographic SHA-256 JSON manifest (`arrays_sha256 = f83c61...`, `content_sha256 = 11b882...`), eliminating insecure `pickle` dependencies.

---

## 6. Consequences & Operational Constraints

- **Storage**: Compact artifact size: `arrays.npz` is ~560 KB, manifest is ~4 KB.
- **Latency**: Single-row transform p50 is $20.0\ \mu\text{s}$ (dense) and $90.0\ \mu\text{s}$ (sparse CSR), easily meeting the $< 1.0\text{ ms}$ production SLA.
- **Immutability Invariant**: Loaded inference artifacts are strictly immutable and never refit.
- **Interpretability Limitation**: Component loadings represent linear stroke patterns across pixels without causal interpretation. Feature stability across bootstrap folds is marked **NOT COMPUTED**.
