# Numerical Hardware & BLAS Acceleration Gap Analysis (Day 6)

1. Linear System Acceleration (n=400): Pure-Python Gaussian elimination achieved nominal 0.0210 GFLOPS (2.035 s), whereas np.linalg.solve (OpenBLAS 1T) achieved nominal 24.70 GFLOPS (0.00173 s), yielding an observed gap of 1178x (range across sessions: ~1760x - 3200x depending on CPU load).
2. Scaling Complexity: Empirical local exponent over n=100..400 (3 points) yields n^3.01 for pure-Python vs n^2.13 for LAPACK dgesv (doubling ratios: py=7.47, np=3.54).
3. Overhead Roots (Hypothesis): The baseline gap combines Python bytecode interpretation and boxed object allocation with LAPACK SIMD/FMA vectorization and cache blocking (combined, not isolated).
4. Multithreading Crossover: Thread synchronization overhead dominates at small sizes (n=400: 1T=1.73 ms vs 4T=3.62 ms); crossover where 4T beats 1T is observed at n=1000 (1T=21.61 ms vs 4T=15.14 ms).
5. Locked Decomposition Scaling: On MNIST (8000x784, k=90), sklearn_pca fit took 1.399 s (1T) vs 1.180 s (4T) using svd_solver=full (nominal formula GFLOPS: 21.1).
6. Algorithm Distinction: Apparent 1.78x speedup between scipy svds (0.750 s) and sklearn PCA (1.335 s) at locked k=90 reflects different algorithms, not attributable to sparsity.
7. Locked Subspace Accuracy: At k=90, scipy_implicit_centered and sklearn_pca achieve identical train reconstruction MSE (319322.58) with max principal angle 2.41e-06° and mean angle 6.26e-07°.
8. Same-Backend Sparsity: scipy_implicit_centered (k=20, not locked config) on CSR vs Dense achieved 2.17x speedup at 1% density and 1.06x at 40% (95% CI: [1.03, 1.08]; no dense crossover observed in tested range).
9. Subprocess Working Set Profile: Sampled peak working set increase (1 ms thread) was 161.3 MB (dense PCA) vs 28.9 MB (sparse CSR); tracemalloc recorded 215.0 MB vs 27.5 MB.
10. Out-of-Core Synthetic Footprint: For synthetic matrix (60000x20000), analytic dense size is 8.94 GB (exceeding 7.75 GB RAM); measured CSR fit working set was only 20.9 MB (never densified).
11. Online Latency: Over 2000 repeats, single-row transform p50 latency is 0.0284 ms (dense) vs 0.0385 ms (sparse CSR).
