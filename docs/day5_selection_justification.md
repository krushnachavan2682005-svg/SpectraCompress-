# Day 5 Selection & Error Analysis Justification Report (Run 003)

**Experiment ID:** `day5_mnist_selection_03`  
**Supersedes:** `run_002 (same rule/config; adds failed-candidate handling, agreement evidence, metadata fix)`  
**Dataset:** MNIST Training Set (`raw_data/mnist_train.csv`, 10000 samples)  
**Partitioning:** Train=8000 samples, Validation=2000 samples (Split seed=42, val_ratio=0.2)  
**Training Data Hash (SHA-256):** `7dbed99748722ddc4ea8733977b88801fe0a3c4a5e4e820145fc2c5056858a8d`  
**Test Set Accessed:** `False` (Strict zero-leakage invariant maintained)  
**Run 003 Content Hash (SHA-256):** `11b882546939181721f87cc4eabda960bb9bfe894842de0001d3bd6c4f189b95`  
**Prior Run 002 Hash:** `8fab482e876ee596b0dc03680405108cfbdae7b38973e9916e871f75ad0d0934`  
**Prior Run 001 Hash:** `ad21e8703a011fe8f1039d2c9747d271f4df97e7655a4f6dfb5381d2763e12c8` ([docs/run_001_rejection.md](file:///d:/Personal%20Projects/ML%20PROJECTS/SpectraCompress/docs/run_001_rejection.md))  

---

## 1. Rule Amendment & Executive Summary

### Formal Rule Amendment Notice
- **Reason for Amendment:** In run 001, the `relative_tolerance` rule was degenerate at the upper grid boundary ($k=100$), selecting the maximum rank solely due to grid truncation.
- **Val-Reuse Disclosure:** Validation results from run 001 had been observed prior to this amendment.
- **Amended Selection Criterion:** Replaced with `variance_threshold`: choose the SMALLEST $k$ such that $V_{val}(k) \ge \tau$, locked at $\tau = 0.9$ with sensitivity analysis at $\tau = 0.95$, across an expanded grid:
  $$k \in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]$$

### Chosen Configuration (Run 003 vs Run 002 Reproduction)
- **Selected Method:** `sklearn_pca`
- **Selected Rank ($k$):** `90`
- **Validation Reconstruction MSE ($E$):** `335548.6751`
- **Training Reconstruction MSE:** `319322.5798`
- **Validation Relative Error ($r$):** `0.0968`
- **Validation Variance Explained ($V_{val}$):** `0.9032` ($90.32\%$)
- **Generalization Gap ($E_{val} - E_{train}$):** `16226.0953`
- **Median Fit Runtime:** `1.4953 s`
- **Grid Boundary State:** `False` (Flagged: `False`)
- **Run 002 vs Run 003 Reproduction:** Both runs deterministically select `sklearn_pca` at $k=90$ for primary threshold $\tau=0.90$, and `sklearn_pca` at $k=150$ for sensitivity threshold $\tau=0.95$. Maximum absolute difference in validation error across all valid rows is $0.0$.

### Sensitivity Analysis Row ($	au = 0.95$)
- Rank at $	au = 0.95$: `200` (`sklearn_pca`, $V_{val} = 0.9650$)

### Baseline Comparison at Equal Rank $k=90$
- Day 1 Baseline Validation Error: `2350015.8578`
- Chosen `sklearn_pca` Validation Error: `335548.6751`
- **Error Reduction:** `85.72\%` lower reconstruction error than baseline.

---

## 2. Root Cause of `from_scratch_covariance` Failure at $k \ge 120$

**One-Sentence Diagnosis:**  
At rank $k=105$, power iteration failed to converge within 2000 iterations due to a tight eigenvalue gap, raising `InputValidation` at `linalg_core.py:37`, which was re-raised as `NonConvergenceError` at `decomposition.py:40` and caught at `model_comparison.py:467` where placeholder infinities and NaNs were injected.

### Diagnostic Evidence (`scripts/diagnose_from_scratch_failure.py`)
1. **$k=100$ Test:** Successfully converged. Extracted 100 eigenvalues matching `numpy.linalg.eigvalsh` within a maximum absolute difference of $4.82 \times 10^{-6}$.
2. **Deflation Breakdown:** From rank 1 to 104, power iteration converges reliably. At Step 105, `power_iteration` exhausts `max_iter=2000` because the deflated matrix norm reaches $\approx 1.902 \times 10^4$ with closely clustered trailing eigenvalues.
3. **Handling in Run 003:** In accordance with repair R6a, failed candidates are recorded with `status="failed"` and structured `failure_reason="NonConvergenceError: Power iteration did not converge"`, while numeric fields are set to null (NaN). Failed candidates are strictly excluded from selection, method agreement, monotonicity checks, and runtime outlier checks.

---

## 3. Model & Rank Comparison Table

| method | k | status | failure_reason | train_recon_error | val_recon_error | val_rel_error | V_train | V_val | gap | runtime_s | runtime_outlier_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sklearn_pca | 5 | ok |  | 2269157.6228 | 2293938.8405 | 0.6616 | 0.3367 | 0.3384 | 24781.2178 | 1.3771 | False |
| sklearn_pca | 10 | ok |  | 1729562.3063 | 1757580.2690 | 0.5069 | 0.4944 | 0.4931 | 28017.9627 | 1.4526 | False |
| sklearn_pca | 20 | ok |  | 1196604.8216 | 1224959.2321 | 0.3533 | 0.6502 | 0.6467 | 28354.4105 | 1.3416 | False |
| sklearn_pca | 30 | ok |  | 902795.0866 | 926762.3693 | 0.2673 | 0.7361 | 0.7327 | 23967.2828 | 1.1575 | False |
| sklearn_pca | 40 | ok |  | 715644.6253 | 735196.7751 | 0.2121 | 0.7908 | 0.7879 | 19552.1498 | 1.5065 | False |
| sklearn_pca | 50 | ok |  | 586348.8717 | 605719.0385 | 0.1747 | 0.8286 | 0.8253 | 19370.1667 | 1.2073 | False |
| sklearn_pca | 60 | ok |  | 491287.0873 | 509629.5345 | 0.1470 | 0.8564 | 0.8530 | 18342.4473 | 1.2084 | False |
| sklearn_pca | 70 | ok |  | 419476.3850 | 435771.2284 | 0.1257 | 0.8774 | 0.8743 | 16294.8434 | 1.3265 | False |
| sklearn_pca | 80 | ok |  | 363723.4874 | 380647.1036 | 0.1098 | 0.8937 | 0.8902 | 16923.6162 | 1.2171 | False |
| sklearn_pca | 90 | ok |  | 319322.5798 | 335548.6751 | 0.0968 | 0.9067 | 0.9032 | 16226.0953 | 1.4953 | False |
| sklearn_pca | 100 | ok |  | 283283.5483 | 298260.8387 | 0.0860 | 0.9172 | 0.9140 | 14977.2904 | 1.2279 | False |
| sklearn_pca | 120 | ok |  | 227544.6390 | 242504.8962 | 0.0699 | 0.9335 | 0.9301 | 14960.2572 | 1.2751 | False |
| sklearn_pca | 150 | ok |  | 169630.2057 | 183888.3500 | 0.0530 | 0.9504 | 0.9470 | 14258.1444 | 1.2632 | False |
| sklearn_pca | 200 | ok |  | 109135.4759 | 121478.2590 | 0.0350 | 0.9681 | 0.9650 | 12342.7831 | 1.4032 | False |
| sklearn_pca | 300 | ok |  | 43646.3920 | 51846.8382 | 0.0150 | 0.9872 | 0.9850 | 8200.4462 | 1.2085 | False |
| from_scratch_covariance | 5 | ok |  | 2269157.6228 | 2293938.7682 | 0.6616 | 0.3367 | 0.3384 | 24781.1454 | 0.3770 | False |
| from_scratch_covariance | 10 | ok |  | 1729562.3063 | 1757580.2751 | 0.5069 | 0.4944 | 0.4931 | 28017.9688 | 0.6115 | False |
| from_scratch_covariance | 20 | ok |  | 1196604.8216 | 1224959.2814 | 0.3533 | 0.6502 | 0.6467 | 28354.4597 | 1.4391 | False |
| from_scratch_covariance | 30 | ok |  | 902795.0866 | 926762.3162 | 0.2673 | 0.7361 | 0.7327 | 23967.2297 | 2.9037 | False |
| from_scratch_covariance | 40 | ok |  | 715644.6253 | 735196.8190 | 0.2121 | 0.7908 | 0.7879 | 19552.1937 | 3.7215 | False |
| from_scratch_covariance | 50 | ok |  | 586348.8718 | 605719.0567 | 0.1747 | 0.8286 | 0.8253 | 19370.1850 | 5.3508 | False |
| from_scratch_covariance | 60 | ok |  | 491287.0873 | 509629.5427 | 0.1470 | 0.8564 | 0.8530 | 18342.4554 | 6.2794 | False |
| from_scratch_covariance | 70 | ok |  | 419476.3850 | 435771.1796 | 0.1257 | 0.8774 | 0.8743 | 16294.7946 | 8.9767 | False |
| from_scratch_covariance | 80 | ok |  | 363723.4874 | 380647.1228 | 0.1098 | 0.8937 | 0.8902 | 16923.6354 | 9.7715 | False |
| from_scratch_covariance | 90 | ok |  | 319322.5798 | 335548.6700 | 0.0968 | 0.9067 | 0.9032 | 16226.0902 | 11.6790 | False |
| from_scratch_covariance | 100 | ok |  | 283283.5483 | 298260.8373 | 0.0860 | 0.9172 | 0.9140 | 14977.2890 | 18.5086 | False |
| from_scratch_covariance | 120 | failed | NonConvergenceError | null | null | null | null | null | null | null | False |
| from_scratch_covariance | 150 | failed | NonConvergenceError | null | null | null | null | null | null | null | False |
| from_scratch_covariance | 200 | failed | NonConvergenceError | null | null | null | null | null | null | null | False |
| from_scratch_covariance | 300 | failed | NonConvergenceError | null | null | null | null | null | null | null | False |
| sklearn_truncated_svd | 5 | ok |  | 2269157.6228 | 2293938.8405 | 0.6616 | 0.3367 | 0.3384 | 24781.2178 | 0.4880 | False |
| sklearn_truncated_svd | 10 | ok |  | 1729562.3063 | 1757580.2690 | 0.5069 | 0.4944 | 0.4931 | 28017.9627 | 0.5216 | False |
| sklearn_truncated_svd | 20 | ok |  | 1196604.8216 | 1224959.2321 | 0.3533 | 0.6502 | 0.6467 | 28354.4105 | 0.8887 | False |
| sklearn_truncated_svd | 30 | ok |  | 902795.0866 | 926762.3693 | 0.2673 | 0.7361 | 0.7327 | 23967.2828 | 0.9383 | False |
| sklearn_truncated_svd | 40 | ok |  | 715644.6253 | 735196.7751 | 0.2121 | 0.7908 | 0.7879 | 19552.1498 | 1.1016 | False |
| sklearn_truncated_svd | 50 | ok |  | 586348.8717 | 605719.0385 | 0.1747 | 0.8286 | 0.8253 | 19370.1667 | 1.2778 | False |
| sklearn_truncated_svd | 60 | ok |  | 491287.0873 | 509629.5345 | 0.1470 | 0.8564 | 0.8530 | 18342.4473 | 1.5354 | False |
| sklearn_truncated_svd | 70 | ok |  | 419476.3850 | 435771.2284 | 0.1257 | 0.8774 | 0.8743 | 16294.8434 | 1.8233 | False |
| sklearn_truncated_svd | 80 | ok |  | 363723.4874 | 380647.1036 | 0.1098 | 0.8937 | 0.8902 | 16923.6162 | 2.0374 | False |
| sklearn_truncated_svd | 90 | ok |  | 319322.5798 | 335548.6751 | 0.0968 | 0.9067 | 0.9032 | 16226.0953 | 2.2863 | False |
| sklearn_truncated_svd | 100 | ok |  | 283283.5483 | 298260.8387 | 0.0860 | 0.9172 | 0.9140 | 14977.2904 | 2.0875 | False |
| sklearn_truncated_svd | 120 | ok |  | 227544.6390 | 242504.8962 | 0.0699 | 0.9335 | 0.9301 | 14960.2572 | 2.7641 | False |
| sklearn_truncated_svd | 150 | ok |  | 169630.2057 | 183888.3500 | 0.0530 | 0.9504 | 0.9470 | 14258.1444 | 3.4854 | False |
| sklearn_truncated_svd | 200 | ok |  | 109135.4759 | 121478.2590 | 0.0350 | 0.9681 | 0.9650 | 12342.7831 | 4.2917 | False |
| sklearn_truncated_svd | 300 | ok |  | 43646.3920 | 51846.8382 | 0.0150 | 0.9872 | 0.9850 | 8200.4462 | 9.1361 | True |
| day1_baseline | 5 | ok |  | 3356829.3798 | 3402954.2365 | 0.9815 | 0.0188 | 0.0185 | 46124.8567 | 0.0514 | False |
| day1_baseline | 10 | ok |  | 3293464.0423 | 3339557.7547 | 0.9632 | 0.0373 | 0.0368 | 46093.7124 | 0.0447 | False |
| day1_baseline | 20 | ok |  | 3167770.1243 | 3213450.0746 | 0.9268 | 0.0741 | 0.0732 | 45679.9502 | 0.0534 | False |
| day1_baseline | 30 | ok |  | 3043137.8892 | 3088734.1528 | 0.8909 | 0.1105 | 0.1091 | 45596.2636 | 0.0370 | False |
| day1_baseline | 40 | ok |  | 2919043.2711 | 2964196.9652 | 0.8550 | 0.1468 | 0.1450 | 45153.6940 | 0.0360 | False |
| day1_baseline | 50 | ok |  | 2795580.3283 | 2839776.9952 | 0.8191 | 0.1828 | 0.1809 | 44196.6669 | 0.0354 | False |
| day1_baseline | 60 | ok |  | 2672779.8216 | 2716026.0736 | 0.7834 | 0.2187 | 0.2166 | 43246.2520 | 0.0350 | False |
| day1_baseline | 70 | ok |  | 2550594.6490 | 2593864.3160 | 0.7481 | 0.2545 | 0.2519 | 43269.6670 | 0.0357 | False |
| day1_baseline | 80 | ok |  | 2428841.7293 | 2471942.5308 | 0.7130 | 0.2900 | 0.2870 | 43100.8015 | 0.0387 | False |
| day1_baseline | 90 | ok |  | 2307505.1172 | 2350015.8578 | 0.6778 | 0.3255 | 0.3222 | 42510.7406 | 0.0391 | False |
| day1_baseline | 100 | ok |  | 2186927.1771 | 2229544.7167 | 0.6431 | 0.3608 | 0.3569 | 42617.5396 | 0.0437 | False |
| day1_baseline | 120 | ok |  | 1948690.9349 | 1989106.6871 | 0.5737 | 0.4304 | 0.4263 | 40415.7523 | 0.0438 | False |
| day1_baseline | 150 | ok |  | 1599944.0398 | 1639692.9162 | 0.4729 | 0.5323 | 0.5271 | 39748.8764 | 0.0379 | False |
| day1_baseline | 200 | ok |  | 1061022.8744 | 1094706.6344 | 0.3157 | 0.6899 | 0.6843 | 33683.7600 | 0.0358 | False |
| day1_baseline | 300 | ok |  | 327498.7010 | 339924.9805 | 0.0980 | 0.9043 | 0.9020 | 12426.2794 | 0.0363 | False |

---

## 4. Method Agreement & Choice Rationale (R6b)

### Rationale
When candidate PCA and SVD methods produce equivalent subspaces on centered data, the tie is broken deterministically by method_priority ('sklearn_pca' first). sklearn PCA performs internal sample-mean centering during transformation and exposes a production-standard estimator interface, eliminating train-serving skew. Across all ranks k <= 100 where all three methods ran successfully (status 'ok'), sklearn_pca, from_scratch_covariance, and sklearn_truncated_svd agreed within a maximum relative validation error difference of 1.12e-07 (well within numerical precision). runtime ordering between sklearn_pca and sklearn_truncated_svd changed between run_001/run_002 timings (for example k=100: 3.26s vs 1.58s for PCA), so runtime is not used as decisive evidence. Solver scaling note: from_scratch_covariance relies on sequential power iteration with Hotelling deflation (A <- A - lambda * v @ v.T). Extracting k components requires k sequential convergence loops over the d x d covariance matrix (O(k * max_iter * d^2)), causing runtime to scale linearly/superlinearly with rank k, whereas SVD solves all components jointly.

---

## 5. Variance Explained Diagnostic Curve

| k | V(k) | marginal_gain | second_diff | elbow_candidate | meets_90_pct | meets_95_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.3384 | 0.3384 | 0.0000 | False | False | False |
| 10 | 0.4931 | 0.1547 | 0.0011 | False | False | False |
| 20 | 0.6467 | 0.1536 | 0.0676 | True | False | False |
| 30 | 0.7327 | 0.0860 | 0.0308 | False | False | False |
| 40 | 0.7879 | 0.0553 | 0.0179 | False | False | False |
| 50 | 0.8253 | 0.0373 | 0.0096 | False | False | False |
| 60 | 0.8530 | 0.0277 | 0.0064 | False | False | False |
| 70 | 0.8743 | 0.0213 | 0.0054 | False | False | False |
| 80 | 0.8902 | 0.0159 | 0.0029 | False | False | False |
| 90 | 0.9032 | 0.0130 | 0.0023 | False | True | False |
| 100 | 0.9140 | 0.0108 | -0.0053 | False | True | False |
| 120 | 0.9301 | 0.0161 | -0.0008 | False | True | False |
| 150 | 0.9470 | 0.0169 | -0.0011 | False | True | False |
| 200 | 0.9650 | 0.0180 | -0.0021 | False | True | True |
| 300 | 0.9850 | 0.0201 | 0.0000 | False | True | True |

*Elbow candidate and threshold flags are descriptive only; selection is governed strictly by the locked variance_threshold rule.*

---

## 6. Corrected Slice Reconstruction Error Analysis

Stratified evaluation across digit classes using leak-free train-only slice assignment and bootstrap confidence intervals on relative error:

| slice | n | mean_e | median_e | trimmed_mean_e | mean_r | ci_low | ci_high | mean_r_ci_low | mean_r_ci_high | ratio_e | ratio_r | ratio_r_after_trim | low_confidence | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 201 | 352037.1929 | 335524.1870 | 340934.9492 | 0.0744 | 334548.8607 | 367612.0090 | 0.0709 | 0.0775 | 1.0491 | 0.7597 | 0.7446 | False | no_gap |
| 1 | 217 | 163031.2585 | 152951.5643 | 156946.9051 | 0.0631 | 155313.4521 | 170725.8409 | 0.0605 | 0.0658 | 0.4859 | 0.6448 | 0.6341 | False | no_gap |
| 2 | 196 | 446156.8040 | 430479.2368 | 440427.6944 | 0.1186 | 428368.6037 | 465296.9801 | 0.1136 | 0.1242 | 1.3296 | 1.2117 | 1.2000 | False | systematic |
| 3 | 213 | 346038.0823 | 307630.3357 | 334332.0625 | 0.0968 | 328720.5577 | 364630.4595 | 0.0929 | 0.1011 | 1.0313 | 0.9890 | 0.9726 | False | no_gap |
| 4 | 210 | 327832.6520 | 305073.9447 | 316480.0223 | 0.1017 | 313856.8526 | 343736.8501 | 0.0983 | 0.1055 | 0.9770 | 1.0386 | 1.0289 | False | no_gap |
| 5 | 195 | 355522.5921 | 339281.9939 | 349566.0613 | 0.1069 | 342423.1900 | 370686.4696 | 0.1030 | 0.1108 | 1.0595 | 1.0915 | 1.0867 | False | no_gap |
| 6 | 195 | 360311.4595 | 328688.8553 | 347679.6083 | 0.1023 | 340346.8336 | 383188.8731 | 0.0975 | 0.1078 | 1.0738 | 1.0449 | 1.0318 | False | no_gap |
| 7 | 201 | 345981.3191 | 311641.9738 | 336845.9262 | 0.1050 | 328947.1978 | 365543.9557 | 0.1015 | 0.1090 | 1.0311 | 1.0728 | 1.0636 | False | no_gap |
| 8 | 164 | 373048.2151 | 335055.0439 | 362367.2402 | 0.1140 | 353691.2536 | 392171.0358 | 0.1082 | 0.1205 | 1.1118 | 1.1648 | 1.1569 | False | systematic |
| 9 | 208 | 310829.7924 | 281998.0057 | 295508.9165 | 0.1025 | 292494.9156 | 329554.0331 | 0.0982 | 0.1075 | 0.9263 | 1.0473 | 1.0265 | False | no_gap |

### Slice Gap Diagnoses
- **Digit `0`:** `no_gap`
- **Digit `1`:** `no_gap`
- **Digit `2`:** `systematic`
- **Digit `3`:** `no_gap`
- **Digit `4`:** `no_gap`
- **Digit `5`:** `no_gap`
- **Digit `6`:** `no_gap`
- **Digit `7`:** `no_gap`
- **Digit `8`:** `systematic`
- **Digit `9`:** `no_gap`


---

## 7. Worst-N Diagnostic (Top 20 Worst Samples by $e$ and by $r$)

*Caveat: N=20 is small; compare each slice's share against the base rate (~10% per digit) and do not interpret this as conclusive proof.*

### Worst 20 Rows by Absolute Error ($e$)
| row_index | e | r | norm_x_minus_mu | slice | slice_share_top_n |
| --- | --- | --- | --- | --- | --- |
| 289 | 1299202.5452 | 0.2711 | 2189.0260 | 6 | 0.1000 |
| 801 | 1109115.0975 | 0.1811 | 2475.0519 | 6 | 0.1000 |
| 1477 | 1017841.3011 | 0.2851 | 1889.3106 | 9 | 0.2000 |
| 1753 | 1010269.6097 | 0.2604 | 1969.8547 | 2 | 0.2000 |
| 1524 | 1009432.5982 | 0.2342 | 2076.1155 | 9 | 0.2000 |
| 877 | 1009184.5416 | 0.1331 | 2753.5598 | 0 | 0.1000 |
| 132 | 984452.5836 | 0.2061 | 2185.5719 | 9 | 0.2000 |
| 417 | 931050.7599 | 0.1793 | 2278.5635 | 4 | 0.1000 |
| 886 | 908126.9361 | 0.1991 | 2135.5378 | 2 | 0.2000 |
| 1849 | 899974.1673 | 0.1889 | 2182.6478 | 8 | 0.1500 |
| 1199 | 858469.0485 | 0.1649 | 2281.3532 | 8 | 0.1500 |
| 1516 | 858152.7756 | 0.1165 | 2713.7472 | 4 | 0.1000 |
| 1270 | 823593.1423 | 0.2775 | 1722.6535 | 3 | 0.1000 |
| 635 | 812551.7687 | 0.1775 | 2139.4574 | 7 | 0.0500 |
| 1564 | 807514.8397 | 0.2048 | 1985.6211 | 0 | 0.1000 |
| 64 | 801307.1540 | 0.2060 | 1972.3818 | 2 | 0.2000 |
| 594 | 798242.1441 | 0.1063 | 2739.7279 | 9 | 0.2000 |
| 750 | 798084.6138 | 0.2243 | 1886.2399 | 8 | 0.1500 |
| 1029 | 794386.1799 | 0.2023 | 1981.8457 | 3 | 0.1000 |
| 1966 | 778577.9632 | 0.1913 | 2017.5359 | 2 | 0.2000 |

### Worst 20 Rows by Relative Error ($r$)
| row_index | e | r | norm_x_minus_mu | slice | slice_share_top_n |
| --- | --- | --- | --- | --- | --- |
| 1477 | 1017841.3011 | 0.2851 | 1889.3106 | 9 | 0.1500 |
| 1270 | 823593.1423 | 0.2775 | 1722.6535 | 3 | 0.0500 |
| 289 | 1299202.5452 | 0.2711 | 2189.0260 | 6 | 0.1000 |
| 1753 | 1010269.6097 | 0.2604 | 1969.8547 | 2 | 0.2500 |
| 187 | 761100.4045 | 0.2415 | 1775.1616 | 2 | 0.2500 |
| 909 | 595756.7600 | 0.2387 | 1579.7548 | 8 | 0.4000 |
| 1168 | 589383.9254 | 0.2382 | 1572.8666 | 8 | 0.4000 |
| 1875 | 584218.1123 | 0.2369 | 1570.3521 | 2 | 0.2500 |
| 442 | 566431.2079 | 0.2345 | 1554.2863 | 8 | 0.4000 |
| 1524 | 1009432.5982 | 0.2342 | 2076.1155 | 9 | 0.1500 |
| 586 | 498801.4601 | 0.2271 | 1481.9557 | 8 | 0.4000 |
| 750 | 798084.6138 | 0.2243 | 1886.2399 | 8 | 0.4000 |
| 1347 | 635687.3197 | 0.2198 | 1700.7111 | 2 | 0.2500 |
| 1407 | 712515.8824 | 0.2175 | 1810.0281 | 6 | 0.1000 |
| 1713 | 551095.2552 | 0.2156 | 1598.8363 | 8 | 0.4000 |
| 1097 | 477844.3767 | 0.2116 | 1502.8437 | 8 | 0.4000 |
| 1549 | 378805.7041 | 0.2098 | 1343.7521 | 8 | 0.4000 |
| 951 | 773843.4019 | 0.2061 | 1937.6405 | 4 | 0.0500 |
| 132 | 984452.5836 | 0.2061 | 2185.5719 | 9 | 0.1500 |
| 64 | 801307.1540 | 0.2060 | 1972.3818 | 2 | 0.2500 |

Reconstruction visualization of the top 8 worst relative-error samples is saved to [`docs/worst_rows_reconstruction.png`](file:///d:/Personal%20Projects/ML%20PROJECTS/SpectraCompress/docs/worst_rows_reconstruction.png).

---

## 8. Leakage & Invariance Verification
- **Duplicate Rows between Train and Validation:** `0` overlapping rows.
- **Distribution Drift Status:** `Max standardized feature mean shift = 0.1720 (threshold: 3.0) [OK: No drift warning fired]`.
- **Test Set Access:** `False`. Held-out test data was never opened or read.

---

## 9. Limitations
- **Single Split:** All metrics are computed on a single 80/20 train/validation split of 10,000 samples.
- **Val-Reuse Optimism:** The selection rule was amended after observing run_001 validation results; reported validation performance may possess slight optimistic bias.
- **Unbiased Assessment Deferred:** True unbiased generalization metrics will only be evaluated once during the final Day 7 test-set run.
- **Fold-Level Variance:** K-fold cross-validation standard deviations are not yet integrated into this increment.

---
*Report generated automatically by `run_day5_experiment.py`.*
