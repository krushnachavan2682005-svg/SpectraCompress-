# Rejection Notice: Experiment Run 001 (`day5_mnist_selection_01`)

## Status: REJECTED

- **Experiment ID:** `day5_mnist_selection_01`
- **Original Report Artifact:** [`selection_report.json`](file:///d:/Personal%20Projects/ML%20PROJECTS/SpectraCompress/selection_report.json)
- **Original Content SHA-256:** `ad21e8703a011fe8f1039d2c9747d271f4df97e7655a4f6dfb5381d2763e12c8`
- **Original Chosen Configuration:**
  - Method: `sklearn_pca`
  - Rank: $k = 100$
  - Validation Reconstruction MSE: $298260.8387$
  - Validation Variance Explained: $V_{\text{val}} = 0.9140$ ($91.40\%$)
  - Rule: `relative_tolerance` with $\epsilon = 0.05$ over grid $[5, 10, 20, 30, 50, 80, 100]$

---

## Reasons for Rejection

1. **Grid-Boundary Degeneracy Artifact:**
   The `relative_tolerance` rule selects candidate ranks satisfying $E \le (1 + \epsilon) \cdot E_{\min}$. In run 001, $E_{\min}$ occurred at the maximum rank tested ($k = 100$, error $298260.84$). The tolerance threshold was $(1 + 0.05) \times 298260.84 = 313173.88$. The adjacent rank in the grid ($k = 80$) had validation error $380647.10 > 313173.88$. Consequently, $k = 100$ was the *only* eligible candidate, forcing the selection to pick the upper boundary of the grid. The outcome was a boundary artifact of grid discretization rather than evidence of optimal compression.

2. **Inverted and Overly Aggressive Slice Diagnoses:**
   In run 001's error analysis module, 7 of 10 digit slices were classified as `systematic` under an overly broad heuristic. In particular, digit `0` had lower relative error ($r = 0.0668$) than the global mean ($r = 0.0860$) and an absolute error within $6\%$ of overall mean, yet was erroneously labelled `systematic`.

3. **Val-Reuse Awareness and Policy Amendment:**
   Because validation results from run 001 were observed, selecting a new configuration requires an explicit formal rule amendment rather than ad-hoc retuning. 

---

## Amendment Implemented in Run 002

1. **Rule Replacement:** The selection criterion is replaced with `variance_threshold`: choose the smallest rank $k$ satisfying $V_{\text{val}}(k) \ge \tau$ (with locked primary threshold $\tau = 0.90$ and sensitivity analysis at $\tau = 0.95$).
2. **Expanded Rank Grid:** The evaluation grid is extended to:
   $$k \in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]$$
3. **Boundary Detection:** If the chosen rank equals $\max(k_{\text{grid}})$, the `grid_boundary` flag fires a mandatory warning. If no rank satisfies $\tau$, a structured `NoEligibleRankError` is raised rather than silently selecting the maximum rank.
4. **Preservation of Run 001 Artifacts:** The original file `selection_report.json` is preserved without modification to maintain cryptographic hash reproducibility. Run 002 results are saved to [`selection_report_run_002.json`](file:///d:/Personal%20Projects/ML%20PROJECTS/SpectraCompress/selection_report_run_002.json).
