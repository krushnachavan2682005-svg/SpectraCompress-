"""
Diagnostic script for investigating from_scratch_covariance failure at k=100 and k=120.
Does not touch any test set. Strictly uses raw_data/mnist_train.csv.
"""
import os
import sys
import traceback
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from split_strategy import RowSplitter
from decomposition import DecompositionConfig, DecompositionTransformer, MethodType, clip_psd_eigenvalues
from linalg_core import top_k_eigenpairs

def diagnose():
    print("=== Loading Data and Splitting ===")
    df_raw = pd.read_csv("raw_data/mnist_train.csv", nrows=10000)
    X_all = df_raw.drop(columns=["label"]).values.astype(np.float64)
    train_idx, val_idx = RowSplitter.safe_split(X_all, val_ratio=0.20, seed=42)
    X_train = X_all[train_idx]
    
    mu_train = np.mean(X_train, axis=0, dtype=np.float64)
    Xc_train = X_train - mu_train
    n, d = Xc_train.shape
    C = (Xc_train.T @ Xc_train) / (n - 1)
    C = (C + C.T) / 2.0
    
    print(f"n_samples={n}, n_features={d}")
    print(f"Covariance matrix C shape: {C.shape}")
    
    # Ground truth eigenvalues using numpy.linalg.eigvalsh
    true_evals = np.linalg.eigvalsh(C)[::-1]
    print(f"Top 5 true eigenvalues: {true_evals[:5]}")
    print(f"True eigenvalue at 100: {true_evals[99]:.6e}")
    print(f"True eigenvalue at 120: {true_evals[119]:.6e}")
    print(f"Number of negative true eigenvalues: {np.sum(true_evals < 0)}")
    
    # Test k=100
    print("\n--- Testing k=100 ---")
    try:
        cfg100 = DecompositionConfig(
            method=MethodType.from_scratch_covariance,
            k=100,
            seed=42,
            eig_tol=1e-6,
            eig_max_iter=2000,
        )
        t100 = DecompositionTransformer(cfg100)
        t100.fit(Xc_train)
        print("k=100 SUCCEEDED!")
        evals_100 = t100.singular_values_**2 / (n - 1)
        print(f"k=100 evals min/max: {evals_100.min():.6e} / {evals_100.max():.6e}")
        print(f"k=100 max abs difference vs true evals: {np.max(np.abs(evals_100 - true_evals[:100])):.6e}")
    except Exception as e:
        print(f"k=100 FAILED with {type(e).__name__}: {e}")
        traceback.print_exc()

    # Test k=120
    print("\n--- Testing k=120 ---")
    try:
        cfg120 = DecompositionConfig(
            method=MethodType.from_scratch_covariance,
            k=120,
            seed=42,
            eig_tol=1e-6,
            eig_max_iter=2000,
        )
        t120 = DecompositionTransformer(cfg120)
        t120.fit(Xc_train)
        print("k=120 SUCCEEDED!")
    except Exception as e:
        print(f"k=120 FAILED with {type(e).__name__}: {e}")
        traceback.print_exc()

    # Step-by-step diagnostic of deflation and power iteration up to k=125
    print("\n--- Detailed step-by-step deflation from k=90 to 125 ---")
    from linalg_core import power_iteration, deflate
    current_A = C.copy()
    extracted_evals = []
    
    for step in range(125):
        norm_before = np.linalg.norm(current_A)
        try:
            ep = power_iteration(A=current_A, tol=1e-6, max_iter=2000, seed=42)
            val = ep["eigenvalue"]
            extracted_evals.append(val)
            if step >= 95:
                print(f"Step {step+1}: eigenvalue={val:.6e}, true_eval={true_evals[step]:.6e}, iterations={ep['iterations_used']}, norm_before={norm_before:.6e}")
            current_A = deflate(A=current_A, eigenvalue=val, eigenvector=ep["eigenvector"])
        except Exception as e:
            print(f"Step {step+1} FAILED: {type(e).__name__}: {e}, norm_before={norm_before:.6e}")
            break

    # Check clip_psd_eigenvalues
    print("\n--- Checking clip_psd_eigenvalues behavior ---")
    arr_extracted = np.array(extracted_evals)
    neg_count = np.sum(arr_extracted < 0)
    near_zero = np.sum(np.abs(arr_extracted) < 1e-12)
    print(f"Extracted count: {len(arr_extracted)}, Negative: {neg_count}, Near zero: {near_zero}")
    try:
        clipped = clip_psd_eigenvalues(arr_extracted, 1e-8)
        print("clip_psd_eigenvalues SUCCEEDED")
    except Exception as e:
        print(f"clip_psd_eigenvalues FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    diagnose()
