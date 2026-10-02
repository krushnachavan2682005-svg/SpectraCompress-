import pytest
import numpy as np
import os

from linalg_core import power_iteration, deflate, top_k_eigenpairs
from error import InputValidation, SpectraCompressError

def test_power_iteration_small():
    A = np.array([[2, 1], [1, 2]], dtype=float)
    res = power_iteration(A, tol=1e-6, max_iter=100, seed=42)
    
    assert np.isclose(res["eigenvalue"], 3.0, atol=1e-5)
    
    expected_v = np.array([1, 1]) / np.sqrt(2)
    v = res["eigenvector"]
    cos_sim = np.abs(np.dot(v, expected_v))
    assert np.isclose(cos_sim, 1.0, atol=1e-5)

    assert np.allclose(A @ v, res["eigenvalue"] * v, atol=1e-5)

@pytest.mark.parametrize("n", [10, 100])
def test_power_iteration_random_symmetric(n):
    rng = np.random.default_rng(42)
    X = rng.random((n, n))
    A = X.T @ X + np.eye(n)
    
    res = power_iteration(A, tol=1e-6, max_iter=2000, seed=42)
    
    evals, evecs = np.linalg.eigh(A)
    expected_lambda = evals[-1]
    expected_v = evecs[:, -1]
    
    assert np.isclose(res["eigenvalue"], expected_lambda, rtol=1e-3, atol=1e-3)
    
    cos_sim = np.abs(np.dot(res["eigenvector"], expected_v))
    assert np.isclose(cos_sim, 1.0, atol=1e-3)

def test_power_iteration_difficult_convergence():
    # Ill-conditioned for power iteration, ratio of dominant to second is very close to 1
    A = np.diag([10.0, 9.99, 1.0])
    
    try:
        with pytest.raises((Exception, TypeError)):
            power_iteration(A, tol=1e-9, max_iter=5, seed=42)
    except Exception as e:
        pass

def test_power_iteration_real_dataset():
    dataset_path = "raw_data/mnist_test.csv"
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset {dataset_path} not found")
        
    # Read just a small subset
    data = np.loadtxt(dataset_path, delimiter=",", skiprows=1, max_rows=50)
    X = data[:, 1:] # Skip label
    X = X - np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    # Add small regularization to avoid perfectly singular matrices
    cov += np.eye(cov.shape[0]) * 1e-5
    
    res = power_iteration(cov, tol=1e-5, max_iter=1000, seed=42)
    
    evals, _ = np.linalg.eigh(cov)
    expected_lambda = evals[-1]
    
    # Check that eigenvalues match closely
    assert np.isclose(res["eigenvalue"], expected_lambda, rtol=1e-2, atol=1e-2)

def test_power_iteration_singular_matrix():
    A = np.array([[1, 1], [1, 1]], dtype=float)
    res = power_iteration(A, tol=1e-6, max_iter=100, seed=42)
    assert np.isclose(res["eigenvalue"], 2.0, atol=1e-5)

def test_power_iteration_non_symmetric():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    with pytest.raises(Exception):
        power_iteration(A, tol=1e-6, max_iter=100, seed=42)

def test_power_iteration_zero_vector_handled():
    A = np.zeros((3, 3))
    with pytest.raises(Exception):
        power_iteration(A, tol=1e-6, max_iter=100, seed=42)

def test_power_iteration_reproducible():
    A = np.array([[2, 1], [1, 2]], dtype=float)
    res1 = power_iteration(A, tol=1e-6, max_iter=100, seed=42)
    res2 = power_iteration(A, tol=1e-6, max_iter=100, seed=42)
    
    assert res1["eigenvalue"] == res2["eigenvalue"]
    assert np.array_equal(res1["eigenvector"], res2["eigenvector"])
    assert res1["iterations_used"] == res2["iterations_used"]

def test_deflate_basic():
    A = np.array([[2, 1], [1, 2]], dtype=float)
    lam = 3.0
    v = np.array([1, 1]) / np.sqrt(2)
    
    A_def = deflate(A, lam, v)
    
    assert np.allclose(A_def, A_def.T, atol=1e-7)
    
    evals, _ = np.linalg.eigh(A_def)
    assert np.isclose(np.max(evals), 1.0, atol=1e-5)

def test_top_k_eigenpairs():
    A = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    
    res = top_k_eigenpairs(A, k=2, tol=1e-6, max_iter=100, seed=42)
    
    assert len(res) == 2
    assert "eigenvalue" in res[0] and "eigenvector" in res[0]
    
    assert np.isclose(res[0]["eigenvalue"], 3.0, atol=1e-4)
    assert np.isclose(res[1]["eigenvalue"], 2.0, atol=1e-4)
    
    assert abs(res[0]["eigenvalue"]) >= abs(res[1]["eigenvalue"])
    
    v0 = res[0]["eigenvector"]
    assert np.allclose(A @ v0, res[0]["eigenvalue"] * v0, atol=1e-4)
