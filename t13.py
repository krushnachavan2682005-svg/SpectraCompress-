import numpy as np
print('\n--- T13 Stability Table ---')
print('| kappa | cond(Xc) | max|mean| | true sigma_min | cov sigma_min | svd sigma_min | err_cov | err_svd | cov >= svd | cov >= 100*svd |')
print('|---|---|---|---|---|---|---|---|---|---|')
kappas = [1e2, 1e4, 1e6, 1e7, 1e8]
n, d = 100, 5
for kappa in kappas:
    s = np.array([1.0, 0.5, 0.2, 0.1, 1.0/kappa])
    rng = np.random.default_rng(42)
    U_raw = rng.normal(size=(n, d))
    U_raw -= np.mean(U_raw, axis=0)
    U, _ = np.linalg.qr(U_raw)
    U -= np.mean(U, axis=0)
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    X = U @ np.diag(s) @ V.T
    
    mean_X = np.mean(X, axis=0)
    max_mean = np.max(np.abs(mean_X))
    
    Xc = X - np.mean(X, axis=0)
    cond_Xc = np.linalg.cond(Xc)
    
    sigma_min_true = 1.0 / kappa
    C = X.T @ X / (n - 1)
    evals, _ = np.linalg.eigh(C)
    sigma_min_cov = np.sqrt(max(0, evals[0] * (n - 1)))
    
    _, S_svd, _ = np.linalg.svd(X)
    sigma_min_svd = S_svd[-1]
    
    err_cov = np.abs(sigma_min_cov - sigma_min_true) / sigma_min_true
    err_svd = np.abs(sigma_min_svd - sigma_min_true) / sigma_min_true
    
    last_col = (err_cov >= 100*err_svd) if kappa == 1e8 else "-"
    print(f'| {kappa} | {cond_Xc:.6g} | {max_mean:.6g} | {sigma_min_true:.6g} | {sigma_min_cov:.6g} | {sigma_min_svd:.6g} | {err_cov:.6g} | {err_svd:.6g} | {err_cov >= err_svd} | {last_col} |')
