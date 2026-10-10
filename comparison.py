import time
import logging
import numpy as np
from sklearn.decomposition import TruncatedSVD

from baseline import TopKVarianceBaseline
from eval.metrics import reconstruction_error, component_cosine_similarity, subspace_distance, mean_squared_reconstruction_error
from decomposition import DecompositionConfig, DecompositionTransformer, MethodType
from error import NonConvergenceError

logger = logging.getLogger(__name__)

GAP_RATIO = 1.05

def format_markdown_table(results: list[dict]) -> str:
    if not results:
        return ""
    headers = list(results[0].keys())
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    rows = []
    for r in results:
        row = []
        for k in headers:
            v = r.get(k)
            if isinstance(v, float):
                row.append(f"{v:.6g}")
            else:
                row.append(str(v))
        rows.append("| " + " | ".join(row) + " |")
        
    return "\n".join([header_str, sep_str] + rows)

def compare_backends(X_train: np.ndarray, X_val: np.ndarray, k_list: list[int], base_config: DecompositionConfig) -> list[dict]:
    """
    Compare Day-1 baseline and all PCA/decomposition backends.
    X_val is used only for transform/evaluation, never for fit.
    """
    n, d = X_train.shape
    max_k = min(n - 1, d)

    k_grid = {1, 2, int(np.ceil(d / 4)), int(np.ceil(d / 2)), d - 1, d}
    k_grid.update(k_list)
    k_values = sorted(k for k in k_grid if 1 <= k <= max_k)

    methods = [
        MethodType.sklearn_pca,
        MethodType.from_scratch_covariance,
        MethodType.sklearn_truncated_svd,
    ]

    results = []

    for k in k_values:
        # Day-1 baseline
        try:
            b_state = TopKVarianceBaseline.fit_baseline(X_train, k)
            
            X_train_hat = np.zeros_like(X_train)
            X_val_hat = np.zeros_like(X_val)
            
            # Use TRAIN mean to fill dropped columns
            train_mean = np.mean(X_train, axis=0)
            X_train_hat[:] = train_mean
            X_val_hat[:] = train_mean

            top_k = b_state.top_k_indices
            X_train_hat[:, top_k] = X_train[:, top_k]
            X_val_hat[:, top_k] = X_val[:, top_k]

            train_mse = mean_squared_reconstruction_error(X_train, X_train_hat)
            val_mse = mean_squared_reconstruction_error(X_val, X_val_hat)

            results.append({
                "k": k,
                "method": "day1_baseline",
                "status": "OK",
                "train_mse": train_mse,
                "val_mse": val_mse,
                "sklearn_var_ratio": None,
                    "uncentered_energy_ratio": None,
                "mean_abs_cos_vs_sklearn_pca": None,
                "min_abs_cos": None,
                "subspace_dist": None,
                "fit_time_s": None,
                "transform_time_s": None,
                "notes": "Used train mean for dropped columns"
            })
        except Exception as exc:
            logger.exception("Baseline failed")
            msg = getattr(exc, "message", str(exc))
            if not msg:
                msg = type(exc).__name__
            results.append({
                "k": k,
                "method": "day1_baseline",
                "status": f"SKIPPED:{type(exc).__name__}",
                "train_mse": None,
                "val_mse": None,
                "sklearn_var_ratio": None,
                    "uncentered_energy_ratio": None,
                "mean_abs_cos_vs_sklearn_pca": None,
                "min_abs_cos": None,
                "subspace_dist": None,
                "fit_time_s": None,
                "transform_time_s": None,
                "notes": msg
            })

        # Precompute full sklearn spectrum for NO_GAP rule
        train_mean_vec = np.mean(X_train, axis=0)
        Xc_train = X_train - train_mean_vec
        s = np.linalg.svd(Xc_train, compute_uv=False)
        lambdas = (s ** 2) / (n - 1)

        has_gap = True
        rank = np.linalg.matrix_rank(Xc_train)
        if k < rank and k < len(lambdas):
            if lambdas[k-1] / (lambdas[k] + 1e-12) < GAP_RATIO:
                has_gap = False

        sklearn_components = None

        for method in methods:
            config = DecompositionConfig(
                method=method,
                k=k,
                seed=base_config.seed,
                eig_tol=base_config.eig_tol,
                eig_max_iter=base_config.eig_max_iter,
                psd_clip_rtol=base_config.psd_clip_rtol,
                pca_svd_solver=base_config.pca_svd_solver,
                tsvd_algorithm=base_config.tsvd_algorithm,
                tsvd_n_iter=base_config.tsvd_n_iter,
                output_dtype=base_config.output_dtype,
            )

            transformer = DecompositionTransformer(config)

            try:
                # Timing for fit
                transformer.fit(X_train) # warm-up
                fit_times = []
                for _ in range(5):
                    start = time.perf_counter()
                    transformer.fit(X_train)
                    fit_times.append(time.perf_counter() - start)
                fit_time_s = float(np.median(fit_times))

                if method == MethodType.sklearn_pca:
                    sklearn_components = transformer.components_.copy()

                X_train_hat = transformer.reconstruct(X_train)
                train_mse = mean_squared_reconstruction_error(X_train, X_train_hat)

                # Timing for transform
                transformer.transform(X_val) # warm-up
                transform_times = []
                for _ in range(5):
                    start = time.perf_counter()
                    Z_val = transformer.transform(X_val)
                    transform_times.append(time.perf_counter() - start)
                transform_time_s = float(np.median(transform_times))

                X_val_hat = transformer.inverse_transform(Z_val)
                val_mse = mean_squared_reconstruction_error(X_val, X_val_hat)

                cum_var_explained_train = float(np.sum(transformer.explained_variance_ratio_))

                status = "NO_GAP" if not has_gap else "OK"
                mean_abs_cos = None
                min_abs_cos = None
                subspace_dist = None

                if method != MethodType.sklearn_pca and sklearn_components is not None:
                    W_current = transformer.components_
                    W_sklearn = sklearn_components
                    subspace_dist = subspace_distance(W_current, W_sklearn)

                    if status != "NO_GAP":
                        abs_cos = component_cosine_similarity(W_current, W_sklearn)
                        mean_abs_cos = float(np.mean(abs_cos))
                        min_abs_cos = float(np.min(abs_cos))

                results.append({
                    "k": k,
                    "method": method.value,
                    "status": status,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "sklearn_var_ratio": cum_var_explained_train,
                    "uncentered_energy_ratio": None,
                    "mean_abs_cos_vs_sklearn_pca": mean_abs_cos,
                    "min_abs_cos": min_abs_cos,
                    "subspace_dist": subspace_dist,
                    "fit_time_s": fit_time_s,
                    "transform_time_s": transform_time_s,
                    "notes": ""
                })
            except Exception as exc:
                if isinstance(exc, NonConvergenceError):
                    status = "NON_CONVERGED"
                    notes = ""
                else:
                    status = f"SKIPPED:{type(exc).__name__}"
                    notes = getattr(exc, "message", str(exc))
                    
                results.append({
                    "k": k,
                    "method": method.value,
                    "status": status,
                    "train_mse": None,
                    "val_mse": None,
                    "sklearn_var_ratio": None,
                    "uncentered_energy_ratio": None,
                    "mean_abs_cos_vs_sklearn_pca": None,
                    "min_abs_cos": None,
                    "subspace_dist": None,
                    "fit_time_s": None,
                    "transform_time_s": None,
                    "notes": notes if notes else type(exc).__name__
                })

        # Diagnostic Uncentered TruncatedSVD
        try:
            tsvd = TruncatedSVD(n_components=k, random_state=base_config.seed)
            tsvd.fit(X_train)
            
            Z_train = tsvd.transform(X_train)
            X_train_hat = tsvd.inverse_transform(Z_train)
            train_mse = mean_squared_reconstruction_error(X_train, X_train_hat)
            
            Z_val = tsvd.transform(X_val)
            X_val_hat = tsvd.inverse_transform(Z_val)
            val_mse = mean_squared_reconstruction_error(X_val, X_val_hat)
            
            cum_var = float(np.sum(tsvd.explained_variance_ratio_))
            
            mean_dir = X_train.mean(axis=0)
            mean_dir_norm = np.linalg.norm(mean_dir)
            if mean_dir_norm > 0:
                mean_dir = mean_dir / mean_dir_norm
            pc1 = tsvd.components_[0]
            cos_sim = abs(np.dot(pc1, mean_dir))
            
            results.append({
                "k": k,
                "method": "diag_uncentered_tsvd",
                "status": "OK",
                "train_mse": train_mse,
                "val_mse": val_mse,
                "sklearn_var_ratio": cum_var,
                "uncentered_energy_ratio": float(np.sum(tsvd.singular_values_**2) / np.sum(X_train**2)),
                "mean_abs_cos_vs_sklearn_pca": None,
                "min_abs_cos": None,
                "subspace_dist": None,
                "fit_time_s": None,
                "transform_time_s": None,
                "notes": f"diagnostic only; |cos(PC1, mean_dir)|={cos_sim:.4g}"
            })
        except Exception as exc:
            pass

    md_table = format_markdown_table(results)
    logger.info("\n" + md_table)
    return results
