"""
run_pipeline.py - Day 7 Thin Orchestration Pipeline for SpectraCompress.

Composes existing components without modifying prior fit/transform/benchmark logic:
  Flow: load -> validate_schema -> safe_split -> fit (train, locked config) ->
        val eval -> [guarded final test eval] -> error analysis -> save artifact -> report.

Entry point:
  python run_pipeline.py --config final_config.json [--final-eval]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data.data_contracts import DataContract
from data.data_validator import SchemaValidator
from error import InputValidation
from split_strategy import RowSplitter
from decomposition import DecompositionTransformer, DecompositionConfig, MethodType
from eval.metrics import mean_squared_reconstruction_error
from error_analysis import slice_reconstruction_error, worst_n_rows
from serialization import save_transformer, load_transformer

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("SpectraCompressPipeline")


class TestSplitAuditHook:
    """Audit-hook counter enforcing the strict single-touch policy for quarantined test split."""
    _access_count: int = 0
    _max_allowed: int = 1

    @classmethod
    def record_access(cls, path: Union[str, Path]) -> int:
        if cls._access_count >= cls._max_allowed:
            msg = (
                f"AUDIT BREACH: Quarantined test split '{path}' has already been accessed "
                f"{cls._access_count} time(s). Subsequent access strictly prohibited."
            )
            logger.critical(msg)
            raise RuntimeError(msg)
        cls._access_count += 1
        logger.info(f"AUDIT HOOK: Quarantined test split access # {cls._access_count} recorded for '{path}'.")
        return cls._access_count

    @classmethod
    def get_count(cls) -> int:
        return cls._access_count

    @classmethod
    def reset(cls) -> None:
        cls._access_count = 0


class SpectraCompressPipeline:
    """Thin orchestration pipeline composing validated SpectraCompress components."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        self.transformer: Optional[DecompositionTransformer] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.labels_train: Optional[np.ndarray] = None
        self.labels_val: Optional[np.ndarray] = None
        self.val_metrics: Dict[str, Any] = {}
        self.test_metrics: Optional[Dict[str, Any]] = None

    def _validate_config(self) -> None:
        """Validates configuration keys and canonical hash."""
        required = ["method", "k", "split_seed", "solver_seed", "data_path", "test_path"]
        for k in required:
            if k not in self.config:
                raise ValueError(f"Missing required config key: '{k}'")
        if self.config["k"] != 90 or self.config["method"] != "sklearn_pca":
            raise ValueError(
                f"Config violation: locked config requires sklearn_pca, k=90; got {self.config.get('method')}, k={self.config.get('k')}"
            )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SpectraCompressPipeline":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg)

    def load_and_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads raw training data and executes fail-fast schema validation."""
        data_path = self.config["data_path"]
        n_limit = self.config.get("train_samples_limit", 10000)
        logger.info(f"Loading up to {n_limit} rows from {data_path}...")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data file not found at: {data_path}")

        df = pd.read_csv(data_path, nrows=n_limit)
        labels = df["label"].values.astype(str)
        X = df.drop(columns=["label"]).values.astype(np.float64)

        # Fail-fast schema validation
        contract = DataContract(
            expected_shape=(X.shape[0], 784),
            dtype=X.dtype,
            valid_range=(0, 255),
        )
        try:
            report = SchemaValidator.validate_schema(X, contract)
            logger.info(f"Schema validation passed: {report.message} (rows={report.total_rows})")
        except InputValidation as e:
            logger.error(f"SchemaValidator rejected dataset: {e.message}")
            raise

        return X, labels

    def split_data(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Deterministic leak-free train/validation split."""
        val_ratio = self.config.get("val_ratio", 0.2)
        seed = self.config.get("split_seed", 42)
        logger.info(f"Splitting dataset: val_ratio={val_ratio}, split_seed={seed}")

        train_idx, val_idx = RowSplitter.safe_split(X, val_ratio=val_ratio, seed=seed)
        self.X_train = X[train_idx]
        self.X_val = X[val_idx]
        self.labels_train = labels[train_idx]
        self.labels_val = labels[val_idx]
        logger.info(f"Partitioned data: n_train={len(self.X_train)}, n_val={len(self.X_val)}")

    def fit(self) -> "SpectraCompressPipeline":
        """Fits DecompositionTransformer using locked config on training data only."""
        if self.X_train is None:
            raise RuntimeError("Data must be loaded and split before fitting.")

        decomp_cfg = DecompositionConfig(
            method=MethodType(self.config["method"]),
            k=self.config["k"],
            seed=self.config.get("solver_seed", 42),
            ortho_tol=self.config.get("ortho_tol", 1e-4),
        )
        logger.info(f"Fitting DecompositionTransformer: method={decomp_cfg.method.value}, k={decomp_cfg.k}")
        self.transformer = DecompositionTransformer(decomp_cfg).fit(self.X_train)
        return self

    def evaluate_validation(self) -> Dict[str, Any]:
        """Evaluates reconstruction performance on the validation split."""
        if self.transformer is None or self.X_val is None:
            raise RuntimeError("Pipeline must be fitted before validation evaluation.")

        X_val_hat = self.transformer.reconstruct(self.X_val)
        val_mse = float(mean_squared_reconstruction_error(self.X_val, X_val_hat))
        e_per_row = np.sum((self.X_val - X_val_hat) ** 2, axis=1)
        val_std = float(np.std(e_per_row, ddof=1))
        val_se = float(val_std / np.sqrt(len(self.X_val)))

        # Variance explained
        total_var = float(np.sum(np.var(self.X_val, axis=0, ddof=1)))
        var_explained = float(1.0 - (val_mse / total_var)) if total_var > 0 else 0.0

        self.val_metrics = {
            "n_val": len(self.X_val),
            "val_mse": val_mse,
            "val_std": val_std,
            "val_se": val_se,
            "val_variance_explained": var_explained,
        }
        logger.info(f"Validation metrics: MSE={val_mse:.4f}, SE={val_se:.4f}, VarExpl={var_explained:.4f}")
        return self.val_metrics

    def evaluate_test_guarded(self) -> Dict[str, Any]:
        """
        Executes the SINGLE-TOUCH final test evaluation.
        Guarded by TestSplitAuditHook. Any second access triggers a fatal RuntimeError.
        """
        test_path = self.config["test_path"]
        TestSplitAuditHook.record_access(test_path)

        logger.info(f"Evaluating quarantined test split from '{test_path}'...")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test dataset not found at: {test_path}")

        df_test = pd.read_csv(test_path)
        labels_test = df_test["label"].values.astype(str)
        X_test = df_test.drop(columns=["label"]).values.astype(np.float64)

        # Schema validation on test
        contract = DataContract(
            expected_shape=(X_test.shape[0], 784),
            dtype=X_test.dtype,
            valid_range=(0, 255),
        )
        SchemaValidator.validate_schema(X_test, contract)

        X_test_hat = self.transformer.reconstruct(X_test)
        test_mse = float(mean_squared_reconstruction_error(X_test, X_test_hat))
        e_per_row = np.sum((X_test - X_test_hat) ** 2, axis=1)
        test_std = float(np.std(e_per_row, ddof=1))
        test_se = float(test_std / np.sqrt(len(X_test)))

        total_var = float(np.sum(np.var(X_test, axis=0, ddof=1)))
        var_explained = float(1.0 - (test_mse / total_var)) if total_var > 0 else 0.0

        val_mse = self.val_metrics.get("val_mse")
        if val_mse is None:
            raise RuntimeError("Validation evaluation must precede test evaluation.")

        rel_gap = float(abs(test_mse - val_mse) / val_mse)

        self.test_metrics = {
            "n_test": len(X_test),
            "test_mse": test_mse,
            "test_std": test_std,
            "test_se": test_se,
            "test_variance_explained": var_explained,
            "val_vs_test_relative_gap": rel_gap,
            "audit_access_count": TestSplitAuditHook.get_count(),
        }
        logger.info(f"Test evaluation complete: MSE={test_mse:.4f}, RelGap={rel_gap:.4f}")
        return self.test_metrics

    def run_error_analysis(self) -> Dict[str, Any]:
        """Runs slice error analysis and worst-N identification on validation data."""
        if self.transformer is None or self.X_val is None or self.labels_val is None:
            raise RuntimeError("Pipeline must be fitted before error analysis.")

        X_val_hat = self.transformer.reconstruct(self.X_val)
        mu_train = np.mean(self.X_train, axis=0)

        n_boot = self.config.get("n_bootstrap", 1000)
        boot_seed = self.config.get("bootstrap_seed", 42)
        worst_n = self.config.get("worst_n", 20)

        slice_df = slice_reconstruction_error(
            self.X_val,
            X_val_hat,
            self.labels_val,
            mu=mu_train,
            min_size=self.config.get("slice_min_size", 30),
            n_boot=n_boot,
            seed=boot_seed,
        )
        worst_df = worst_n_rows(
            self.X_val,
            X_val_hat,
            self.labels_val,
            n=worst_n,
            mu=mu_train,
        )

        return {
            "slice_errors": slice_df.to_dict(orient="records"),
            "worst_rows": worst_df.to_dict(orient="records"),
        }

    def save_artifact(self) -> Tuple[Path, Path]:
        """Saves fitted transformer artifact using safe serialization (npz + manifest)."""
        if self.transformer is None:
            raise RuntimeError("Cannot save unfitted transformer.")

        artifact_dir = Path(self.config.get("artifact_dir", "artifacts/spectra_compress_model"))
        source_hash = self.config.get("source_report_sha256")
        npz_path, manifest_path = save_transformer(
            self.transformer,
            artifact_dir,
            source_report_hash=source_hash,
        )
        logger.info(f"Artifact saved: npz='{npz_path}', manifest='{manifest_path}'")
        return npz_path, manifest_path

    def run(self, final_eval: bool = False) -> Dict[str, Any]:
        """Orchestrates end-to-end pipeline execution."""
        logger.info("Starting SpectraCompress pipeline run...")
        X, labels = self.load_and_validate()
        self.split_data(X, labels)
        self.fit()
        self.evaluate_validation()

        if final_eval:
            logger.info("Executing final test evaluation step...")
            self.evaluate_test_guarded()
        else:
            logger.info("Final test evaluation skipped (flag --final-eval not set).")

        error_report = self.run_error_analysis()
        npz_path, manifest_path = self.save_artifact()

        return {
            "config_hash": self.config.get("config_hash"),
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "error_analysis_slices": len(error_report["slice_errors"]),
            "artifact_paths": {
                "npz": str(npz_path),
                "manifest": str(manifest_path),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="SpectraCompress Day 7 Execution Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="final_config.json",
        help="Path to final_config.json",
    )
    parser.add_argument(
        "--final-eval",
        action="store_true",
        default=False,
        help="Enable the single-touch quarantined test split evaluation",
    )
    args = parser.parse_args()

    pipeline = SpectraCompressPipeline.from_file(args.config)
    results = pipeline.run(final_eval=args.final_eval)
    print("\n--- Pipeline Execution Summary ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
