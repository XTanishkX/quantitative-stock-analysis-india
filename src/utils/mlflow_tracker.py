"""
================================================================================
PHASE 7: MLflow Experiment Tracking Utilities
================================================================================
Provides:
    - @mlflow_experiment decorator for automatic metric logging
    - SHAPLogger for feature importance tracking
    - CPCVMetricsLogger for backtest result logging
================================================================================
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("MLflowTracker")

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Tracking disabled.")


def mlflow_experiment(
    experiment_name: str,
    tags: Optional[dict] = None,
    log_params: Optional[dict] = None,
):
    """
    Decorator: automatically starts an MLflow run, logs parameters,
    metrics, and artifacts returned by the wrapped function.

    Usage:
        @mlflow_experiment("CPCV_Backtest", tags={"strategy": "momentum"})
        def run_backtest(alpha, returns):
            result = ...
            return {
                "metrics": {"sharpe": 1.24, "max_dd": -0.08, "pbo": 0.33},
                "params":  {"n_splits": 6, "cvar_alpha": 0.95},
                "artifacts": ["/tmp/shap_plot.png"],
            }
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not _MLFLOW_AVAILABLE:
                return func(*args, **kwargs)

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(tags=tags or {}):
                # Log pre-specified params
                if log_params:
                    mlflow.log_params(log_params)

                result = func(*args, **kwargs)

                # Expect result dict with optional keys: metrics, params, artifacts
                if isinstance(result, dict):
                    if "metrics" in result:
                        mlflow.log_metrics(result["metrics"])
                        logger.info(f"[MLflow] Logged metrics: {result['metrics']}")
                    if "params" in result:
                        mlflow.log_params(result["params"])
                    if "artifacts" in result:
                        for artifact_path in result["artifacts"]:
                            mlflow.log_artifact(artifact_path)

                return result
        return wrapper
    return decorator


class CPCVMetricsLogger:
    """
    Log CPCV backtest results to MLflow with a structured schema.
    """

    def log(self, cpcv_result: Any, run_name: str = "CPCV_Backtest") -> None:
        if not _MLFLOW_AVAILABLE:
            return

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics({
                "mean_sharpe":      cpcv_result.mean_sharpe,
                "std_sharpe":       cpcv_result.std_sharpe,
                "pbo_probability":  cpcv_result.pbo_probability,
                "deflated_sharpe":  cpcv_result.deflated_sharpe,
                "n_paths":          cpcv_result.n_paths,
                "max_sharpe":       float(np.max(cpcv_result.sharpe_distribution)),
                "min_sharpe":       float(np.min(cpcv_result.sharpe_distribution)),
                "pct_positive":     float(np.mean(cpcv_result.sharpe_distribution > 0)),
            })

            # Log Sharpe distribution as a CSV artifact
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write("path,sharpe\n")
                for i, sr in enumerate(cpcv_result.sharpe_distribution):
                    f.write(f"{i},{sr:.6f}\n")
                tmp_path = f.name

            mlflow.log_artifact(tmp_path, artifact_path="cpcv_sharpe_distribution")
            os.unlink(tmp_path)
            logger.info(f"[MLflow] CPCV results logged: Mean Sharpe={cpcv_result.mean_sharpe:.3f}, PBO={cpcv_result.pbo_probability:.1%}")


class SHAPLogger:
    """
    Log SHAP feature importance plots to MLflow.
    Requires shap library.
    """

    def log_shap_summary(self, model: Any, X: np.ndarray, feature_names: list[str]) -> None:
        try:
            import shap
            import matplotlib.pyplot as plt
            import tempfile, os

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches="tight")
                tmp_path = f.name
            plt.close()

            if _MLFLOW_AVAILABLE:
                mlflow.log_artifact(tmp_path, artifact_path="shap_plots")
            os.unlink(tmp_path)
            logger.info("[MLflow] SHAP summary plot logged.")
        except ImportError:
            logger.warning("SHAP not installed. Skipping SHAP logging.")