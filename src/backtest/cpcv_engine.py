"""
================================================================================
PHASE 6: INSTITUTIONAL BACKTESTER — CPCV ENGINE
================================================================================
Modules:
    1. PurgedKFold            — Purged + Embargoed cross-validation splits
    2. CombinatorialPurgedCV  — CPCV (Lopez de Prado, Ch. 12)
    3. VectorizedBacktester   — Vectorized PnL engine with realistic costs

CPCV Key Concepts:
    - Purging: Remove training samples whose labels overlap with test labels
      in time (prevents leakage from overlapping return windows).
    - Embargo: Further exclude a buffer after each test period (prevents
      information bleed from microstructural correlations).
    - Combinatorial: Instead of K sequential folds, use C(K, K-2) path
      combinations → produces S backtest paths → backtest distribution,
      not just a single Sharpe ratio.

Dependencies: numpy, pandas, scipy, numba (optional for speedup)
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger("CPCVEngine")


# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Full backtesting result for a single strategy path."""
    returns: np.ndarray             # Strategy returns series
    cumulative_pnl: np.ndarray      # Cumulative PnL
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float                 # Fraction of profitable periods
    avg_win: float
    avg_loss: float
    profit_factor: float            # Gross profit / Gross loss
    turnover: float                 # Average 2-way turnover per period
    total_commission: float


@dataclass
class CPCVResult:
    """Aggregated CPCV backtest results across all combinatorial paths."""
    path_results: list[BacktestResult]
    sharpe_distribution: np.ndarray  # Distribution of Sharpe ratios
    mean_sharpe: float
    std_sharpe: float
    pbo_probability: float           # Probability of Backtest Overfitting
    deflated_sharpe: float           # Deflated Sharpe Ratio (Bailey et al.)
    n_paths: int


# ──────────────────────────────────────────────────────────────────────────────
# 1. PURGED K-FOLD CROSS-VALIDATOR
# ──────────────────────────────────────────────────────────────────────────────

class PurgedKFold:
    """
    Purged K-Fold cross-validation for financial time series.

    Standard K-Fold leaks information when:
        1. Labels overlap in time (e.g., 5-day return overlaps across folds).
        2. Autocorrelation of features transfers information across boundaries.

    Solution (Lopez de Prado, 2018, Chapter 7):
        - Purge: Remove training observations whose label spans overlap with
                 any test observation's label span.
        - Embargo: Remove the first `embargo_pct` fraction of training obs
                   immediately after each test fold.

    Args:
        n_splits:     Number of folds K.
        embargo_pct:  Fraction of fold to embargo after test set.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray,
        pred_times: pd.Series,
        eval_times: pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged and embargoed train/test splits.

        Args:
            X:           Feature matrix (N, F). Used only for shape.
            pred_times:  pd.Series of prediction timestamps (index start time).
            eval_times:  pd.Series of evaluation timestamps (label end time).
                         eval_times[i] > pred_times[i] by at least the holding period.

        Yields:
            (train_idx, test_idx) pairs of integer index arrays.
        """
        n = len(X)
        embargo_size = max(1, int(n * self.embargo_pct))
        fold_size = n // self.n_splits
        indices = np.arange(n)

        for fold in range(self.n_splits):
            # Test set indices
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            # Test time window
            test_pred_start = pred_times.iloc[test_start]
            test_eval_end = eval_times.iloc[test_end - 1]

            # Training indices: exclude test fold
            train_candidates = np.concatenate([
                indices[:test_start],
                indices[test_end:],
            ])

            # Purge: remove training samples whose evaluation overlaps with test prediction window
            purge_mask = np.array([
                eval_times.iloc[i] < test_pred_start or pred_times.iloc[i] > test_eval_end
                for i in train_candidates
            ])
            train_purged = train_candidates[purge_mask]

            # Embargo: remove samples immediately after the test fold
            embargo_start = test_end
            embargo_end = min(test_end + embargo_size, n)
            embargo_idx = set(range(embargo_start, embargo_end))
            train_final = np.array([i for i in train_purged if i not in embargo_idx])

            if len(train_final) == 0:
                logger.warning(f"[PurgedKFold] Fold {fold}: empty training set after purging.")
                continue

            logger.debug(
                f"[PurgedKFold] Fold {fold}: "
                f"train={len(train_final)}, test={len(test_idx)}, "
                f"purged={len(train_candidates)-len(train_purged)}, "
                f"embargoed={len(embargo_idx)}"
            )
            yield train_final, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# 2. COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# ──────────────────────────────────────────────────────────────────────────────

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (Lopez de Prado, 2018, Chapter 12).

    Standard walk-forward backtesting produces ONE backtest path → ONE Sharpe.
    This single number is highly overfit to the specific data realization.

    CPCV produces a DISTRIBUTION of Sharpe ratios:
        1. Split data into K groups (chronological).
        2. For each combination C(K, K-φ): use (K-φ) groups for training, φ for test.
        3. Concatenate the φ test-fold predictions across all combinations
           → produces multiple non-overlapping backtest paths.
        4. The distribution of Sharpe ratios is used to estimate:
           - P(Backtest Overfitting)  [PBO]
           - Deflated Sharpe Ratio    [DSR]

    Args:
        n_splits:     K — number of groups.
        n_test_splits: φ — number of groups used for testing.
        embargo_pct:  Embargo fraction between train and test.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

        # Number of backtest paths = C(K, φ)
        from math import comb
        self.n_paths = comb(n_splits, n_test_splits)
        logger.info(
            f"[CPCV] K={n_splits}, φ={n_test_splits} → "
            f"{self.n_paths} backtest paths (C({n_splits},{n_test_splits}))"
        )

    def get_splits(
        self,
        n_samples: int,
        pred_times: Optional[pd.Series] = None,
        eval_times: Optional[pd.Series] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate all C(K, φ) train/test splits.

        Args:
            n_samples:   Total number of observations.
            pred_times:  Optional pd.Series of prediction times for purging.
            eval_times:  Optional pd.Series of evaluation times for purging.

        Returns:
            List of (train_idx, test_idx) tuples.
        """
        indices = np.arange(n_samples)
        group_size = n_samples // self.n_splits
        embargo_size = max(1, int(group_size * self.embargo_pct))

        # Assign each sample to a group
        groups = [
            indices[k * group_size: (k + 1) * group_size if k < self.n_splits - 1 else n_samples]
            for k in range(self.n_splits)
        ]

        splits = []
        for test_group_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_group_set = set(test_group_combo)
            train_groups = [groups[k] for k in range(self.n_splits) if k not in test_group_set]
            test_groups = [groups[k] for k in sorted(test_group_combo)]

            train_idx = np.concatenate(train_groups) if train_groups else np.array([], dtype=int)
            test_idx = np.concatenate(test_groups)

            # Apply embargo after each test group (remove first embargo_size from training)
            for test_group_k in test_group_combo:
                embargo_start = groups[test_group_k][-1] + 1
                embargo_end = min(embargo_start + embargo_size, n_samples)
                embargo_range = set(range(embargo_start, embargo_end))
                train_idx = np.array([i for i in train_idx if i not in embargo_range])

            # Apply purging if time information available
            if pred_times is not None and eval_times is not None:
                train_idx = self._purge(
                    train_idx, test_idx, pred_times, eval_times
                )

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, np.sort(test_idx)))

        return splits

    def _purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        pred_times: pd.Series,
        eval_times: pd.Series,
    ) -> np.ndarray:
        """Remove training samples whose label spans overlap with test prediction window."""
        if len(test_idx) == 0:
            return train_idx

        test_pred_start = pred_times.iloc[test_idx[0]]
        test_eval_end = eval_times.iloc[test_idx[-1]]

        keep = [
            i for i in train_idx
            if eval_times.iloc[i] <= test_pred_start or pred_times.iloc[i] >= test_eval_end
        ]
        return np.array(keep)


# ──────────────────────────────────────────────────────────────────────────────
# 3. VECTORIZED BACKTESTER
# ──────────────────────────────────────────────────────────────────────────────

class VectorizedBacktester:
    """
    High-performance vectorized backtesting engine.

    Features:
        - Commission: Fixed bps per trade.
        - Slippage: Volume-proportional market impact (Almgren-Chriss simplified).
        - Position limits: Max gross exposure.
        - Vectorized PnL computation (no Python loops over time steps).
        - Tearsheet metrics: Sharpe, Sortino, Max DD, Calmar, PF.

    Args:
        commission_bps:    Round-trip commission in basis points.
        slippage_bps:      Market impact slippage per unit of ADV traded.
        max_gross_exposure: Max |w|.sum() — leverage limit.
        annual_factor:     Periods per year (252 for daily, 52 for weekly, 12 monthly).
    """

    def __init__(
        self,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        max_gross_exposure: float = 1.5,
        annual_factor: int = 252,
    ) -> None:
        self.commission_bps = commission_bps / 10_000
        self.slippage_bps = slippage_bps / 10_000
        self.max_gross_exposure = max_gross_exposure
        self.annual_factor = annual_factor

    def run(
        self,
        weights: np.ndarray,          # (T, N) portfolio weights
        returns: np.ndarray,          # (T, N) asset returns
        volumes: Optional[np.ndarray] = None,  # (T, N) volume for slippage
        initial_capital: float = 10_000_000.0,
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Args:
            weights:  (T, N) target portfolio weights (pre-computed by optimizer).
            returns:  (T, N) realized asset returns.
            volumes:  (T, N) relative volumes for slippage scaling (optional).
            initial_capital: Starting NAV in INR.

        Returns:
            BacktestResult with full tearsheet metrics.
        """
        T, N = returns.shape
        assert weights.shape == (T, N), f"Weight shape mismatch: {weights.shape} vs ({T},{N})"

        # Enforce exposure limit
        gross = np.abs(weights).sum(axis=1, keepdims=True)
        scale = np.where(gross > self.max_gross_exposure, self.max_gross_exposure / gross, 1.0)
        weights = weights * scale

        # ── Turnover & Transaction Costs ──────────────────────────────────
        weight_changes = np.diff(weights, axis=0, prepend=weights[[0]] * 0)
        turnover = np.abs(weight_changes).sum(axis=1)  # (T,) 2-way turnover

        commission = turnover * self.commission_bps

        # Volume-proportional slippage: larger trades relative to volume → more impact
        if volumes is not None:
            vol_ratio = np.abs(weight_changes) / (volumes + 1e-10)
            slippage = (vol_ratio * np.abs(weight_changes)).sum(axis=1) * self.slippage_bps
        else:
            slippage = turnover * self.slippage_bps

        total_costs = commission + slippage

        # ── Gross Strategy Returns ────────────────────────────────────────
        # Use lagged weights (yesterday's weights applied to today's returns)
        lagged_weights = np.roll(weights, 1, axis=0)
        lagged_weights[0] = 0.0  # No position on day 0

        gross_returns = (lagged_weights * returns).sum(axis=1)    # (T,)
        net_returns = gross_returns - total_costs

        # ── Tearsheet Metrics ─────────────────────────────────────────────
        cum_pnl = initial_capital * np.cumprod(1 + net_returns)

        result = self._compute_metrics(
            net_returns=net_returns,
            cum_pnl=cum_pnl,
            turnover=turnover,
            total_commission=(commission.sum() + slippage.sum()) * initial_capital,
        )
        return result

    def _compute_metrics(
        self,
        net_returns: np.ndarray,
        cum_pnl: np.ndarray,
        turnover: np.ndarray,
        total_commission: float,
    ) -> BacktestResult:
        """Compute all tearsheet performance metrics from return series."""
        af = self.annual_factor

        # Annualized Sharpe
        mean_ret = np.mean(net_returns)
        std_ret = np.std(net_returns) + 1e-10
        sharpe = float(mean_ret / std_ret * np.sqrt(af))

        # Sortino (downside deviation only)
        downside = net_returns[net_returns < 0]
        downside_std = np.std(downside) + 1e-10 if len(downside) > 1 else std_ret
        sortino = float(mean_ret / downside_std * np.sqrt(af))

        # Max Drawdown
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = (cum_pnl - running_max) / running_max
        max_drawdown = float(drawdowns.min())

        # Calmar Ratio: Annualized Return / |Max Drawdown|
        ann_return = float((1 + mean_ret) ** af - 1)
        calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

        # Win/Loss metrics
        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        hit_rate = float(len(wins) / (len(net_returns) + 1e-10))
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        gross_profit = np.sum(wins)
        gross_loss = abs(np.sum(losses)) + 1e-10
        profit_factor = float(gross_profit / gross_loss)

        return BacktestResult(
            returns=net_returns,
            cumulative_pnl=cum_pnl,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            hit_rate=hit_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            turnover=float(np.mean(turnover)),
            total_commission=total_commission,
        )

    def run_cpcv(
        self,
        alpha_signals: np.ndarray,    # (T, N) raw alpha signals
        returns: np.ndarray,           # (T, N) realized returns
        cpcv: CombinatorialPurgedCV,
        optimizer_fn,                  # Callable: (alpha, scenario_returns) → weights (N,)
        pred_times: Optional[pd.Series] = None,
        eval_times: Optional[pd.Series] = None,
    ) -> CPCVResult:
        """
        Run full CPCV backtest: train optimizer on each split, collect OOS returns.

        Args:
            alpha_signals: (T, N) — alpha forecasts aligned with returns.
            returns:       (T, N) — realized returns.
            cpcv:          Configured CombinatorialPurgedCV instance.
            optimizer_fn:  A callable that takes (alpha_signals, scenario_returns)
                           and returns portfolio weights (N,).
            pred_times:    For purging.
            eval_times:    For purging.

        Returns:
            CPCVResult with Sharpe distribution and overfitting diagnostics.
        """
        T, N = returns.shape
        splits = cpcv.get_splits(T, pred_times, eval_times)
        path_results: list[BacktestResult] = []
        sharpe_values: list[float] = []

        logger.info(f"[CPCV] Running {len(splits)} backtest paths...")

        for path_idx, (train_idx, test_idx) in enumerate(splits):
            # Train: fit optimizer parameters on training data
            train_alpha = alpha_signals[train_idx]
            train_returns = returns[train_idx]
            test_alpha = alpha_signals[test_idx]
            test_returns = returns[test_idx]

            # Construct OOS weights using the optimizer fitted on train
            test_weights = np.zeros((len(test_idx), N))
            for t in range(len(test_idx)):
                # Use train scenario returns for CVaR estimation
                try:
                    w = optimizer_fn(test_alpha[t], train_returns)
                    test_weights[t] = w
                except Exception as e:
                    logger.warning(f"  [CPCV] Path {path_idx}, step {t}: optimizer failed ({e}). Using equal weight.")
                    test_weights[t] = np.ones(N) / N

            # Backtest OOS period
            result = self.run(test_weights, test_returns)
            path_results.append(result)
            sharpe_values.append(result.sharpe_ratio)

            logger.info(
                f"  [CPCV] Path {path_idx+1:3d}/{len(splits)} | "
                f"Sharpe={result.sharpe_ratio:.3f} | "
                f"MaxDD={result.max_drawdown:.2%} | "
                f"Calmar={result.calmar_ratio:.2f}"
            )

        sharpe_arr = np.array(sharpe_values)
        pbo = self._compute_pbo(sharpe_arr)
        dsr = self._deflated_sharpe_ratio(sharpe_arr)

        logger.info(
            f"[CPCV] Results: Mean Sharpe={np.mean(sharpe_arr):.3f} ± {np.std(sharpe_arr):.3f} | "
            f"PBO={pbo:.1%} | DSR={dsr:.3f}"
        )

        return CPCVResult(
            path_results=path_results,
            sharpe_distribution=sharpe_arr,
            mean_sharpe=float(np.mean(sharpe_arr)),
            std_sharpe=float(np.std(sharpe_arr)),
            pbo_probability=pbo,
            deflated_sharpe=dsr,
            n_paths=len(splits),
        )

    @staticmethod
    def _compute_pbo(sharpe_distribution: np.ndarray) -> float:
        """
        Probability of Backtest Overfitting (Bailey et al., 2014).

        PBO = P(OOS Sharpe < median(IS Sharpe))
            ≈ fraction of backtest paths with negative Sharpe.

        A high PBO (>0.5) indicates the strategy is likely overfit.
        """
        median_sharpe = np.median(sharpe_distribution)
        return float(np.mean(sharpe_distribution < median_sharpe))

    @staticmethod
    def _deflated_sharpe_ratio(
        sharpe_distribution: np.ndarray,
        annual_factor: int = 252,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

        Adjusts the maximum observed Sharpe for the selection bias introduced
        by testing multiple strategy configurations:

        DSR = PSR(SR* - E[max SR under H0]) where PSR is the Probabilistic SR.

        Simplified estimator:
            E[max SR_n] ≈ ((1 - γ)Z^{-1}(1 - 1/n) + γZ^{-1}(1 - 1/(n·e))) * σ̂_sr
        """
        n = len(sharpe_distribution)
        if n < 2:
            return float(sharpe_distribution[0]) if len(sharpe_distribution) > 0 else 0.0

        sr_max = np.max(sharpe_distribution)
        sr_mean = np.mean(sharpe_distribution)
        sr_std = np.std(sharpe_distribution) + 1e-10

        # Euler-Mascheroni constant γ ≈ 0.5772
        gamma_em = 0.5772
        expected_max = sr_std * (
            (1 - gamma_em) * norm.ppf(1 - 1.0 / n) +
            gamma_em * norm.ppf(1 - 1.0 / (n * np.e))
        )

        # Probabilistic Sharpe Ratio adjustment
        dsr = norm.cdf((sr_max - expected_max) / (sr_std + 1e-10))
        return float(dsr)