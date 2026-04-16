"""
================================================================================
PHASE 5: PORTFOLIO CONSTRUCTION & RISK DECOMPOSITION
================================================================================
Modules:
    1. CovarianceDenoiser   — RMT + Marchenko-Pastur + Ledoit-Wolf shrinkage
    2. RiskFactorDecomposer — Barra-style systematic / idiosyncratic split
    3. PortfolioOptimizer   — CVXPY: Maximize E[R] - λ·CVaR(α) subject to
                              max turnover, leverage, and sector neutrality

Dependencies: numpy, scipy, sklearn, cvxpy
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

logger = logging.getLogger("PortfolioOptimizer")


# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DenoisedCovResult:
    cov_empirical: np.ndarray       # Raw sample covariance (N, N)
    cov_denoised: np.ndarray        # RMT-denoised covariance (N, N)
    cov_shrunk: np.ndarray          # Ledoit-Wolf shrunk covariance (N, N)
    lambda_plus: float              # Marchenko-Pastur upper bound
    n_signal_eigenvalues: int       # Eigenvalues above λ+ (signal)
    explained_variance_ratio: float # Variance retained by signal components


@dataclass
class OptimizationResult:
    weights: np.ndarray             # (N,) optimal portfolio weights
    expected_return: float
    cvar: float
    sharpe_estimate: float
    status: str                     # CVXPY solver status
    turnover: float


# ──────────────────────────────────────────────────────────────────────────────
# 1. COVARIANCE DENOISER — RANDOM MATRIX THEORY
# ──────────────────────────────────────────────────────────────────────────────

class CovarianceDenoiser:
    """
    Denoise the empirical covariance matrix using Random Matrix Theory (RMT).

    The core insight (Marchenko & Pastur, 1967):
        For a (T × N) random matrix of i.i.d. entries, the eigenvalue distribution
        of the sample covariance matrix converges to the Marchenko-Pastur law:
            λ± = σ² (1 ± √(N/T))²

    Eigenvalues outside [λ-, λ+] carry information (signal).
    Eigenvalues inside [λ-, λ+] are pure noise → replace with their mean.

    Then apply Ledoit-Wolf analytical shrinkage for additional regularization.

    This is the industry-standard approach for covariance estimation in
    portfolios with N ≈ T (e.g., 50 stocks, 250 trading days).
    """

    def __init__(
        self,
        alpha_shrink: float = 0.5,          # Interpolation: 0=empirical, 1=fully shrunk
        ledoit_wolf: bool = True,            # Apply Ledoit-Wolf after RMT
        variance_scaling: bool = True,       # Rescale denoised matrix to preserve trace
    ) -> None:
        self.alpha_shrink = alpha_shrink
        self.use_ledoit_wolf = ledoit_wolf
        self.variance_scaling = variance_scaling

    def _marchenko_pastur_bounds(self, q: float, sigma_sq: float = 1.0) -> tuple[float, float]:
        """
        Theoretical Marchenko-Pastur eigenvalue bounds.

        Args:
            q:        Ratio N/T (number of assets / number of observations).
            sigma_sq: Variance of the random entries (typically 1.0 for correlation).

        Returns:
            (lambda_minus, lambda_plus): Lower and upper MP bounds.
        """
        lambda_plus = sigma_sq * (1.0 + np.sqrt(q)) ** 2
        lambda_minus = sigma_sq * (1.0 - np.sqrt(q)) ** 2
        return lambda_minus, lambda_plus

    def _fit_mp_sigma(
        self, eigenvalues: np.ndarray, q: float, n_bins: int = 100
    ) -> float:
        """
        Fit σ² of Marchenko-Pastur distribution to empirical eigenvalues
        via KDE-based minimum χ² fitting.

        In practice, we use the trace of the correlation matrix (= N for standardized)
        divided by N as a consistent estimator of σ².
        """
        # Simple consistent estimator: use mean of eigenvalues in noise band
        sigma_sq_init = 1.0
        lm, lp = self._marchenko_pastur_bounds(q, sigma_sq_init)
        noise_eigs = eigenvalues[(eigenvalues >= lm) & (eigenvalues <= lp)]
        if len(noise_eigs) == 0:
            return sigma_sq_init
        # Refine: σ² ≈ mean of noise eigenvalues (property of MP distribution)
        return float(np.mean(noise_eigs))

    def denoise(self, returns: np.ndarray) -> DenoisedCovResult:
        """
        Full denoising pipeline: Empirical → RMT Clip → Ledoit-Wolf Shrinkage.

        Args:
            returns: (T, N) returns matrix. T observations, N assets.

        Returns:
            DenoisedCovResult with empirical, denoised, and shrunk covariances.
        """
        T, N = returns.shape
        q = N / T  # Key ratio for Marchenko-Pastur

        # Step 1: Sample covariance + correlation
        cov_empirical = np.cov(returns, rowvar=False)            # (N, N)
        std_devs = np.sqrt(np.diag(cov_empirical))
        corr_matrix = cov_empirical / np.outer(std_devs, std_devs)  # (N, N) correlation

        # Step 2: Eigendecomposition of correlation matrix
        # eigh guarantees real eigenvalues and is faster for symmetric matrices
        eigenvalues, eigenvectors = eigh(corr_matrix)            # ascending order
        eigenvalues = np.maximum(eigenvalues, 0)                 # numerical stability

        # Step 3: Fit MP distribution and find signal threshold
        sigma_sq = self._fit_mp_sigma(eigenvalues, q)
        lambda_minus, lambda_plus = self._marchenko_pastur_bounds(q, sigma_sq)

        # Step 4: Separate signal and noise eigenvalues
        is_signal = eigenvalues > lambda_plus
        n_signal = int(np.sum(is_signal))
        noise_mean = np.mean(eigenvalues[~is_signal]) if np.any(~is_signal) else 1.0

        # Step 5: Replace noise eigenvalues with their mean (minimum variance)
        eigenvalues_clean = eigenvalues.copy()
        eigenvalues_clean[~is_signal] = noise_mean

        # Step 6: Reconstruct denoised correlation matrix
        corr_denoised = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T

        # Rescale to preserve original variance
        if self.variance_scaling:
            scale = N / np.trace(corr_denoised)
            corr_denoised *= scale

        # Convert back to covariance
        cov_denoised = corr_denoised * np.outer(std_devs, std_devs)

        # Step 7: Ledoit-Wolf shrinkage
        if self.use_ledoit_wolf:
            lw = LedoitWolf().fit(returns)
            cov_lw = lw.covariance_
            # Blend denoised with Ledoit-Wolf
            cov_final = (1 - self.alpha_shrink) * cov_denoised + self.alpha_shrink * cov_lw
        else:
            cov_final = cov_denoised

        # Ensure positive definiteness
        cov_final = self._ensure_psd(cov_final)

        explained_var = float(
            np.sum(eigenvalues[is_signal]) / np.sum(eigenvalues)
        ) if n_signal > 0 else 0.0

        logger.info(
            f"[CovarianceDenoiser] q={q:.2f}, λ+={lambda_plus:.4f}, "
            f"signal eigenvalues={n_signal}/{N}, "
            f"explained variance={explained_var:.1%}"
        )

        return DenoisedCovResult(
            cov_empirical=cov_empirical,
            cov_denoised=cov_denoised,
            cov_shrunk=cov_final,
            lambda_plus=lambda_plus,
            n_signal_eigenvalues=n_signal,
            explained_variance_ratio=explained_var,
        )

    @staticmethod
    def _ensure_psd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Clip negative eigenvalues to ensure positive semi-definiteness."""
        eigenvalues, eigenvectors = eigh(cov)
        eigenvalues = np.maximum(eigenvalues, eps)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


# ──────────────────────────────────────────────────────────────────────────────
# 2. RISK FACTOR DECOMPOSER — BARRA STYLE
# ──────────────────────────────────────────────────────────────────────────────

class RiskFactorDecomposer:
    """
    Barra-style risk decomposition into systematic and specific risk.

    Portfolio risk decomposition:
        σ²_p = w' Σ w
             = w' (B F B' + D) w
             = [w'B] F [B'w] + w'Dw
             = Systematic Risk + Idiosyncratic Risk

    Where:
        B: (N, K) factor loading matrix (betas)
        F: (K, K) factor covariance matrix
        D: (N, N) diagonal idiosyncratic variance matrix
    """

    def fit(
        self,
        returns: np.ndarray,          # (T, N) asset returns
        factor_returns: np.ndarray,   # (T, K) factor returns
    ) -> dict:
        """
        Estimate factor loadings via OLS time-series regression.

        For each asset i: r_it = α_i + Σ_k β_{ik} f_kt + ε_it

        Returns:
            dict with 'B' (factor loadings), 'F' (factor cov), 'D' (idio variances),
            'residuals' (T, N), and 'r_squared' per asset.
        """
        T, N = returns.shape
        K = factor_returns.shape[1]

        F_mat = np.column_stack([np.ones(T), factor_returns])  # (T, K+1) with intercept
        B_full = np.linalg.lstsq(F_mat, returns, rcond=None)[0]  # (K+1, N)

        alphas = B_full[0]       # (N,) intercepts
        B = B_full[1:]           # (K, N) factor loadings

        residuals = returns - (F_mat @ B_full)   # (T, N)
        idio_var = np.var(residuals, axis=0)     # (N,) specific variance

        factor_cov = np.cov(factor_returns, rowvar=False)  # (K, K)

        # R² per asset
        ss_res = np.sum(residuals**2, axis=0)
        ss_tot = np.sum((returns - returns.mean(axis=0))**2, axis=0)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return {
            "alphas": alphas,
            "B": B.T,                    # (N, K) — convention: N assets, K factors
            "F": factor_cov,             # (K, K)
            "D": np.diag(idio_var),      # (N, N) diagonal
            "idio_var": idio_var,        # (N,)
            "residuals": residuals,
            "r_squared": r_squared,
        }

    def decompose_portfolio_risk(
        self,
        weights: np.ndarray,
        B: np.ndarray,
        F: np.ndarray,
        D: np.ndarray,
    ) -> dict[str, float]:
        """
        Decompose total portfolio variance into systematic + idiosyncratic.

        Args:
            weights: (N,) portfolio weights.
            B: (N, K) factor loading matrix.
            F: (K, K) factor covariance.
            D: (N, N) diagonal idiosyncratic covariance.

        Returns:
            dict with systematic_var, idio_var, total_var, and their fractions.
        """
        systematic_cov = B @ F @ B.T          # (N, N)
        systematic_var = float(weights @ systematic_cov @ weights)
        idio_var = float(weights @ D @ weights)
        total_var = systematic_var + idio_var

        return {
            "systematic_var": systematic_var,
            "idiosyncratic_var": idio_var,
            "total_var": total_var,
            "systematic_fraction": systematic_var / (total_var + 1e-10),
            "idio_fraction": idio_var / (total_var + 1e-10),
            "tracking_error_annualized": np.sqrt(idio_var * 252),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. PORTFOLIO OPTIMIZER — CVXPY
# ──────────────────────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer using CVXPY.

    Objective:
        maximize  E[R_p] - λ · CVaR_α(R_p)

    CVaR (Conditional Value at Risk / Expected Shortfall) formulation
    (Rockafellar & Uryasev, 2000):

        CVaR_α = min_{γ} { γ + (1/(1-α)) · E[max(−R_p − γ, 0)] }

    This is a LINEAR program in the scenario returns — no quadratic cone needed.
    For N assets and S scenarios, the LP has N + S + 1 variables.

    Constraints:
        - Budget: Σ w_i = 1 (fully invested)
        - Long-only: w_i ≥ 0  (or allow shorting with bounds)
        - Max weight: w_i ≤ max_weight
        - Max turnover: Σ |w_i - w_prev_i| ≤ turnover_limit
        - Sector neutrality: Σ_{i ∈ sector_s} w_i = 1/n_sectors ± tolerance
        - Leverage: Σ |w_i| ≤ max_leverage

    Args:
        cvar_alpha:      CVaR confidence level (e.g., 0.95 → 95% CVaR = ES).
        risk_aversion:   λ — trade-off between return and CVaR.
        max_weight:      Maximum weight per asset (e.g., 0.1 = 10%).
        turnover_limit:  Maximum total turnover from current weights.
        max_leverage:    Maximum gross leverage (1.0 = long-only unlevered).
        long_only:       If True, all weights ≥ 0.
    """

    def __init__(
        self,
        cvar_alpha: float = 0.95,
        risk_aversion: float = 1.0,
        max_weight: float = 0.10,
        turnover_limit: float = 0.30,
        max_leverage: float = 1.0,
        long_only: bool = True,
        solver: str = "CLARABEL",
    ) -> None:
        self.cvar_alpha = cvar_alpha
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.turnover_limit = turnover_limit
        self.max_leverage = max_leverage
        self.long_only = long_only
        self.solver = solver

    def optimize(
        self,
        expected_returns: np.ndarray,        # (N,) alpha signal / expected returns
        scenario_returns: np.ndarray,        # (S, N) historical or simulated scenarios
        current_weights: Optional[np.ndarray] = None,  # (N,) for turnover constraint
        sector_map: Optional[np.ndarray] = None,       # (N,) int sector labels
        sector_tolerance: float = 0.05,
    ) -> OptimizationResult:
        """
        Solve the CVaR-based portfolio optimization problem.

        Args:
            expected_returns: Alpha forecast vector (N,).
            scenario_returns: Return scenarios matrix (S, N). Used to estimate CVaR.
            current_weights:  Current portfolio weights for turnover constraint.
            sector_map:       Array of sector indices [0, n_sectors-1] per asset.
            sector_tolerance: Allowed deviation from equal sector weights.

        Returns:
            OptimizationResult with optimal weights and risk metrics.
        """
        N = len(expected_returns)
        S = len(scenario_returns)

        # ── Decision variables ─────────────────────────────────────────────
        w = cp.Variable(N, name="weights")           # Portfolio weights
        gamma = cp.Variable(name="var_threshold")     # VaR threshold (scalar)
        # Auxiliary loss variables for CVaR linearization
        z = cp.Variable(S, nonneg=True, name="cvar_aux")

        # ── Scenario portfolio returns ─────────────────────────────────────
        port_scenarios = scenario_returns @ w         # (S,) scenario-level returns

        # ── CVaR via Rockafellar-Uryasev LP formulation ───────────────────
        # CVaR = γ + E[z_s] / (1 - α)
        # where z_s ≥ -port_return_s - γ, z_s ≥ 0
        cvar = gamma + (1.0 / ((1 - self.cvar_alpha) * S)) * cp.sum(z)

        # ── Objective: Maximize E[R_p] - λ · CVaR ────────────────────────
        expected_return = expected_returns @ w
        objective = cp.Maximize(expected_return - self.risk_aversion * cvar)

        # ── Constraints ────────────────────────────────────────────────────
        constraints = [
            # CVaR linearization: z_s ≥ -R_p_s - γ
            z >= -port_scenarios - gamma,

            # Budget constraint: fully invested
            cp.sum(w) == 1.0,

            # Leverage constraint
            cp.norm1(w) <= self.max_leverage,

            # Maximum position size
            w <= self.max_weight,
        ]

        # Long-only constraint
        if self.long_only:
            constraints.append(w >= 0.0)
        else:
            constraints.append(w >= -self.max_weight)

        # Turnover constraint
        if current_weights is not None:
            constraints.append(
                cp.norm1(w - current_weights) <= self.turnover_limit
            )

        # Sector neutrality constraints
        if sector_map is not None:
            n_sectors = int(np.max(sector_map)) + 1
            target_sector_weight = 1.0 / n_sectors
            for s in range(n_sectors):
                sector_mask = (sector_map == s)
                if np.any(sector_mask):
                    constraints += [
                        cp.sum(w[sector_mask]) <= target_sector_weight + sector_tolerance,
                        cp.sum(w[sector_mask]) >= target_sector_weight - sector_tolerance,
                    ]

        # ── Solve ──────────────────────────────────────────────────────────
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver, verbose=False)
        except cp.SolverError:
            logger.warning(f"[Optimizer] {self.solver} failed, retrying with SCS...")
            problem.solve(solver="SCS", verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"[Optimizer] Optimization failed: {problem.status}")
            fallback_w = np.ones(N) / N
            return OptimizationResult(
                weights=fallback_w, expected_return=0.0, cvar=0.0,
                sharpe_estimate=0.0, status=problem.status, turnover=0.0,
            )

        optimal_weights = w.value
        optimal_weights = np.maximum(optimal_weights, 0) if self.long_only else optimal_weights
        optimal_weights /= np.sum(np.abs(optimal_weights)) + 1e-10

        exp_ret = float(expected_returns @ optimal_weights)
        cvar_val = float(cvar.value) if cvar.value is not None else 0.0

        # Sharpe estimate from scenarios
        port_rets = scenario_returns @ optimal_weights
        sharpe = float(np.mean(port_rets) / (np.std(port_rets) + 1e-10) * np.sqrt(252))

        turnover = float(
            np.sum(np.abs(optimal_weights - current_weights))
            if current_weights is not None else 0.0
        )

        logger.info(
            f"[Optimizer] Status={problem.status} | E[R]={exp_ret:.4f} | "
            f"CVaR={cvar_val:.4f} | Sharpe≈{sharpe:.2f} | Turnover={turnover:.2%}"
        )

        return OptimizationResult(
            weights=optimal_weights,
            expected_return=exp_ret,
            cvar=cvar_val,
            sharpe_estimate=sharpe,
            status=problem.status,
            turnover=turnover,
        )

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        scenario_returns: np.ndarray,
        n_points: int = 20,
    ) -> list[OptimizationResult]:
        """
        Trace the efficient frontier by varying risk_aversion λ.

        Args:
            expected_returns: (N,) alpha signals.
            scenario_returns: (S, N) return scenarios.
            n_points:         Number of frontier points.

        Returns:
            List of OptimizationResults for increasing λ values.
        """
        lambdas = np.logspace(-2, 2, n_points)
        frontier = []
        original_lambda = self.risk_aversion

        for lam in lambdas:
            self.risk_aversion = lam
            result = self.optimize(expected_returns, scenario_returns)
            frontier.append(result)

        self.risk_aversion = original_lambda
        return frontier