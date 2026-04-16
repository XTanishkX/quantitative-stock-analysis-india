"""
================================================================================
PHASE 2: THE ALPHA FACTORY — ADVANCED FEATURE ENGINEERING
================================================================================
Modules:
    1. StationarityEngine    — Fractional Differencing + ADF pipeline
    2. MicrostructureEngine  — VPIN, Amihud Illiquidity, Roll's Impact Measure
    3. StatArbEngine         — Ornstein-Uhlenbeck estimator + Johansen Cointegration
    4. InformationTheoryEngine — Mutual Information + Transfer Entropy

Dependencies: polars, numpy, scipy, statsmodels, scikit-learn
================================================================================
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")
logger = logging.getLogger("AlphaFactory")


# ──────────────────────────────────────────────────────────────────────────────
# BASE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class BaseEngine(ABC):
    """Abstract base for all Alpha Factory sub-engines."""

    @abstractmethod
    def compute(self, *args, **kwargs): ...

    def _validate_array(self, arr: np.ndarray, name: str = "input") -> None:
        if arr is None or len(arr) == 0:
            raise ValueError(f"'{name}' must be non-empty.")
        if not np.isfinite(arr).all():
            raise ValueError(f"'{name}' contains NaN/Inf. Clean data first.")


# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ADFResult:
    statistic: float
    p_value: float
    n_lags: int
    is_stationary: bool
    significance_level: float = 0.05

    def __repr__(self) -> str:
        verdict = "STATIONARY" if self.is_stationary else "NON-STATIONARY"
        return (f"ADFResult(stat={self.statistic:.4f}, p={self.p_value:.4f}, "
                f"lags={self.n_lags}, verdict={verdict})")


@dataclass
class FracDiffResult:
    d_optimal: float
    series_fracdiff: np.ndarray
    adf_results: list[ADFResult] = field(default_factory=list)


@dataclass
class VPINResult:
    vpin_series: np.ndarray       # Rolling VPIN per bucket
    buy_volume_fraction: np.ndarray
    total_buckets: int
    mean_vpin: float
    std_vpin: float


@dataclass
class OUResult:
    mu: float           # Long-run mean
    theta: float        # Mean-reversion speed
    sigma: float        # Volatility
    half_life: float    # ln(2) / theta  [in bar units]
    residuals: np.ndarray


@dataclass
class JohansenResult:
    is_cointegrated: bool
    n_cointegrating_vectors: int
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    trace_stats: np.ndarray
    critical_values_90: np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# 1. STATIONARITY ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class StationarityEngine(BaseEngine):
    """
    Implements Fractional Differencing (Lopez de Prado, Chapter 5) and
    Augmented Dickey-Fuller stationarity testing.

    Fractional differencing preserves long-memory in financial series while
    achieving stationarity — a critical property lost by integer differencing.

    The weight vector for fractional differencing of order d is:
        w_k = prod_{i=0}^{k-1} (d - i) / (i + 1)   for k >= 1,  w_0 = 1

    We truncate weights when |w_k| < threshold (memory-efficient fixed-width window).
    """

    def __init__(
        self,
        d_range: tuple[float, float] = (0.0, 1.0),
        d_step: float = 0.05,
        adf_significance: float = 0.05,
        weight_threshold: float = 1e-5,
    ) -> None:
        self.d_range = d_range
        self.d_step = d_step
        self.adf_significance = adf_significance
        self.weight_threshold = weight_threshold

    # ── Core weight generator ─────────────────────────────────────────────────

    def _get_weights(self, d: float, size: int) -> np.ndarray:
        """
        Compute the binomial series coefficients for fractional differencing.

        Args:
            d:    Differencing order in (0, 1).
            size: Maximum number of lags to compute.

        Returns:
            weights: Array of shape (size,), weights[0] = 1.
        """
        weights = [1.0]
        for k in range(1, size):
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < self.weight_threshold:
                break
            weights.append(w)
        return np.array(weights[::-1])  # oldest lag first

    def _fracdiff_fixed_window(
        self, series: np.ndarray, d: float
    ) -> np.ndarray:
        """
        Apply fixed-width fractional differencing (memory-preserving).

        Skips early observations where the full weight window isn't available,
        preventing look-ahead bias.

        Args:
            series: Raw price series (log prices recommended).
            d:      Differencing order.

        Returns:
            fd_series: Fractionally differenced series, NaN-padded at start.
        """
        weights = self._get_weights(d, len(series))
        width = len(weights)
        output = np.full(len(series), np.nan)

        for t in range(width - 1, len(series)):
            window = series[t - width + 1 : t + 1]
            output[t] = float(np.dot(weights, window))

        return output

    # ── ADF Test ──────────────────────────────────────────────────────────────

    def run_adf(self, series: np.ndarray) -> ADFResult:
        """
        Run the Augmented Dickey-Fuller test on a series.

        H0: Unit root present (non-stationary)
        H1: No unit root (stationary)

        Args:
            series: 1D time series array (NaN values are dropped).

        Returns:
            ADFResult dataclass with test stats and stationarity verdict.
        """
        clean = series[~np.isnan(series)]
        result = adfuller(clean, autolag="AIC")
        stat, pval, lags = result[0], result[1], result[2]
        return ADFResult(
            statistic=float(stat),
            p_value=float(pval),
            n_lags=int(lags),
            is_stationary=pval < self.adf_significance,
            significance_level=self.adf_significance,
        )

    # ── Optimal d Search ─────────────────────────────────────────────────────

    def compute(self, log_prices: np.ndarray) -> FracDiffResult:
        """
        Find the minimum d in [d_min, d_max] such that the fractionally
        differenced series passes the ADF stationarity test.

        This is the Lopez de Prado (2018) "minimum d" approach — we retain
        maximum memory while guaranteeing stationarity.

        Args:
            log_prices: Log-price series (apply np.log to raw prices first).

        Returns:
            FracDiffResult with optimal d, the differenced series, and ADF trail.
        """
        self._validate_array(log_prices, "log_prices")
        d_values = np.arange(self.d_range[0], self.d_range[1] + self.d_step, self.d_step)
        adf_trail: list[ADFResult] = []
        d_optimal: float = self.d_range[1]
        series_optimal: np.ndarray = log_prices.copy()

        for d in d_values:
            fd = self._fracdiff_fixed_window(log_prices, d)
            adf_res = self.run_adf(fd)
            adf_trail.append(adf_res)
            logger.debug(f"  d={d:.2f} → {adf_res}")

            if adf_res.is_stationary:
                d_optimal = d
                series_optimal = fd
                logger.info(f"[StationarityEngine] Minimum stationary d = {d:.2f}")
                break

        return FracDiffResult(
            d_optimal=d_optimal,
            series_fracdiff=series_optimal,
            adf_results=adf_trail,
        )

    def batch_compute(self, price_frame: pl.DataFrame, log_transform: bool = True) -> pl.DataFrame:
        """
        Apply fractional differencing to every column of a Polars DataFrame.

        Args:
            price_frame:    Wide-format DataFrame, columns = ticker symbols.
            log_transform:  If True, apply log() before differencing.

        Returns:
            DataFrame of fractionally differenced series (same shape).
        """
        results: dict[str, list[float]] = {}
        for col in price_frame.columns:
            series = price_frame[col].to_numpy().astype(float)
            if log_transform:
                series = np.log(series + 1e-10)
            fd_result = self.compute(series)
            results[col] = fd_result.series_fracdiff.tolist()
        return pl.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# 2. MICROSTRUCTURE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class MicrostructureEngine(BaseEngine):
    """
    Computes market microstructure signals:
        - VPIN  : Volume-Synchronized Probability of Informed Trading
        - Amihud: Illiquidity ratio (price impact per unit volume)
        - Roll  : Effective bid-ask spread from return serial covariance
    """

    def __init__(self, vpin_bucket_size: int = 50, vpin_window: int = 50) -> None:
        """
        Args:
            vpin_bucket_size: Volume per bucket V* (in share units or normalized).
            vpin_window:      Number of buckets in rolling VPIN window.
        """
        self.vpin_bucket_size = vpin_bucket_size
        self.vpin_window = vpin_window

    # ── VPIN ──────────────────────────────────────────────────────────────────

    def _classify_volume(
        self, returns: np.ndarray, volumes: np.ndarray, n_buckets: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify volume into buy/sell using the bulk volume classification (BVC)
        method (Easley et al., 2012):
            V_buy = V * CDF(ΔP / σ_ΔP)
            V_sell = V - V_buy

        Args:
            returns: Bar-level returns.
            volumes: Bar-level volumes.
            n_buckets: Number of equal-volume buckets.

        Returns:
            buy_vols, sell_vols per bucket.
        """
        sigma = np.std(returns) + 1e-10
        z = returns / sigma
        buy_frac = norm.cdf(z)

        buy_vol = volumes * buy_frac
        sell_vol = volumes * (1 - buy_frac)

        # Aggregate into equal-volume buckets
        cumvol = np.cumsum(volumes)
        bucket_edges = np.arange(1, n_buckets + 1) * (cumvol[-1] / n_buckets)

        buy_buckets = np.zeros(n_buckets)
        sell_buckets = np.zeros(n_buckets)
        bucket_idx = 0

        for i, cv in enumerate(cumvol):
            if bucket_idx >= n_buckets:
                break
            while bucket_idx < n_buckets and cv >= bucket_edges[bucket_idx]:
                buy_buckets[bucket_idx] = buy_vol[i]
                sell_buckets[bucket_idx] = sell_vol[i]
                bucket_idx += 1

        return buy_buckets, sell_buckets

    def compute_vpin(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
    ) -> VPINResult:
        """
        Compute Volume-Synchronized Probability of Informed Trading (VPIN).

        VPIN_n = (1/n) * sum_{t=n-L+1}^{n} |V_buy_t - V_sell_t| / V*

        A rising VPIN signals increased toxicity — informed traders dominate
        order flow, exposing market makers to adverse selection.

        Args:
            returns: 1D array of bar returns (close-to-close or tick returns).
            volumes: 1D array of bar volumes (same length as returns).

        Returns:
            VPINResult with rolling VPIN series and summary statistics.
        """
        self._validate_array(returns, "returns")
        self._validate_array(volumes, "volumes")

        n_buckets = len(returns) // self.vpin_bucket_size
        if n_buckets < self.vpin_window:
            raise ValueError(
                f"Insufficient data: need {self.vpin_window * self.vpin_bucket_size} bars, "
                f"got {len(returns)}."
            )

        buy_buckets, sell_buckets = self._classify_volume(returns, volumes, n_buckets)
        order_imbalance = np.abs(buy_buckets - sell_buckets)
        buy_frac_series = buy_buckets / (buy_buckets + sell_buckets + 1e-10)

        # Rolling sum over vpin_window buckets
        vpin_series = np.array([
            np.sum(order_imbalance[max(0, i - self.vpin_window + 1): i + 1])
            / (self.vpin_window * self.vpin_bucket_size)
            for i in range(len(order_imbalance))
        ])

        return VPINResult(
            vpin_series=vpin_series,
            buy_volume_fraction=buy_frac_series,
            total_buckets=n_buckets,
            mean_vpin=float(np.nanmean(vpin_series)),
            std_vpin=float(np.nanstd(vpin_series)),
        )

    # ── Amihud Illiquidity ────────────────────────────────────────────────────

    def compute_amihud(
        self,
        returns: np.ndarray,
        dollar_volumes: np.ndarray,
        rolling_window: int = 21,
    ) -> np.ndarray:
        """
        Compute the Amihud (2002) illiquidity ratio:
            ILLIQ_t = (1/D) * sum_{d=1}^{D} |R_d| / DVOL_d

        A high ILLIQ indicates large price impact per dollar traded — illiquid.

        Args:
            returns:        Daily returns.
            dollar_volumes: Daily dollar volumes (price × volume).
            rolling_window: Rolling average window in trading days.

        Returns:
            illiq: Rolling Amihud illiquidity series.
        """
        self._validate_array(returns, "returns")
        self._validate_array(dollar_volumes, "dollar_volumes")

        ratio = np.abs(returns) / (dollar_volumes + 1e-10)

        # Rolling mean via convolution
        kernel = np.ones(rolling_window) / rolling_window
        illiq = np.convolve(ratio, kernel, mode="full")[: len(ratio)]
        illiq[:rolling_window - 1] = np.nan
        return illiq

    # ── Roll's Effective Spread ───────────────────────────────────────────────

    def compute_roll_spread(
        self,
        close_prices: np.ndarray,
        rolling_window: int = 21,
    ) -> np.ndarray:
        """
        Roll (1984) effective bid-ask spread estimator from transaction prices.

        Roll's model: Cov(ΔP_t, ΔP_{t-1}) = -c^2
        → Effective spread = 2 * sqrt(-Cov) if Cov < 0, else 0.

        Args:
            close_prices:   Transaction or close price series.
            rolling_window: Window for rolling covariance estimation.

        Returns:
            roll_spread: Time series of effective spreads.
        """
        self._validate_array(close_prices, "close_prices")

        price_changes = np.diff(close_prices)
        n = len(price_changes)
        roll_spread = np.full(n + 1, np.nan)

        for t in range(rolling_window, n):
            window = price_changes[t - rolling_window: t]
            cov = np.cov(window[1:], window[:-1])[0, 1]
            roll_spread[t] = 2.0 * np.sqrt(max(-cov, 0.0))

        return roll_spread

    def compute(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        close_prices: np.ndarray,
        dollar_volumes: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute all microstructure signals in one call.

        Returns dict with keys: 'vpin', 'amihud', 'roll_spread'.
        """
        if dollar_volumes is None:
            dollar_volumes = close_prices * volumes

        vpin_result = self.compute_vpin(returns, volumes)
        amihud = self.compute_amihud(returns, dollar_volumes)
        roll = self.compute_roll_spread(close_prices)

        return {
            "vpin": vpin_result.vpin_series,
            "amihud": amihud,
            "roll_spread": roll,
            "buy_volume_fraction": vpin_result.buy_volume_fraction,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL ARBITRAGE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class StatArbEngine(BaseEngine):
    """
    Statistical Arbitrage signal generation:
        - Ornstein-Uhlenbeck process estimation (MLE) → half-life
        - Johansen cointegration test for pairs/basket construction
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    # ── OU Process MLE ────────────────────────────────────────────────────────

    def _ou_log_likelihood(
        self, params: np.ndarray, spread: np.ndarray, dt: float
    ) -> float:
        """
        Negative log-likelihood of discrete OU process (Euler–Maruyama):
            dX_t = θ(μ - X_t)dt + σ dW_t

        Discretized: X_{t+1} = X_t + θ(μ - X_t)dt + ε_t,  ε ~ N(0, σ²dt)

        Equivalent AR(1) OLS:
            X_{t+1} = a + b * X_t + ε_t
            θ = -ln(b)/dt,  μ = a/(1-b),  σ = std(ε)/sqrt(dt/(1-b²)/(2θ))
        """
        theta, mu, sigma = params
        if theta <= 0 or sigma <= 0:
            return 1e10

        n = len(spread)
        x_t = spread[:-1]
        x_next = spread[1:]

        # Conditional mean under OU
        exp_decay = np.exp(-theta * dt)
        cond_mean = x_t * exp_decay + mu * (1 - exp_decay)
        cond_var = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)

        residuals = x_next - cond_mean
        ll = -0.5 * n * np.log(2 * np.pi * cond_var) - np.sum(residuals**2) / (2 * cond_var)
        return -ll  # negative LL for minimizer

    def fit_ou(
        self,
        spread: np.ndarray,
        dt: float = 1.0 / 252,
    ) -> OUResult:
        """
        Fit an Ornstein-Uhlenbeck process to a spread/residual series via MLE.

        The half-life gives traders the expected time for the spread to revert
        halfway to its mean: τ_{1/2} = ln(2) / θ

        Args:
            spread: Stationary spread series (e.g., residuals from OLS hedge ratio).
            dt:     Time step in years (1/252 for daily, 1/252/6.5/60 for 1-min).

        Returns:
            OUResult with mu, theta, sigma, half_life, and residuals.
        """
        self._validate_array(spread, "spread")

        # Initial estimate via OLS AR(1) regression
        x_t, x_next = spread[:-1], spread[1:]
        b_ols = np.cov(x_t, x_next)[0, 1] / (np.var(x_t) + 1e-10)
        a_ols = np.mean(x_next) - b_ols * np.mean(x_t)
        theta0 = max(-np.log(b_ols) / dt, 0.01)
        mu0 = a_ols / (1 - b_ols + 1e-10)
        sigma0 = np.std(x_next - b_ols * x_t - a_ols) / np.sqrt(dt + 1e-10)

        result = minimize(
            self._ou_log_likelihood,
            x0=[theta0, mu0, sigma0],
            args=(spread, dt),
            method="L-BFGS-B",
            bounds=[(1e-4, None), (None, None), (1e-6, None)],
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        theta, mu, sigma = result.x
        half_life = np.log(2) / theta  # in units of dt
        residuals = spread[1:] - (spread[:-1] * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt)))

        logger.info(
            f"[StatArbEngine] OU fit → θ={theta:.4f}, μ={mu:.4f}, "
            f"σ={sigma:.4f}, half_life={half_life:.2f} bars"
        )
        return OUResult(mu=mu, theta=theta, sigma=sigma, half_life=half_life, residuals=residuals)

    # ── Johansen Cointegration ────────────────────────────────────────────────

    def johansen_test(
        self,
        price_matrix: np.ndarray,
        det_order: int = 0,
        k_ar_diff: int = 1,
    ) -> JohansenResult:
        """
        Johansen (1991) maximum likelihood cointegration test for a system
        of price series.

        Determines the number of cointegrating relationships (rank r) among
        the N assets, and returns the eigenvectors (hedge ratios) for
        portfolio construction.

        Args:
            price_matrix: Shape (T, N) — T observations, N asset prices.
            det_order:    -1 (no const), 0 (restricted const), 1 (unrestricted).
            k_ar_diff:    Number of lagged differences in the VECM.

        Returns:
            JohansenResult with cointegration rank, eigenvectors, and test stats.
        """
        if price_matrix.ndim != 2 or price_matrix.shape[1] < 2:
            raise ValueError("price_matrix must be (T, N) with N >= 2.")

        result = coint_johansen(price_matrix, det_order=det_order, k_ar_diff=k_ar_diff)

        # Trace statistic vs 90% critical value
        trace_stats = result.lr1          # shape (N,)
        crit_90 = result.cvt[:, 0]        # 90% critical values

        # Count cointegrating vectors where trace stat > critical value
        n_coint = int(np.sum(trace_stats > crit_90))

        logger.info(
            f"[StatArbEngine] Johansen: {n_coint} cointegrating vector(s) found "
            f"at 90% confidence."
        )
        return JohansenResult(
            is_cointegrated=n_coint >= 1,
            n_cointegrating_vectors=n_coint,
            eigenvectors=result.evec,
            eigenvalues=result.eig,
            trace_stats=trace_stats,
            critical_values_90=crit_90,
        )

    def compute_hedge_ratio_spread(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        OLS hedge ratio + spread for a two-asset pair.
        Uses log prices to ensure the spread is scale-invariant.

        Returns:
            (beta, spread) where spread = log(A) - beta * log(B)
        """
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)
        beta = np.cov(log_a, log_b)[0, 1] / (np.var(log_b) + 1e-10)
        spread = log_a - beta * log_b
        return float(beta), spread

    def compute(
        self,
        price_matrix: np.ndarray,
        dt: float = 1.0 / 252,
    ) -> dict:
        """
        Full StatArb pipeline: cointegration test + OU fitting on first
        cointegrating spread.

        Args:
            price_matrix: Shape (T, N).
            dt: Time step in years.

        Returns:
            dict with JohansenResult and OUResult for the primary spread.
        """
        joh = self.johansen_test(price_matrix)

        # Primary spread: first eigenvector portfolio
        hedge_weights = joh.eigenvectors[:, 0]
        log_prices = np.log(price_matrix + 1e-10)
        spread = log_prices @ hedge_weights
        spread -= np.mean(spread)

        ou = self.fit_ou(spread, dt=dt)
        return {"johansen": joh, "ou": ou, "spread": spread, "hedge_weights": hedge_weights}


# ──────────────────────────────────────────────────────────────────────────────
# 4. INFORMATION THEORY ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class InformationTheoryEngine(BaseEngine):
    """
    Non-linear dependency detection using information-theoretic measures:
        - Mutual Information (MI): captures any dependency (linear + non-linear).
        - Transfer Entropy (TE): directed information flow from X → Y.

    Transfer Entropy:
        TE(X→Y) = MI(Y_future ; X_past | Y_past)
                = H(Y_t+1 | Y_past) - H(Y_t+1 | Y_past, X_past)

    We discretize continuous returns into bins for entropy estimation.
    For large datasets, k-nearest-neighbor MI estimators (sklearn) are used.
    """

    def __init__(self, n_bins: int = 10, lag: int = 1) -> None:
        """
        Args:
            n_bins: Number of histogram bins for discretization.
            lag:    Look-back lag for transfer entropy calculation.
        """
        self.n_bins = n_bins
        self.lag = lag

    def _discretize(self, series: np.ndarray) -> np.ndarray:
        """Bin continuous returns into n_bins equal-frequency bins."""
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(series, quantiles)
        edges[0] -= 1e-10
        edges[-1] += 1e-10
        return np.digitize(series, edges[1:-1])

    def _entropy(self, x: np.ndarray) -> float:
        """Shannon entropy H(X) = -sum p(x) log p(x)."""
        _, counts = np.unique(x, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def _joint_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Joint entropy H(X, Y)."""
        pairs = x * (self.n_bins + 1) + y  # unique encoding
        return self._entropy(pairs)

    def compute_mutual_information(
        self,
        returns_x: np.ndarray,
        returns_y: np.ndarray,
        method: str = "knn",
    ) -> float:
        """
        Compute Mutual Information MI(X; Y) = H(X) + H(Y) - H(X, Y).

        Args:
            returns_x: Return series for asset X.
            returns_y: Return series for asset Y.
            method:    'knn' (sklearn, continuous) or 'histogram' (discrete).

        Returns:
            mi: Mutual information in nats (knn) or bits (histogram).
        """
        self._validate_array(returns_x, "returns_x")
        self._validate_array(returns_y, "returns_y")

        if method == "knn":
            mi_vals = mutual_info_regression(
                returns_x.reshape(-1, 1), returns_y, n_neighbors=5, random_state=42
            )
            return float(mi_vals[0])
        else:
            dx = self._discretize(returns_x)
            dy = self._discretize(returns_y)
            return self._entropy(dx) + self._entropy(dy) - self._joint_entropy(dx, dy)

    def compute_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute Transfer Entropy TE(source → target).

        TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

        A significant TE(X→Y) implies X Granger-causes Y in an
        information-theoretic sense — X improves prediction of Y
        beyond Y's own history.

        Args:
            source: The driving series X (potential cause).
            target: The driven series Y (potential effect).

        Returns:
            te: Transfer entropy in bits. Higher = stronger causal flow.
        """
        self._validate_array(source, "source")
        self._validate_array(target, "target")

        lag = self.lag
        ds = self._discretize(source[:-lag])   # X_{t-1}
        dt_past = self._discretize(target[:-lag])  # Y_{t-1}
        dt_fut = self._discretize(target[lag:])    # Y_t

        # H(Y_t | Y_{t-1}) = H(Y_t, Y_{t-1}) - H(Y_{t-1})
        h_yt_cond_ypast = self._joint_entropy(dt_fut, dt_past) - self._entropy(dt_past)

        # H(Y_t | Y_{t-1}, X_{t-1}) = H(Y_t, Y_{t-1}, X_{t-1}) - H(Y_{t-1}, X_{t-1})
        triple = dt_fut * (self.n_bins + 1) ** 2 + dt_past * (self.n_bins + 1) + ds
        h_yt_cond_both = self._entropy(triple) - self._joint_entropy(dt_past, ds)

        te = h_yt_cond_ypast - h_yt_cond_both
        return float(max(te, 0.0))  # TE is non-negative by definition

    def compute_te_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the full N×N Transfer Entropy matrix for an asset universe.

        te_matrix[i, j] = TE(asset_i → asset_j)

        Args:
            returns_matrix: Shape (T, N) — T time steps, N assets.

        Returns:
            te_matrix: Shape (N, N), zeros on diagonal.
        """
        T, N = returns_matrix.shape
        te_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    te_matrix[i, j] = self.compute_transfer_entropy(
                        returns_matrix[:, i], returns_matrix[:, j]
                    )
        return te_matrix

    def compute(
        self,
        returns_matrix: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Compute both MI and TE matrices for the full asset universe.

        Args:
            returns_matrix: Shape (T, N).

        Returns:
            dict with 'mi_matrix' and 'te_matrix', both (N, N).
        """
        T, N = returns_matrix.shape
        mi_matrix = np.zeros((N, N))
        te_matrix = self.compute_te_matrix(returns_matrix)

        for i in range(N):
            for j in range(i + 1, N):
                mi = self.compute_mutual_information(
                    returns_matrix[:, i], returns_matrix[:, j]
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi  # symmetric

        return {"mi_matrix": mi_matrix, "te_matrix": te_matrix}


# ──────────────────────────────────────────────────────────────────────────────
# ALPHA FACTORY ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

class AlphaFactory:
    """
    Top-level orchestrator that pipelines all signal generators.

    Usage:
        factory = AlphaFactory()
        signals = factory.generate_all(price_df, returns_df, volume_df)
    """

    def __init__(
        self,
        stationarity_engine: Optional[StationarityEngine] = None,
        microstructure_engine: Optional[MicrostructureEngine] = None,
        stat_arb_engine: Optional[StatArbEngine] = None,
        info_theory_engine: Optional[InformationTheoryEngine] = None,
    ) -> None:
        self.stationarity = stationarity_engine or StationarityEngine()
        self.microstructure = microstructure_engine or MicrostructureEngine()
        self.stat_arb = stat_arb_engine or StatArbEngine()
        self.info_theory = info_theory_engine or InformationTheoryEngine()

    def generate_all(
        self,
        prices: pl.DataFrame,
        returns: np.ndarray,
        volumes: np.ndarray,
        close_prices: np.ndarray,
        dt: float = 1.0 / 252,
    ) -> dict:
        """
        Run all signal engines and return a unified signal dictionary.

        Args:
            prices:       Wide Polars DataFrame (T × N), raw prices.
            returns:      Shape (T, N) numpy array of returns.
            volumes:      Shape (T,) bar volumes for VPIN/Amihud (single asset or first asset).
            close_prices: Shape (T,) close prices.
            dt:           Time step in years.

        Returns:
            signals: Nested dict with keys: 'fracdiff', 'microstructure', 'stat_arb', 'info_theory'.
        """
        logger.info("[AlphaFactory] Starting full signal generation pipeline...")

        # 1. Fractional Differencing
        fracdiff_df = self.stationarity.batch_compute(prices, log_transform=True)
        logger.info("[AlphaFactory] ✓ Fractional Differencing complete.")

        # 2. Microstructure
        micro_signals = self.microstructure.compute(
            returns[:, 0], volumes, close_prices
        )
        logger.info("[AlphaFactory] ✓ Microstructure signals complete.")

        # 3. StatArb (on full price matrix)
        price_matrix = prices.to_numpy().astype(float)
        stat_arb_signals = self.stat_arb.compute(price_matrix, dt=dt)
        logger.info("[AlphaFactory] ✓ Stat Arb (OU + Johansen) complete.")

        # 4. Information Theory
        it_signals = self.info_theory.compute(returns)
        logger.info("[AlphaFactory] ✓ Information Theory signals complete.")

        return {
            "fracdiff": fracdiff_df,
            "microstructure": micro_signals,
            "stat_arb": stat_arb_signals,
            "info_theory": it_signals,
        }