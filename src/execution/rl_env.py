"""
================================================================================
PHASE 4: OPTIMAL EXECUTION & REINFORCEMENT LEARNING
================================================================================
Modules:
    1. AlmgrenChrissModel   — Temporary and permanent market impact framework
    2. ExecutionEnv         — OpenAI Gym-compatible RL environment for
                              optimal liquidation / acquisition

The agent's goal: liquidate Q shares over T periods, minimizing
Implementation Shortfall (IS) = (Arrival Price × Q) - Actual Proceeds.

The optimal deterministic Almgren-Chriss strategy is the *benchmark* — the
RL agent must learn to do better by adapting to stochastic market conditions.

Dependencies: numpy, gymnasium (or gym)
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

logger = logging.getLogger("ExecutionRL")


# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketImpactParams:
    """Parameters for the Almgren-Chriss market impact model."""
    sigma: float          # Daily return volatility of the stock
    eta: float            # Temporary impact coefficient (η) — liquidity cost
    gamma: float          # Permanent impact coefficient (γ) — price depression
    epsilon: float        # Fixed spread cost (half bid-ask spread)
    tau: float            # Duration of one trading interval (in days)
    T: int                # Total number of intervals
    Q: float              # Initial inventory (total shares to liquidate)
    risk_aversion: float  # λ — trader's risk aversion to execution risk


@dataclass
class ExecutionStep:
    """Result of a single execution step."""
    inventory_remaining: float
    shares_traded: float
    price: float
    market_impact_cost: float
    implementation_shortfall: float
    reward: float
    done: bool


# ──────────────────────────────────────────────────────────────────────────────
# 1. ALMGREN-CHRISS MARKET IMPACT MODEL
# ──────────────────────────────────────────────────────────────────────────────

class AlmgrenChrissModel:
    """
    Almgren & Chriss (2000) optimal execution framework.

    Price dynamics under the AC model:
        S_k = S_{k-1} - γ · n_k + σ · √τ · ξ_k     (permanent impact)
        S̃_k = S_k - η · n_k / τ                      (effective price with temporary impact)

    Where:
        n_k  = shares traded in interval k
        γ    = permanent impact per share (shifts price permanently)
        η    = temporary impact per share (instantaneous liquidity cost)
        σ    = volatility, τ = interval length, ξ ~ N(0,1)

    Expected Shortfall (deterministic):
        E[IS] = (ε · Q) + (1/2) γ Q² + η Σ n_k²/τ

    Variance of Shortfall:
        Var[IS] = σ² τ Σ x_k²   where x_k = remaining inventory at time k

    The optimal strategy minimizes the risk-adjusted cost:
        E[IS] + λ · Var[IS]

    → Closed-form solution: hyperbolic sine (sinh) liquidation trajectory.
    """

    def __init__(self, params: MarketImpactParams) -> None:
        self.p = params

    def optimal_trajectory(self) -> np.ndarray:
        """
        Compute the Almgren-Chriss deterministic optimal liquidation trajectory.

        Returns the inventory schedule x[k] for k = 0, 1, ..., T,
        where x[0] = Q and x[T] = 0.

        The closed-form solution (no-shorting constraint ignored for now):
            x(t) = Q · sinh(κ(T-t)) / sinh(κT)

            κ² = λ σ² / (η/τ)

        Returns:
            trajectory: (T+1,) array of remaining inventories.
        """
        p = self.p
        kappa_sq = p.risk_aversion * p.sigma**2 * p.tau / (p.eta + 1e-10)
        kappa = np.sqrt(max(kappa_sq, 0.0))

        times = np.arange(p.T + 1)

        if kappa < 1e-6:
            # Risk-neutral limit: uniform liquidation
            trajectory = p.Q * (1 - times / p.T)
        else:
            trajectory = p.Q * np.sinh(kappa * (p.T - times)) / (np.sinh(kappa * p.T) + 1e-10)

        trajectory = np.maximum(trajectory, 0.0)
        trajectory[p.T] = 0.0  # Force full liquidation
        return trajectory

    def optimal_schedule(self) -> np.ndarray:
        """
        Compute per-period trading schedule n[k] = x[k-1] - x[k].

        Returns:
            schedule: (T,) array of shares to trade per interval.
        """
        traj = self.optimal_trajectory()
        return np.diff(-traj)   # positive = selling

    def expected_shortfall(self, schedule: np.ndarray) -> float:
        """
        Compute the expected implementation shortfall for a given schedule.

        IS = ε·Q + (1/2)γ·Q² + η·Σ(n_k²/τ)

        Args:
            schedule: (T,) per-period trades.

        Returns:
            Expected IS in dollar terms (per-share × initial price).
        """
        p = self.p
        spread_cost = p.epsilon * p.Q
        permanent_cost = 0.5 * p.gamma * p.Q**2
        temporary_cost = p.eta * np.sum(schedule**2) / p.tau
        return float(spread_cost + permanent_cost + temporary_cost)

    def shortfall_variance(self) -> float:
        """
        Compute the variance of implementation shortfall under optimal strategy.

        Var[IS] = σ²τ Σ x_k²

        Returns:
            Variance of IS.
        """
        p = self.p
        trajectory = self.optimal_trajectory()
        return float(p.sigma**2 * p.tau * np.sum(trajectory**2))

    def twap_schedule(self) -> np.ndarray:
        """Time-Weighted Average Price benchmark: uniform liquidation."""
        return np.full(self.p.T, self.p.Q / self.p.T)

    def vwap_schedule(self, volume_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Volume-Weighted Average Price benchmark.
        Trades proportionally to a U-shaped intraday volume profile.

        Args:
            volume_profile: (T,) normalized volume weights. If None, use U-shape.

        Returns:
            schedule: (T,) VWAP-proportional trades.
        """
        if volume_profile is None:
            # U-shaped intraday volume (high at open and close)
            t = np.linspace(0, 1, self.p.T)
            volume_profile = 0.5 + 0.5 * np.cos(np.pi * t) ** 2
            volume_profile /= volume_profile.sum()
        return self.p.Q * volume_profile


# ──────────────────────────────────────────────────────────────────────────────
# 2. REINFORCEMENT LEARNING EXECUTION ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionEnv(gym.Env):
    """
    OpenAI Gymnasium-compatible environment for optimal execution via RL.

    The agent must liquidate Q shares in T intervals, minimizing
    implementation shortfall against the VWAP benchmark.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ STATE SPACE (observation_space):                                    │
    │   s_t = [                                                           │
    │     x_t / Q,          # Normalized remaining inventory [0, 1]      │
    │     t / T,            # Normalized time remaining [0, 1]            │
    │     P_t / P_0 - 1,    # Price return from arrival [~-0.1, 0.1]     │
    │     spread_t,         # Current bid-ask spread (normalized)         │
    │     volatility_t,     # Rolling 5-period realized vol               │
    │     vwap_shortfall_t, # Cumulative IS vs VWAP benchmark             │
    │   ]                                                                  │
    │                                                                     │
    │ ACTION SPACE (action_space):                                        │
    │   Continuous: a_t ∈ [-1, 1] normalized                             │
    │   Actual trade: n_t = clip(a_t × Q/T, 0, x_t)   (long-only sell)  │
    │                                                                     │
    │ REWARD:                                                             │
    │   r_t = -(market_impact_cost_t + λ × IS_vs_VWAP_t²)               │
    │   Terminal: -penalty if leftover inventory > 0                      │
    └─────────────────────────────────────────────────────────────────────┘

    Args:
        params:           AlmgrenChriss market impact parameters.
        initial_price:    Arrival price S_0 (INR for Indian markets).
        max_steps:        Maximum steps per episode (= params.T).
        adverse_prob:     Probability of an adverse price jump per step.
        adverse_magnitude: Size of adverse jump as fraction of price.
        vwap_volume_profile: (T,) normalized intraday volume for VWAP baseline.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        params: MarketImpactParams,
        initial_price: float = 1000.0,
        adverse_prob: float = 0.05,
        adverse_magnitude: float = 0.005,
        vwap_volume_profile: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.params = params
        self.initial_price = initial_price
        self.adverse_prob = adverse_prob
        self.adverse_magnitude = adverse_magnitude

        self.ac_model = AlmgrenChrissModel(params)
        self.vwap_schedule = self.ac_model.vwap_schedule(vwap_volume_profile)
        self.vwap_cumulative = np.cumsum(self.vwap_schedule)

        # Observation: 6-dimensional continuous state
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -0.5, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.5, 0.1, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Continuous action in [-1, 1], scaled to actual trade size
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.step_idx: int = 0
        self.inventory: float = self.params.Q
        self.price: float = self.initial_price
        self.cumulative_proceeds: float = 0.0
        self.vwap_proceeds: float = 0.0
        self.price_path: list[float] = [self.initial_price]

    def _simulate_price(self, trade_size: float) -> float:
        """
        Simulate next price under Almgren-Chriss price dynamics.

        P_t = P_{t-1}
              - γ · n_t             (permanent impact)
              + σ √τ · ξ            (diffusion)
              ± adverse jump        (adverse selection event)

        Args:
            trade_size: Shares sold this period n_t.

        Returns:
            new_price after permanent impact and diffusion.
        """
        p = self.params
        diffusion = p.sigma * np.sqrt(p.tau) * np.random.randn()
        permanent_impact = p.gamma * trade_size
        adverse = 0.0
        if np.random.rand() < self.adverse_prob:
            adverse = -self.adverse_magnitude * self.price * np.sign(self.inventory)

        return max(self.price - permanent_impact + diffusion + adverse, 0.01)

    def _execution_price(self, trade_size: float, post_price: float) -> float:
        """
        Effective execution price including temporary market impact.

        S̃_t = S_t - η · n_t / τ  (sell at discount due to urgency)
        """
        p = self.params
        temp_impact = p.eta * trade_size / (p.tau + 1e-10)
        return post_price - temp_impact - p.epsilon  # minus fixed spread

    def _get_obs(self, implementation_shortfall: float) -> np.ndarray:
        """Construct normalized state observation vector."""
        rolling_vol = np.std(np.diff(self.price_path[-6:])) if len(self.price_path) > 5 else self.params.sigma
        spread_norm = self.params.epsilon / (self.price + 1e-10)
        vwap_is_norm = implementation_shortfall / (self.initial_price * self.params.Q + 1e-10)

        return np.array([
            self.inventory / self.params.Q,
            self.step_idx / self.params.T,
            self.price / self.initial_price - 1.0,
            min(spread_norm, 0.1),
            min(rolling_vol / self.params.sigma, 1.0),
            np.clip(vwap_is_norm, -1.0, 1.0),
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to start of liquidation episode."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        obs = self._get_obs(0.0)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute one liquidation step.

        Args:
            action: (1,) array in [-1, 1]. Scaled to trade size:
                    n_t = clip(action × Q/T × 2, 0, inventory)

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Clip action and map to trade size
        action_scalar = float(np.clip(action[0], -1, 1))
        # Only allow selling (≥ 0) and cap at remaining inventory
        trade_size = float(np.clip(
            action_scalar * (self.params.Q / self.params.T) * 2.0,
            0.0,
            self.inventory,
        ))

        # VWAP benchmark proceeds for this step
        vwap_trade = min(self.vwap_schedule[self.step_idx], self.inventory)
        self.vwap_proceeds += vwap_trade * self.price  # VWAP at mid-price

        # Simulate post-trade price
        new_price = self._simulate_price(trade_size)
        exec_price = self._execution_price(trade_size, new_price)
        self.price_path.append(new_price)

        # Execution proceeds
        proceeds = trade_size * max(exec_price, 0.0)
        self.cumulative_proceeds += proceeds
        self.inventory -= trade_size
        self.price = new_price
        self.step_idx += 1

        # Market impact cost (vs. trading at arrival price)
        arrival_value = trade_size * self.initial_price
        impact_cost = arrival_value - proceeds

        # Implementation shortfall vs VWAP
        is_vs_vwap = self.vwap_proceeds - self.cumulative_proceeds

        # Reward: negative cost — penalize both impact and IS deviation
        reward = -(impact_cost + 0.5 * max(is_vs_vwap, 0.0))

        terminated = (self.step_idx >= self.params.T) or (self.inventory <= 1e-4)
        truncated = False

        # Terminal penalty for unexecuted inventory
        if terminated and self.inventory > 1e-4:
            # Forced market order at 5% discount — severe penalty
            emergency_proceeds = self.inventory * self.price * 0.95
            self.cumulative_proceeds += emergency_proceeds
            penalty = self.inventory * self.initial_price * 0.1
            reward -= penalty
            self.inventory = 0.0
            logger.debug(f"[ExecutionEnv] Emergency liquidation: penalty={penalty:.2f}")

        obs = self._get_obs(is_vs_vwap)
        info = {
            "step": self.step_idx,
            "inventory": self.inventory,
            "price": self.price,
            "trade_size": trade_size,
            "proceeds": proceeds,
            "impact_cost": impact_cost,
            "is_vs_vwap": is_vs_vwap,
            "cumulative_proceeds": self.cumulative_proceeds,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Print current execution state."""
        pct_complete = 1.0 - self.inventory / self.params.Q
        print(
            f"Step {self.step_idx:3d}/{self.params.T} | "
            f"Inventory: {self.inventory:8.0f} ({pct_complete:.1%} complete) | "
            f"Price: ₹{self.price:8.2f} | "
            f"Proceeds: ₹{self.cumulative_proceeds:,.0f}"
        )

    def benchmark_vwap_shortfall(self) -> float:
        """
        Run the AC optimal trajectory as a deterministic baseline and
        return the expected IS vs VWAP.
        """
        ac = AlmgrenChrissModel(self.params)
        schedule = ac.optimal_schedule()
        return ac.expected_shortfall(schedule)


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY: BUILD ENVIRONMENT WITH SENSIBLE DEFAULTS FOR NIFTY 50
# ──────────────────────────────────────────────────────────────────────────────

def make_nifty_execution_env(
    ticker_avg_daily_volume: float = 1_000_000,
    initial_price: float = 1500.0,
    position_to_liquidate_pct: float = 0.02,
    n_intervals: int = 78,  # 5-min bars in NSE session (9:15–15:30)
    sigma_daily: float = 0.015,
) -> ExecutionEnv:
    """
    Create a realistic NSE execution environment.

    Args:
        ticker_avg_daily_volume: Average daily volume in shares.
        initial_price:           Arrival price in INR.
        position_to_liquidate_pct: Position as fraction of ADV.
        n_intervals:             Number of 5-min intervals in NSE session.
        sigma_daily:             Daily return volatility.

    Returns:
        ExecutionEnv ready for RL training.
    """
    Q = ticker_avg_daily_volume * position_to_liquidate_pct
    tau = 1.0 / 252 / (n_intervals / 78)  # Interval duration in years

    params = MarketImpactParams(
        sigma=sigma_daily,
        eta=0.0001 * initial_price,    # ~1 bps temporary impact per 1% ADV
        gamma=0.00005 * initial_price, # ~0.5 bps permanent impact
        epsilon=initial_price * 0.0005,  # 5 bps half-spread
        tau=tau,
        T=n_intervals,
        Q=Q,
        risk_aversion=1e-6,
    )

    logger.info(
        f"[ExecutionEnv] NSE env: Q={Q:.0f} shares, T={n_intervals} intervals, "
        f"σ_daily={sigma_daily:.1%}, P0=₹{initial_price:.0f}"
    )
    return ExecutionEnv(params=params, initial_price=initial_price)