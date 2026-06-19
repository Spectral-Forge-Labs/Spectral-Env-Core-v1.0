"""
spectral_env_core.backtest
===========================
Backtesting harness for trained RL agents.

Two modes:
  - Synthetic: run the model on N stochastic episodes from SpectralTradingEnv
  - Historical: replay real price data through the environment mechanics

Usage
-----
    from spectral_env_core.backtest import backtest, backtest_historical

    # Synthetic — test on infinite generated data
    report = backtest(model, env_kwargs, n_episodes=100)
    print(report.summary)
    report.plot()

    # Historical — test on real price data the agent was never trained on
    report = backtest_historical(model, tickers=["NVDA", "AAPL"], period="1y")
    print(report.summary)
    report.plot()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestReport:
    """
    Structured results from a backtest run.

    Attributes
    ----------
    stats : dict
        Summary statistics (sharpe, max_drawdown, win_rate, etc.)
    episode_returns : np.ndarray
        Per-episode total returns (% of starting cash)
    episode_frictions : np.ndarray
        Per-episode cumulative transaction + slippage costs
    equity_curves : list[np.ndarray]
        Per-episode portfolio value series (for plotting)
    mode : str
        'synthetic' or 'historical'
    """
    stats: dict = field(default_factory=dict)
    episode_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    episode_frictions: np.ndarray = field(default_factory=lambda: np.array([]))
    equity_curves: list = field(default_factory=list)
    mode: str = "synthetic"

    @property
    def summary(self) -> str:
        """Formatted summary string."""
        s = self.stats
        lines = [
            f"",
            f"{'═' * 50}",
            f" Backtest Report ({self.mode})",
            f"{'═' * 50}",
            f" Episodes:          {s.get('n_episodes', 0)}",
            f" Mean Return:       {s.get('mean_return', 0):+.2f}%",
            f" Std Return:        {s.get('std_return', 0):.2f}%",
            f" Sharpe Ratio:      {s.get('sharpe_ratio', 0):.3f}",
            f" Sortino Ratio:     {s.get('sortino_ratio', 0):.3f}",
            f" Max Drawdown:      {s.get('max_drawdown', 0):.2f}%",
            f" Win Rate:          {s.get('win_rate', 0):.1%}",
            f" Profit Factor:     {s.get('profit_factor', 0):.2f}",
            f" Avg Friction/Ep:   ${s.get('avg_friction', 0):,.2f}",
            f" Calmar Ratio:      {s.get('calmar_ratio', 0):.3f}",
            f"{'═' * 50}",
        ]
        return "\n".join(lines)

    def plot(self):
        """Plot equity curves and return distribution."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Equity curves (first 20 episodes)
        for i, curve in enumerate(self.equity_curves[:20]):
            axes[0, 0].plot(curve, alpha=0.5, lw=0.8)
        axes[0, 0].set_title(f'Equity Curves ({min(20, len(self.equity_curves))} episodes)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)

        # Return distribution
        axes[0, 1].hist(self.episode_returns, bins=30, color='steelblue',
                        edgecolor='white', alpha=0.8)
        axes[0, 1].axvline(0, color='black', lw=1, ls='--')
        axes[0, 1].axvline(self.stats.get('mean_return', 0), color='red', lw=2,
                           label=f"Mean: {self.stats.get('mean_return', 0):+.2f}%")
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].set_xlabel('Episode Return (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Drawdown
        if self.equity_curves:
            worst_idx = int(np.argmin(self.episode_returns))
            worst_curve = self.equity_curves[worst_idx]
            running_max = np.maximum.accumulate(worst_curve)
            drawdown = (worst_curve - running_max) / running_max * 100
            axes[1, 0].fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.4)
            axes[1, 0].set_title(f'Worst Episode Drawdown ({self.episode_returns[worst_idx]:+.1f}%)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True, alpha=0.3)

        # Friction
        axes[1, 1].hist(self.episode_frictions, bins=30, color='purple',
                        edgecolor='white', alpha=0.8)
        axes[1, 1].set_title('Friction Distribution')
        axes[1, 1].set_xlabel('Episode Friction ($)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Backtest Results ({self.mode})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()


def _compute_stats(returns: np.ndarray, frictions: np.ndarray,
                   equity_curves: list) -> dict:
    """Compute summary statistics from backtest results."""
    mean_r = float(returns.mean())
    std_r  = float(returns.std()) if len(returns) > 1 else 1.0

    # Sharpe (annualised if episodes are 252 steps ≈ 1 year)
    sharpe = mean_r / std_r if std_r > 0 else 0.0

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 1.0
    sortino = mean_r / downside_std if downside_std > 0 else 0.0

    # Max drawdown (worst intra-episode drawdown across all episodes)
    max_dd = 0.0
    for curve in equity_curves:
        if len(curve) < 2:
            continue
        running_max = np.maximum.accumulate(curve)
        dd = (curve - running_max) / running_max * 100
        episode_max_dd = abs(float(dd.min()))
        max_dd = max(max_dd, episode_max_dd)

    # Win rate
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

    # Profit factor
    gross_profit = float(returns[returns > 0].sum()) if (returns > 0).any() else 0.0
    gross_loss   = abs(float(returns[returns < 0].sum())) if (returns < 0).any() else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calmar
    calmar = mean_r / max_dd if max_dd > 0 else 0.0

    return {
        "n_episodes":    len(returns),
        "mean_return":   mean_r,
        "std_return":    std_r,
        "sharpe_ratio":  sharpe,
        "sortino_ratio": sortino,
        "max_drawdown":  max_dd,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "avg_friction":  float(frictions.mean()),
        "calmar_ratio":  calmar,
    }


def backtest(
    model,
    env_kwargs: dict,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed_start: int = 0,
) -> BacktestReport:
    """
    Run a synthetic backtest: N stochastic episodes on SpectralTradingEnv.

    Parameters
    ----------
    model : object
        Trained model with .predict(obs, deterministic=...) method (e.g. SB3 PPO).
    env_kwargs : dict
        Kwargs passed to SpectralTradingEnv constructor.
    n_episodes : int
        Number of episodes to run.
    deterministic : bool
        Use deterministic policy (True for eval).
    seed_start : int
        First episode seed. Episodes use seeds [seed_start, seed_start + n_episodes).

    Returns
    -------
    BacktestReport
    """
    from spectral_env_core import SpectralTradingEnv

    env = SpectralTradingEnv(**env_kwargs)
    episode_returns  = []
    episode_frictions = []
    equity_curves    = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        total_reward  = 0.0
        total_friction = 0.0
        curve = [info['cash'] + sum(
            s * p for s, p in zip(info['shares'], info['prices'])
        )]

        for _ in range(env.num_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward  += reward
            total_friction += info.get('friction_leak', 0.0)
            portfolio_val = info['cash'] + sum(
                s * p for s, p in zip(info['shares'], info['prices'])
            )
            curve.append(portfolio_val)
            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        episode_frictions.append(total_friction)
        equity_curves.append(np.array(curve))

    returns   = np.array(episode_returns)
    frictions = np.array(episode_frictions)
    stats     = _compute_stats(returns, frictions, equity_curves)

    return BacktestReport(
        stats=stats,
        episode_returns=returns,
        episode_frictions=frictions,
        equity_curves=equity_curves,
        mode="synthetic",
    )


def backtest_historical(
    model,
    tickers: list,
    period: str = "1y",
    env_kwargs: Optional[dict] = None,
    deterministic: bool = True,
) -> BacktestReport:
    """
    Run a historical backtest: replay real price data through the environment.

    Fetches actual price history for the given tickers, injects it as the
    price path (replacing the stochastic generator), and runs the trained
    model against real market data it was never trained on.

    Parameters
    ----------
    model : object
        Trained model with .predict(obs, deterministic=...) method.
    tickers : list[str]
        Ticker symbols to fetch (e.g. ["NVDA", "AAPL"]).
    period : str
        yfinance period string (e.g. "1y", "2y", "6mo").
    env_kwargs : dict, optional
        Additional env kwargs (starting_cash, max_shares, etc.).
        If None, uses sensible defaults.
    deterministic : bool
        Use deterministic policy.

    Returns
    -------
    BacktestReport

    Notes
    -----
    The environment is constructed normally but its brownian_path is replaced
    with real price data. All mechanics (fees, slippage, terminal liquidation,
    reward computation) operate identically — only the price source changes.

    Each non-overlapping 252-day window becomes one episode. If the data
    contains 504 days, you get 2 episodes.
    """
    from spectral_env_core import SpectralTradingEnv
    from spectral_env_core.est_env_params import fetch_prices

    # Fetch real prices for each ticker
    all_prices = []
    min_len = float('inf')
    for ticker in tickers:
        prices = fetch_prices(ticker, period=period)
        all_prices.append(prices)
        min_len = min(min_len, len(prices))

    # Align to same length
    min_len = int(min_len)
    price_matrix = np.column_stack([p[-min_len:] for p in all_prices])
    # price_matrix shape: (min_len, num_assets)

    num_assets = len(tickers)
    num_steps  = 252  # one trading year per episode
    n_episodes = min_len // (num_steps + 1)  # non-overlapping windows

    if n_episodes < 1:
        raise ValueError(
            f"Not enough historical data for a full episode. "
            f"Got {min_len} days, need at least {num_steps + 1}. "
            f"Try a longer period (e.g. period='2y')."
        )

    # Build env kwargs
    default_kwargs = {
        "num_assets":          num_assets,
        "num_steps":           num_steps,
        "initial_price":       price_matrix[0].tolist(),
        "volatility":          0.25,
        "drift":               0.1,
        "starting_cash":       100_000,
        "max_shares":          1_000,
        "max_trade_size":      100,
        "transaction_cost_pct": 0.0005,
        "garch_alpha":         0.05,
        "garch_beta":          0.90,
    }
    if env_kwargs:
        default_kwargs.update(env_kwargs)
    default_kwargs["num_assets"] = num_assets
    default_kwargs["num_steps"]  = num_steps

    env = SpectralTradingEnv(**default_kwargs)

    episode_returns  = []
    episode_frictions = []
    equity_curves    = []

    for ep in range(n_episodes):
        # Extract the window for this episode
        start_idx = ep * num_steps
        end_idx   = start_idx + num_steps + 1  # +1 because path[0] is initial price
        window = price_matrix[start_idx:end_idx]  # shape (num_steps+1, num_assets)

        # Reset env normally (to initialise all state), then inject real prices
        obs, info = env.reset(seed=ep)
        env.brownian_path = window.copy()
        env.current_price = window[0].copy()

        # Recompute initial obs with real prices
        obs = env._get_obs()

        total_reward   = 0.0
        total_friction = 0.0
        curve = [info['cash']]

        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward   += reward
            total_friction += info.get('friction_leak', 0.0)
            portfolio_val = info['cash'] + sum(
                s * p for s, p in zip(info['shares'], info['prices'])
            )
            curve.append(portfolio_val)
            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        episode_frictions.append(total_friction)
        equity_curves.append(np.array(curve))

    returns   = np.array(episode_returns)
    frictions = np.array(episode_frictions)
    stats     = _compute_stats(returns, frictions, equity_curves)
    stats["tickers"] = tickers
    stats["period"]  = period

    return BacktestReport(
        stats=stats,
        episode_returns=returns,
        episode_frictions=frictions,
        equity_curves=equity_curves,
        mode=f"historical ({', '.join(tickers)})",
    )
