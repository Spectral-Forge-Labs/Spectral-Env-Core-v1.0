"""
spectral_env_core.journal
==========================
Trade journal for recording and analysing agent behaviour during episodes.

The journal is a pure observer — it doesn't modify the environment or agent.
Attach it during eval episodes to capture every step, then inspect the results
as a DataFrame, summary stats, or equity curve plot.

Usage
-----
    from spectral_env_core.journal import TradeJournal

    journal = TradeJournal()

    obs, info = env.reset()
    journal.begin_episode(info)

    for _ in range(env.num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        journal.record_step(action, reward, info)
        if terminated or truncated:
            break

    journal.end_episode(info)

    # Inspect
    df = journal.to_dataframe()
    print(journal.summary())
    journal.plot_equity_curve()

    # Multi-episode usage — just loop begin/end:
    for ep in range(20):
        obs, info = env.reset()
        journal.begin_episode(info)
        ...
        journal.end_episode(info)

    df = journal.to_dataframe()  # all episodes concatenated
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _EpisodeRecord:
    """Internal storage for a single episode."""
    episode_id: int = 0
    steps: list = field(default_factory=list)
    initial_cash: float = 0.0
    initial_prices: list = field(default_factory=list)


class TradeJournal:
    """
    Records step-by-step trading data across one or more episodes.

    Attributes
    ----------
    episodes : list[_EpisodeRecord]
        All recorded episodes.
    """

    def __init__(self):
        self.episodes: list[_EpisodeRecord] = []
        self._current: Optional[_EpisodeRecord] = None
        self._episode_counter = 0

    def begin_episode(self, info: dict) -> None:
        """
        Start recording a new episode.

        Parameters
        ----------
        info : dict
            The info dict returned by env.reset().
        """
        self._episode_counter += 1
        self._current = _EpisodeRecord(
            episode_id=self._episode_counter,
            initial_cash=info.get("cash", 0.0),
            initial_prices=list(info.get("prices", [])),
        )

    def record_step(self, action, reward: float, info: dict) -> None:
        """
        Record a single step.

        Parameters
        ----------
        action : array-like
            The action taken (raw from model.predict).
        reward : float
            Step reward.
        info : dict
            The info dict returned by env.step().
        """
        if self._current is None:
            raise RuntimeError("Call begin_episode() before record_step().")

        action_list = np.atleast_1d(action).tolist()
        self._current.steps.append({
            "episode":   self._current.episode_id,
            "step":      info.get("step", len(self._current.steps)),
            "action":    action_list,
            "prices":    list(info.get("prices", [])),
            "shares":    list(info.get("shares", [])),
            "cash":      info.get("cash", 0.0),
            "reward":    reward,
            "friction":  info.get("friction_leak", 0.0),
            "regime":    info.get("regime", 0),
        })

    def end_episode(self, info: Optional[dict] = None) -> None:
        """
        Finalise the current episode and store it.

        Parameters
        ----------
        info : dict, optional
            Final info dict (for terminal state recording).
        """
        if self._current is None:
            return
        self.episodes.append(self._current)
        self._current = None

    def to_dataframe(self):
        """
        Convert all recorded steps into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: episode, step, action, prices, shares, cash, reward,
                     friction, regime, portfolio_value
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required: pip install pandas")

        rows = []
        for ep in self.episodes:
            for step_data in ep.steps:
                row = step_data.copy()
                # Compute portfolio value
                prices = row["prices"]
                shares = row["shares"]
                cash   = row["cash"]
                port_val = cash + sum(s * p for s, p in zip(shares, prices))
                row["portfolio_value"] = port_val
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def summary(self) -> dict:
        """
        Compute summary statistics across all recorded episodes.

        Returns
        -------
        dict with keys:
            n_episodes, mean_return, std_return, sharpe_ratio,
            max_drawdown, win_rate, total_trades, avg_trades_per_episode,
            avg_hold_time, avg_friction_per_episode, profit_factor
        """
        if not self.episodes:
            return {}

        episode_returns  = []
        episode_trades   = []
        episode_frictions = []
        max_drawdown     = 0.0
        all_hold_times   = []

        for ep in self.episodes:
            if not ep.steps:
                continue

            # Episode return
            initial_val = ep.initial_cash
            final_step = ep.steps[-1]
            final_val = final_step["cash"] + sum(
                s * p for s, p in zip(final_step["shares"], final_step["prices"])
            )
            ret_pct = (final_val / initial_val - 1.0) * 100
            episode_returns.append(ret_pct)

            # Friction
            ep_friction = sum(s["friction"] for s in ep.steps)
            episode_frictions.append(ep_friction)

            # Count trades (steps where action magnitude > 0.05)
            n_trades = sum(
                1 for s in ep.steps
                if any(abs(a) > 0.05 for a in s["action"])
            )
            episode_trades.append(n_trades)

            # Drawdown
            values = [initial_val]
            for s in ep.steps:
                v = s["cash"] + sum(
                    sh * pr for sh, pr in zip(s["shares"], s["prices"])
                )
                values.append(v)
            values = np.array(values)
            running_max = np.maximum.accumulate(values)
            dd = (values - running_max) / running_max * 100
            ep_max_dd = abs(float(dd.min()))
            max_drawdown = max(max_drawdown, ep_max_dd)

            # Hold time approximation: count consecutive steps with non-zero position
            in_position = False
            hold_start = 0
            for s in ep.steps:
                has_pos = any(sh > 0 for sh in s["shares"])
                if has_pos and not in_position:
                    in_position = True
                    hold_start = s["step"]
                elif not has_pos and in_position:
                    in_position = False
                    all_hold_times.append(s["step"] - hold_start)
            # If still holding at end of episode
            if in_position and ep.steps:
                all_hold_times.append(ep.steps[-1]["step"] - hold_start)

        returns = np.array(episode_returns)
        mean_r  = float(returns.mean()) if len(returns) > 0 else 0.0
        std_r   = float(returns.std()) if len(returns) > 1 else 1.0

        # Sharpe
        sharpe = mean_r / std_r if std_r > 0 else 0.0

        # Win rate
        win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

        # Profit factor
        gross_profit = float(returns[returns > 0].sum()) if (returns > 0).any() else 0.0
        gross_loss   = abs(float(returns[returns < 0].sum())) if (returns < 0).any() else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Avg hold time
        avg_hold = float(np.mean(all_hold_times)) if all_hold_times else 0.0

        return {
            "n_episodes":              len(episode_returns),
            "mean_return":             mean_r,
            "std_return":              std_r,
            "sharpe_ratio":            sharpe,
            "max_drawdown":            max_drawdown,
            "win_rate":                win_rate,
            "total_trades":            sum(episode_trades),
            "avg_trades_per_episode":  float(np.mean(episode_trades)) if episode_trades else 0.0,
            "avg_hold_time":           avg_hold,
            "avg_friction_per_episode": float(np.mean(episode_frictions)) if episode_frictions else 0.0,
            "profit_factor":           profit_factor,
        }

    def plot_equity_curve(self, max_episodes: int = 10):
        """
        Plot equity curves for recorded episodes.

        Parameters
        ----------
        max_episodes : int
            Maximum number of episodes to plot (avoids clutter).
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

        n_plot = min(max_episodes, len(self.episodes))

        for ep in self.episodes[:n_plot]:
            if not ep.steps:
                continue
            values = [ep.initial_cash]
            for s in ep.steps:
                v = s["cash"] + sum(
                    sh * pr for sh, pr in zip(s["shares"], s["prices"])
                )
                values.append(v)

            ret_pct = [(v / values[0] - 1) * 100 for v in values]
            axes[0].plot(ret_pct, alpha=0.6, lw=1)

        axes[0].axhline(0, color='black', lw=1, ls='--')
        axes[0].set_title(f'Portfolio Return (%) — {n_plot} episodes', fontsize=12)
        axes[0].set_ylabel('Return (%)')
        axes[0].set_xlabel('Step')
        axes[0].grid(True, alpha=0.3)

        # Cumulative friction per episode
        for ep in self.episodes[:n_plot]:
            if not ep.steps:
                continue
            cum_friction = np.cumsum([s["friction"] for s in ep.steps])
            axes[1].plot(cum_friction, alpha=0.6, lw=1)

        axes[1].set_title('Cumulative Friction ($) per Episode', fontsize=12)
        axes[1].set_ylabel('Friction ($)')
        axes[1].set_xlabel('Step')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        s = self.summary()
        if not s:
            print("No episodes recorded.")
            return

        print(f"\n{'═' * 45}")
        print(f" Trade Journal Summary")
        print(f"{'═' * 45}")
        print(f" Episodes:           {s['n_episodes']}")
        print(f" Mean Return:        {s['mean_return']:+.2f}%")
        print(f" Std Return:         {s['std_return']:.2f}%")
        print(f" Sharpe Ratio:       {s['sharpe_ratio']:.3f}")
        print(f" Max Drawdown:       {s['max_drawdown']:.2f}%")
        print(f" Win Rate:           {s['win_rate']:.1%}")
        print(f" Profit Factor:      {s['profit_factor']:.2f}")
        print(f" Total Trades:       {s['total_trades']}")
        print(f" Avg Trades/Episode: {s['avg_trades_per_episode']:.1f}")
        print(f" Avg Hold Time:      {s['avg_hold_time']:.1f} steps")
        print(f" Avg Friction/Ep:    ${s['avg_friction_per_episode']:,.2f}")
        print(f"{'═' * 45}")
