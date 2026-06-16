"""
SpectralRenderWrapper
=====================
Adds live matplotlib rendering to SpectralTradingEnv.

Apply this wrapper only for human evaluation — never during parallel training.
The base SpectralTradingEnv is intentionally pickle-safe (no matplotlib state),
so SubprocVecEnv / DummyVecEnv work without this wrapper.

Usage
-----
    from spectral_env_core import SpectralTradingEnv
    from spectral_env_core.wrappers import SpectralRenderWrapper

    env = SpectralRenderWrapper(SpectralTradingEnv(...))
    obs, info = env.reset()
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


class SpectralRenderWrapper(gym.Wrapper):
    """
    Wraps SpectralTradingEnv with a live 4-pane matplotlib dashboard:

        Pane 1 — Price action with buy (▲) / sell (▼) trade markers
        Pane 2 — Portfolio return (%)
        Pane 3 — Active market regime
        Pane 4 — Shares held per asset
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.fig = None
        self.ax  = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._render_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._render_frame()
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax  = None
        self.env.close()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self):
        core = self.env  # unwrapped base env

        if len(core.price_history) < 2:
            return

        n      = core.num_assets
        cmap   = plt.cm.get_cmap('tab10', max(n, 1))
        colors = [cmap(i) for i in range(n)]
        labels = [f"Asset {i}" for i in range(n)]

        if self.fig is None:
            plt.ion()
            self.fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
            self.ax = list(axes)
            plt.tight_layout(pad=3.0)

        for ax in self.ax:
            ax.clear()
            ax.grid(True, alpha=0.3)

        # Convert deques to lists once for all panes
        price_hist   = list(core.price_history)
        shares_hist  = list(core.shares_history)
        pv_hist      = list(core.portfolio_value_history)
        regime_hist  = list(core.regime_history)
        action_hist  = list(core.action_history)
        time_steps   = np.arange(len(price_hist))

        # --- Pane 1: Price action & trade markers ---
        for i in range(n):
            prices_i = [row[i] for row in price_hist]
            self.ax[0].plot(prices_i, color=colors[i], lw=1.5, label=labels[i])

            buys = [
                t + 1 for t, a in enumerate(action_hist)
                if round(float(a[i]) * core.max_trade_size) > 0
            ]
            sells = [
                t + 1 for t, a in enumerate(action_hist)
                if round(float(a[i]) * core.max_trade_size) < 0
            ]
            if buys:
                self.ax[0].scatter(
                    buys, [prices_i[t] for t in buys],
                    marker='^', color=colors[i], zorder=5, s=80,
                )
            if sells:
                self.ax[0].scatter(
                    sells, [prices_i[t] for t in sells],
                    marker='v', color=colors[i], zorder=5, s=80,
                )

        phi_str = ', '.join(f'{p:.2f}' for p in core.phi_arr)
        self.ax[0].set_title(
            f"Episode | Assets: {n} | Phi: [{phi_str}] | "
            f"GARCH α={core.garch_alpha}, β={core.garch_beta} | "
            f"Jump λ={core.jump_intensity}"
        )
        self.ax[0].set_ylabel("Price ($)")
        if n <= 5:
            self.ax[0].legend(loc='upper left', fontsize=8)

        # --- Pane 2: Portfolio return (%) ---
        if pv_hist:
            v0      = pv_hist[0]
            returns = [(v - v0) / v0 * 100 for v in pv_hist]
            self.ax[1].plot(time_steps, returns, color='green', label='Net Return %')
            self.ax[1].axhline(0, color='black', lw=1, ls='--')
            r_min, r_max = min(returns), max(returns)
            self.ax[1].set_ylim(r_min - 0.5, r_max + 0.5)
            self.ax[1].set_ylabel("Return (%)")
            self.ax[1].legend(loc='upper left', fontsize=8)

        # --- Pane 3: Regime ---
        if regime_hist:
            self.ax[2].step(
                time_steps, regime_hist, where='post',
                color='steelblue', lw=1.5, label='Regime',
            )
            self.ax[2].fill_between(
                time_steps, regime_hist,
                color='steelblue', alpha=0.2,
            )
            self.ax[2].set_yticks(range(core.n_regimes))
            tick_labels = [
                f"R{r} (dx{core.regime_drift_mults[r]:.1f}, vx{core.regime_vol_mults[r]:.1f})"
                for r in range(core.n_regimes)
            ]
            self.ax[2].set_yticklabels(tick_labels, fontsize=7)
            self.ax[2].set_ylabel("Regime")
            self.ax[2].set_ylim(-0.5, core.n_regimes - 0.5)

        # --- Pane 4: Shares held per asset ---
        for i in range(n):
            shares_i = [int(row[i]) for row in shares_hist]
            self.ax[3].step(
                time_steps, shares_i, where='post',
                color=colors[i], label=labels[i], lw=1.5,
            )
            self.ax[3].fill_between(
                time_steps, shares_i, step='post',
                color=colors[i], alpha=0.15,
            )
        self.ax[3].set_ylabel("Shares Held")
        self.ax[3].set_xlabel("Time Step")
        self.ax[3].set_ylim(0, int(core.max_shares_arr.max()) + 1)
        self.ax[3].set_xlim(0, core.num_steps)
        if n <= 5:
            self.ax[3].legend(loc='upper left', fontsize=8)

        plt.pause(0.01)
