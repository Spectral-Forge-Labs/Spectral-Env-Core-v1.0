import gymnasium as gym
from gymnasium import spaces
from collections import deque
import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# Vectorised GARCH(1,1) kernel
# ---------------------------------------------------------------------------

def _garch_vectorised(
    z: np.ndarray,
    regime_seq: np.ndarray,
    volatility_arr: np.ndarray,
    regime_vol_mults: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    max_h: float = 25.0,
) -> np.ndarray:
    """
    Compute time-varying conditional volatility via GARCH(1,1).

    Keeps the per-step recurrence but avoids all Python-level attribute
    lookups and array allocations inside the loop by working with
    pre-extracted scalars and pre-allocated output.

    Parameters
    ----------
    z            : (num_steps, num_assets) standardised shocks
    regime_seq   : (num_steps,) integer regime indices
    volatility_arr: (num_assets,) base annualised volatilities
    regime_vol_mults : (n_regimes,) volatility multipliers per regime
    omega, alpha, beta : GARCH(1,1) scalar parameters
    max_h        : variance cap to prevent overflow (default 25x long-run)

    Returns
    -------
    vol_t : (num_steps, num_assets)
    """
    num_steps, num_assets = z.shape
    vol_t   = np.empty((num_steps, num_assets), dtype=np.float64)
    h       = np.ones(num_assets, dtype=np.float64)
    prev_z2 = np.ones(num_assets, dtype=np.float64)

    for t in range(num_steps):
        h = omega + alpha * prev_z2 + beta * h
        # Cap h to prevent exp() overflow in price path
        np.clip(h, 1e-8, max_h, out=h)
        vol_baseline = volatility_arr * regime_vol_mults[regime_seq[t]]
        vol_t[t]     = vol_baseline * np.sqrt(h)
        prev_z2      = z[t] ** 2

    return vol_t


# ---------------------------------------------------------------------------
# Vectorised Markov regime sequence
# ---------------------------------------------------------------------------

def _regime_sequence_vectorised(
    np_random,
    num_steps: int,
    n_regimes: int,
    switch_prob: float,
) -> np.ndarray:
    """
    Generate a Markov-chain regime sequence without a Python conditional loop.

    Strategy: pre-draw all switch decisions and candidate regimes, then
    resolve the sequence with a single cumulative pass that only touches
    Python integers at switch points (typically rare).
    """
    seq         = np.empty(num_steps, dtype=np.int32)
    switch_mask = np_random.random(num_steps) < switch_prob
    # Candidates in [0, n_regimes-1]; we'll offset those >= current below
    candidates  = np_random.integers(0, max(n_regimes - 1, 1), size=num_steps)

    regime = int(np_random.integers(0, n_regimes))
    for t in range(num_steps):
        seq[t] = regime
        if n_regimes > 1 and switch_mask[t]:
            c = int(candidates[t])
            if c >= regime:
                c += 1
            regime = c
    return seq


class SpectralTradingEnv(gym.Env):
    """
    High-fidelity stochastic trading environment for RL research.

    Price paths combine:
      - Student-t fat tails
      - AR(1) autocorrelation (scipy lfilter)
      - Inter-asset Cholesky correlation
      - GARCH(1,1) conditional volatility
      - Markov regime switching
      - Merton (1976) jump diffusion

    Rendering has been fully decoupled from the core environment.
    Use SpectralRenderWrapper (wrappers.py) for human-mode visualisation.
    This class is pickle-safe and compatible with SubprocVecEnv.
    """

    # render_mode='human' is handled by SpectralRenderWrapper, not here.
    metadata = {'render_modes': ['none'], 'render_fps': 30}

    def __init__(
        self,
        num_assets: int = 1,
        num_steps: int = 100,
        time_total: float = 1.0,
        initial_price=100.0,
        volatility=0.1,
        drift=0.5,
        transaction_cost_pct: float = 0.01,
        starting_cash: float = 1000.0,
        max_shares=10,
        max_trade_size: int = 1,
        lookback_window: int = 30,
        phi: float = 0.0,
        df: int = 15,
        correlation: np.ndarray = None,
        # GARCH
        garch_alpha: float = 0.05,
        garch_beta: float = 0.90,
        # Regime switching
        n_regimes: int = 2,
        regime_drift_mults=(1.5, -0.5),
        regime_vol_mults=(0.7, 1.8),
        regime_switch_prob: float = 0.05,
        # Jump diffusion
        jump_intensity: float = 0.0,
        jump_mean: float = -0.05,
        jump_std: float = 0.07,
        # Risk controls
        bankruptcy_threshold: float = 0.1,
        trailing_stop_pct: float = 0.15,
        hard_trailing_stop: bool = False,
        # Price floor — below this fraction of initial_price the asset is
        # treated as delisted and the episode truncates.
        price_floor_pct: float = 0.01,
        # Fractional positions — set True for crypto / fractional share trading.
        # When True:
        #   - shares are stored as float64 (e.g. 0.00312 BTC)
        #   - max_trade_size is interpreted as max notional USD per trade
        #   - max_shares is interpreted as max position value in USD per asset
        #   - trade sizing maps action → notional / price (no rounding)
        fractional: bool = False,
        # AR(1) behaviour
        randomize_phi: bool = True,
        render_mode=None,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.fractional = fractional

        # --- Per-asset parameter arrays ---
        self.initial_price_arr = self._broadcast(initial_price, num_assets, 'initial_price', float)
        self.volatility_arr    = self._broadcast(volatility,    num_assets, 'volatility',    float)
        self.drift_arr         = self._broadcast(drift,         num_assets, 'drift',         float)

        # max_shares_arr:
        #   integer mode → max share count per asset (int)
        #   fractional mode → max position value in USD per asset (float)
        if fractional:
            self.max_shares_arr = self._broadcast(max_shares, num_assets, 'max_shares', float)
        else:
            self.max_shares_arr = self._broadcast(max_shares, num_assets, 'max_shares', int)

        # Scalar aliases (backward compat / normalisation reference)
        self.initial_price   = float(self.initial_price_arr.mean())
        self.volatility      = float(self.volatility_arr.mean())
        self.base_volatility = self.volatility

        # --- Correlation (Cholesky factor with PSD regularisation) ---
        self._chol = None
        if correlation is not None and num_assets > 1:
            corr = np.asarray(correlation, dtype=float)
            if corr.shape != (num_assets, num_assets):
                raise ValueError(
                    f"correlation must be ({num_assets}, {num_assets}), got {corr.shape}"
                )
            # Eigenvalue-floor regularisation: guarantees PSD even for
            # empirically estimated near-singular matrices.
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.maximum(eigvals, 1e-6)
            corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Re-normalise back to a correlation matrix
            d = np.sqrt(np.diag(corr_psd))
            corr_psd = corr_psd / np.outer(d, d)
            self._chol = np.linalg.cholesky(corr_psd)

        # --- Scalar env parameters ---
        self.num_steps            = num_steps
        self.time_total           = time_total
        self.transaction_cost_pct = transaction_cost_pct
        self.starting_cash        = starting_cash
        self.max_trade_size       = max_trade_size
        self.lookback_window      = lookback_window
        self.phi                  = phi
        self.df                   = df
        self.bankruptcy_threshold = bankruptcy_threshold
        self.trailing_stop_pct    = trailing_stop_pct
        self.hard_trailing_stop   = hard_trailing_stop
        self.price_floor_pct      = price_floor_pct
        self.randomize_phi        = randomize_phi
        self.render_mode          = render_mode
        self.dt                   = time_total / num_steps

        # Absolute price floors per asset (computed once)
        self._price_floors = self.initial_price_arr * price_floor_pct

        # --- t-distribution variance correction: Var(t_df) = df/(df-2) ---
        self._t_std_correction = np.sqrt(df / (df - 2))

        # --- GARCH(1,1) parameters ---
        if garch_alpha + garch_beta >= 1.0:
            raise ValueError(
                f"GARCH requires alpha + beta < 1 for stationarity, "
                f"got {garch_alpha} + {garch_beta} = {garch_alpha + garch_beta:.4f}."
            )
        self.garch_alpha  = garch_alpha
        self.garch_beta   = garch_beta
        self._garch_omega = 1.0 - garch_alpha - garch_beta

        # --- Regime switching parameters ---
        regime_drift_mults = list(regime_drift_mults)
        regime_vol_mults   = list(regime_vol_mults)
        if len(regime_drift_mults) != n_regimes:
            raise ValueError(
                f"regime_drift_mults must have length {n_regimes}, "
                f"got {len(regime_drift_mults)}."
            )
        if len(regime_vol_mults) != n_regimes:
            raise ValueError(
                f"regime_vol_mults must have length {n_regimes}, "
                f"got {len(regime_vol_mults)}."
            )
        if any(v <= 0 for v in regime_vol_mults):
            raise ValueError("All regime_vol_mults must be positive.")
        self.n_regimes          = n_regimes
        self.regime_drift_mults = np.array(regime_drift_mults, dtype=float)
        self.regime_vol_mults   = np.array(regime_vol_mults,   dtype=float)
        self.regime_switch_prob = float(regime_switch_prob)

        # --- Jump diffusion parameters ---
        # jump_mean is the mean *log-jump size* (log-return units).
        # Drift compensation inside _generate_jumps() uses this convention.
        if jump_std <= 0:
            raise ValueError("jump_std must be positive.")
        self.jump_intensity = float(jump_intensity)
        self.jump_mean      = float(jump_mean)
        self.jump_std       = float(jump_std)

        # --- Action space: continuous [-1, 1] per asset ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_assets,), dtype=np.float32
        )

        # --- Observation space: flat float32 vector ---
        self.num_price_features = num_assets * lookback_window
        self.num_meta_features  = num_assets + 4  # shares(N) + cash + portfolio + unrealised_exit_cost + time_remaining
        obs_dim = self.num_price_features + self.num_meta_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Internal state (properly initialised by reset()) ---
        self.current_step    = 0
        self.current_price   = self.initial_price_arr.copy()
        self.cash            = starting_cash
        self._shares_dtype   = np.float64 if fractional else np.int32
        self.shares          = np.zeros(num_assets, dtype=self._shares_dtype)
        self.peak_price      = self.initial_price_arr.copy()
        self.phi_arr         = np.full(num_assets, phi, dtype=float)
        self.current_regime  = 0
        self.regime_sequence = np.zeros(num_steps, dtype=int)

        # Bounded history deques — O(1) append, bounded memory.
        # Full-episode histories are available via self.brownian_path and
        # self.regime_sequence after reset(); these deques serve rendering
        # and any per-step diagnostics only.
        _hist = lookback_window + 1
        self.price_history           = deque(maxlen=_hist)
        self.cash_history            = deque(maxlen=_hist)
        self.shares_history          = deque(maxlen=_hist)
        self.portfolio_value_history = deque(maxlen=_hist)
        self.action_history          = deque(maxlen=_hist)
        self.regime_history          = deque(maxlen=_hist)

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _broadcast(value, n: int, name: str, dtype) -> np.ndarray:
        """Convert scalar or sequence to a numpy array of length n."""
        arr = np.atleast_1d(np.asarray(value, dtype=dtype))
        if arr.size == 1:
            return np.full(n, arr.item(), dtype=dtype)
        if arr.size != n:
            raise ValueError(
                f"'{name}' must be a scalar or have length {n}, got {arr.size}."
            )
        return arr.copy()

    @staticmethod
    def _ar1_filter(x: np.ndarray, phi: float) -> np.ndarray:
        """
        Apply AR(1) recurrence y[t] = phi * y[t-1] + x[t] via scipy lfilter.
        Equivalent to the previous Python loop but runs in C.
        """
        return lfilter([1.0], [1.0, -phi], x)

    # -----------------------------------------------------------------------
    # Path generation sub-routines
    # -----------------------------------------------------------------------

    def _generate_regime_sequence(self) -> np.ndarray:
        """
        Draw a Markov-chain regime sequence of length num_steps.
        Delegates to the module-level vectorised helper.
        """
        return _regime_sequence_vectorised(
            self.np_random,
            self.num_steps,
            self.n_regimes,
            self.regime_switch_prob,
        )

    def _apply_garch(self, z: np.ndarray, regime_seq: np.ndarray) -> np.ndarray:
        """
        Compute time-varying conditional volatility via GARCH(1,1).
        Delegates to the module-level vectorised kernel.
        """
        return _garch_vectorised(
            z,
            regime_seq,
            self.volatility_arr,
            self.regime_vol_mults,
            self._garch_omega,
            self.garch_alpha,
            self.garch_beta,
        )

    def _generate_jumps(self) -> np.ndarray:
        """
        Generate log-return contribution from Merton (1976) jump diffusion.

        jump_mean is treated as the mean log-jump size. Drift compensation
        uses the standard risk-neutral correction: λ * μ_J * dt.
        """
        if self.jump_intensity <= 0.0:
            return np.zeros((self.num_steps, self.num_assets))

        lambda_dt = self.jump_intensity * self.dt

        n_jumps = self.np_random.poisson(
            lambda_dt, size=(self.num_steps, self.num_assets)
        )

        jump_component = np.zeros((self.num_steps, self.num_assets))
        mask = n_jumps > 0
        if mask.any():
            n_nz = n_jumps[mask]
            jump_component[mask] = (
                n_nz * self.jump_mean
                + np.sqrt(n_nz) * self.jump_std
                * self.np_random.standard_normal(n_nz.shape)
            )

        # Risk-neutral drift compensation
        jump_component -= self.jump_intensity * self.jump_mean * self.dt
        return jump_component

    # -----------------------------------------------------------------------
    # Main path generator
    # -----------------------------------------------------------------------

    def _generate_brownian_path(self) -> np.ndarray:
        """
        Generate price paths for all assets incorporating:
          - Student-t fat tails
          - Inter-asset correlation (Cholesky)
          - AR(1) autocorrelation (scipy lfilter)
          - GARCH(1,1) conditional volatility
          - Markov regime switching
          - Merton jump diffusion

        Returns
        -------
        path : np.ndarray, shape (num_steps + 1, num_assets)
            path[0] is the initial price; path[t] is the price after step t.
        """
        # 1. Regime sequence
        regime_seq = self._generate_regime_sequence()
        self.regime_sequence = regime_seq

        # 2. Fat-tail standardised shocks, unit variance
        raw_shocks = self.np_random.standard_t(
            df=self.df, size=(self.num_steps, self.num_assets)
        )
        z = raw_shocks / self._t_std_correction

        # 3. Inter-asset correlation via Cholesky
        if self._chol is not None:
            z = z @ self._chol.T

        # 4. AR(1) autocorrelation — variance-preserving scaling, per asset
        for i in range(self.num_assets):
            phi_i = self.phi_arr[i]
            if phi_i != 0.0:
                z[:, i] = self._ar1_filter(
                    z[:, i] * np.sqrt(1.0 - phi_i ** 2), phi_i
                )

        # 5. GARCH(1,1) + regime vol
        vol_t = self._apply_garch(z, regime_seq)

        # 6. Regime-dependent drift
        regime_drift_mults = self.regime_drift_mults[regime_seq]
        drift_t = self.drift_arr * regime_drift_mults[:, None]

        # 7. Jump diffusion
        jump_t = self._generate_jumps()

        # 8. Log-returns (Itô correction applied to drift)
        log_returns = (
            (drift_t - 0.5 * vol_t ** 2) * self.dt
            + vol_t * np.sqrt(self.dt) * z
            + jump_t
        )

        # Clamp log-returns to ±5 to guard against extreme GARCH blowup
        np.clip(log_returns, -5.0, 5.0, out=log_returns)

        # 9. Convert to price path via cumulative sum
        log_cumsum = np.vstack(
            [np.zeros((1, self.num_assets)), np.cumsum(log_returns, axis=0)]
        )
        path = self.initial_price_arr * np.exp(log_cumsum)

        # Floor: replace values below price_floor with the floor itself
        # (preserves array shape; delisting detection happens in step())
        return np.maximum(path, self._price_floors)

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Flat float32 observation vector:
            [norm_price_window_asset_0, ..., norm_price_window_asset_N,
             norm_cash, norm_shares_0, ..., norm_shares_N, norm_portfolio]

        Price windows are % deviation from each asset's current price.
        Slices directly into brownian_path — no Python list iteration.
        """
        step = self.current_step
        price_features = []

        for i in range(self.num_assets):
            start = max(0, step - self.lookback_window + 1)
            window = self.brownian_path[start: step + 1, i]

            if len(window) < self.lookback_window:
                pad   = np.full(self.lookback_window - len(window), window[0], dtype=np.float32)
                window = np.concatenate([pad, window])

            norm = (window / self.current_price[i] - 1.0).astype(np.float32)
            price_features.append(norm)

        norm_cash      = self.cash / self.starting_cash
        # Fractional mode: max_shares_arr is in USD, normalise by position value / max_value
        # Integer mode: normalise by share count / max share count
        if self.fractional:
            position_values = self.shares * self.current_price
            norm_shares     = position_values / self.max_shares_arr
        else:
            norm_shares = self.shares / self.max_shares_arr
        norm_portfolio = (
            self.cash + float(np.sum(self.shares * self.current_price))
        ) / self.starting_cash

        # Unrealised exit cost — what the agent would owe in sell fees if it
        # liquidated its entire position right now. Makes the deferred liability
        # of holding visible to the policy, aiding temporal credit assignment.
        unrealised_exit_cost = (
            float(np.sum(self.shares * self.current_price)) * self.transaction_cost_pct
            / self.starting_cash
        )

        # Time remaining in episode [1.0 → 0.0]. Gives the critic an explicit
        # structural hint that the terminal liquidation cliff is approaching,
        # without distorting the MDP objective via reward shaping.
        time_remaining = 1.0 - (self.current_step / self.num_steps)

        meta = np.array(
            [norm_cash, *norm_shares.tolist(), norm_portfolio, unrealised_exit_cost, time_remaining],
            dtype=np.float32
        )
        return np.concatenate([*price_features, meta]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step       = 0
        self.cash               = self.starting_cash
        self.shares             = np.zeros(self.num_assets, dtype=self._shares_dtype)
        self.last_friction_leak = 0.0

        # Per-asset AR(1) coefficients.
        # randomize_phi=True: sample magnitude from U[0.05, 0.40] with random sign —
        #   agent trains across momentum and mean-reversion environments.
        # randomize_phi=False: use self.phi exactly (0.0 → i.i.d. shocks).
        if self.randomize_phi:
            mags     = self.np_random.uniform(0.05, 0.40, size=self.num_assets)
            signs    = np.where(self.np_random.random(self.num_assets) > 0.5, 1.0, -1.0)
            self.phi_arr = mags * signs
        else:
            self.phi_arr = np.full(self.num_assets, self.phi, dtype=float)

        # Generate full episode price paths
        self.brownian_path  = self._generate_brownian_path()
        self.current_price  = self.brownian_path[0, :].copy()
        self.current_regime = int(self.regime_sequence[0])

        # Passive equal-weight buy-and-hold benchmark
        budget_per_asset = self.starting_cash / self.num_assets
        if self.fractional:
            # Fractional: buy exactly budget / price units per asset,
            # capped by max position value (max_shares_arr in USD)
            raw_shares = budget_per_asset / self.brownian_path[0]
            max_units  = self.max_shares_arr / self.brownian_path[0]
            self._benchmark_shares = np.minimum(raw_shares, max_units)
        else:
            self._benchmark_shares = np.minimum(
                np.floor(
                    budget_per_asset / (self.brownian_path[0] * (1.0 + self.transaction_cost_pct))
                ).astype(np.int32),
                self.max_shares_arr,
            )

        self.peak_price           = self.current_price.copy()
        self.peak_portfolio_value = self.starting_cash

        # Seed bounded history deques
        for h in (self.price_history, self.cash_history, self.shares_history,
                  self.portfolio_value_history, self.action_history, self.regime_history):
            h.clear()

        self.price_history.append(self.current_price.copy())
        self.shares_history.append(self.shares.copy())
        self.cash_history.append(self.cash)
        self.portfolio_value_history.append(
            self.cash + float(np.sum(self.shares * self.current_price))
        )
        self.regime_history.append(self.current_regime)

        return self._get_obs(), self._get_info()

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action):

        action = np.atleast_1d(np.asarray(action, dtype=np.float32))
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action.tolist()}. Must be in {self.action_space}."
            )

        self.action_history.append(action.copy())

        # 1. Pre-action snapshot
        prev_price           = self.current_price.copy()
        prev_portfolio_value = self.cash + float(np.sum(self.shares * self.current_price))  # noqa: F841

        # 2. Execute per-asset trades
        friction_penalty = 0.0
        action = np.where(np.abs(action) < 0.05, 0.0, action)
        scaled = np.clip(action, -1.0, 1.0) * self.max_trade_size

        if self.fractional:
            # Fractional mode: scaled is notional USD, convert to units
            trades_units = scaled / self.current_price  # fractional shares
        else:
            # Integer mode: scaled is share count, round to nearest int
            trades_units = np.round(scaled).astype(int)

        for i in range(self.num_assets):
            qty     = trades_units[i]
            price_i = self.current_price[i]
            cost_pct = self.transaction_cost_pct

            if qty > 0:
                # Buys: no transaction fee
                max_affordable = self.cash / price_i  # fractional units affordable
                if self.fractional:
                    # Cap by max position value (max_shares_arr in USD)
                    current_value = self.shares[i] * price_i
                    room = max(0.0, (self.max_shares_arr[i] - current_value) / price_i)
                    actual_qty = min(float(qty), float(max_affordable), float(room))
                else:
                    max_affordable = int(max_affordable)
                    room = int(self.max_shares_arr[i]) - int(self.shares[i])
                    actual_qty = max(0, min(int(qty), max_affordable, room))

                self.cash      -= actual_qty * price_i
                self.shares[i] += actual_qty

            elif qty < 0:
                # Sells: transaction fee applied to proceeds
                if self.fractional:
                    actual_qty = min(abs(float(qty)), float(self.shares[i]))
                else:
                    actual_qty = max(0, min(int(-qty), int(self.shares[i])))

                self.cash        += actual_qty * price_i * (1 - cost_pct)
                self.shares[i]   -= actual_qty
                friction_penalty += actual_qty * price_i * cost_pct

        # 3. Advance time
        self.current_step += 1
        if self.current_step < len(self.brownian_path):
            self.current_price  = self.brownian_path[self.current_step, :].copy()
            self.current_regime = int(
                self.regime_sequence[min(self.current_step, self.num_steps - 1)]
            )

        # 4. Trailing stop — per asset
        trailing_stop_penalty = 0.0
        hard_stop_triggered   = False

        for i in range(self.num_assets):
            if self.shares[i] > 0:
                self.peak_price[i] = max(self.peak_price[i], self.current_price[i])
                drop = (self.peak_price[i] - self.current_price[i]) / self.peak_price[i]
                if drop > 0.02:
                    if self.hard_trailing_stop and drop > self.trailing_stop_pct:
                        qty = float(self.shares[i]) if self.fractional else int(self.shares[i])
                        self.cash        += qty * self.current_price[i] * (1 - self.transaction_cost_pct)
                        friction_penalty += qty * self.current_price[i] * self.transaction_cost_pct
                        self.shares[i]    = 0
                        # Reset peak so a re-entry doesn't inherit the old high
                        self.peak_price[i] = self.current_price[i]
                        hard_stop_triggered = True
                    else:
                        position_value         = float(self.shares[i] * self.current_price[i])
                        trailing_stop_penalty += (drop ** 2) * position_value
            else:
                self.peak_price[i] = self.current_price[i]

        # 5. Reward — alpha-relative PnL as % of starting_cash
        price_delta     = self.current_price - prev_price
        movement_reward = float(np.sum(self.shares * price_delta))
        market_move     = float(np.sum(self._benchmark_shares * price_delta))
        alpha_pnl       = movement_reward - market_move

        raw_reward              = alpha_pnl - friction_penalty - trailing_stop_penalty
        final_reward            = float((raw_reward / self.starting_cash) * 100)
        self.last_friction_leak = friction_penalty

        # 6. Termination / truncation
        current_portfolio_value = self.cash + float(np.sum(self.shares * self.current_price))
        terminated  = int(self.current_step) >= self.num_steps
        bankruptcy  = current_portfolio_value < self.bankruptcy_threshold * self.starting_cash
        # Delisting: any asset price has hit its absolute floor
        delisting   = bool(np.any(self.current_price <= self._price_floors))
        truncated   = hard_stop_triggered or bankruptcy or delisting

        # Forced terminal liquidation — applied when the episode ends naturally
        # at step num_steps. All remaining shares are sold at current price with
        # the standard sell-side fee. This closes the loophole where the agent
        # could hold positions indefinitely and never incur exit costs.
        # The terminal friction hits the final reward so the value function
        # learns to discount large near-terminal positions appropriately.
        if terminated and not truncated:
            terminal_friction = 0.0
            for i in range(self.num_assets):
                if self.shares[i] > 0:
                    qty = float(self.shares[i])
                    proceeds               = qty * self.current_price[i]
                    fee                    = proceeds * self.transaction_cost_pct
                    self.cash             += proceeds - fee
                    terminal_friction     += fee
                    self.shares[i]         = 0
            # Deduct terminal friction from the final step reward
            final_reward -= float((terminal_friction / self.starting_cash) * 100)
            self.last_friction_leak += terminal_friction

        # 7. History updates
        self.price_history.append(self.current_price.copy())
        self.cash_history.append(self.cash)
        self.shares_history.append(self.shares.copy())
        self.portfolio_value_history.append(current_portfolio_value)
        self.regime_history.append(self.current_regime)

        return self._get_obs(), final_reward, terminated, truncated, self._get_info()

    # -----------------------------------------------------------------------
    # Info / render stubs / cleanup
    # -----------------------------------------------------------------------

    def _get_info(self) -> dict:
        return {
            "prices":        self.current_price.tolist(),
            "shares":        self.shares.tolist(),
            "cash":          self.cash,
            "step":          self.current_step,
            "regime":        self.current_regime,
            "friction_leak": getattr(self, 'last_friction_leak', 0.0),
            "fractional":    self.fractional,
        }

    def render(self):
        """
        Rendering is handled by SpectralRenderWrapper (wrappers.py).
        Calling render() on the base env is a no-op.
        """

    def close(self):
        """No resources to release in the base environment."""
