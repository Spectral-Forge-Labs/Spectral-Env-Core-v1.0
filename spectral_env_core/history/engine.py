# (c) 2026 Spectral Forge Labs. All rights reserved. 
# Use of this source code is governed by the Spectral Forge EULA.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class SpectralTradingEnv(gym.Env):
    """
    A Reinforcement Learning environment where an agent trades based on
    a price driven by Brownian motion.
    The agent can 'buy', 'hold', or 'sell' at each step, aiming for profit.
    """
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 30}

    def __init__(self,
                 num_steps=100,
                 time_total=1.0,
                 initial_price=100.0,
                 volatility=0.1,  
                 drift=0.05,
                 transaction_cost_pct=0.01, 
                 starting_cash=1000.0,
                 max_shares=10,
                 lookback_window=30,
                 phi=0,
                 df=15,
                 render_mode=None):

        super().__init__()

        # --- Environment Parameters ---
        self.num_steps = num_steps
        self.time_total = time_total
        self.initial_price = initial_price
        self.peak_price = 0.0
        self.volatility = volatility
        self.drift = drift
        self.transaction_cost_pct = transaction_cost_pct
        self.starting_cash = starting_cash
        self.max_shares = max_shares
        self.render_mode = render_mode
        self.lookback_window = lookback_window
        self.phi = phi 
        self.df = df 
        self.base_volatility = volatility

        # Calculate dt for Brownian motion
        self.dt = self.time_total / self.num_steps

        # --- Action Space ---
        # 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # --- Observation Space ---
        # plus the price history window
        # Ensure this matches your _get_obs() return length exactly
        self.num_meta_features = 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_meta_features + self.lookback_window,), 
            dtype=np.float32
        )

        # --- Internal State Initialization ---
        self.current_step = 0
        self.current_price = initial_price
        self.cash = starting_cash
        self.shares = 0
        
        # New State for Sortino Reward
        self.returns_window = [] 
        
        # Histories for rendering/tracking
        self.price_history = []
        self.cash_history = []
        self.shares_history = []
        self.portfolio_value_history = []
        self.action_history = []

        # For rendering
        self.fig = None
        self.ax = None


    def _generate_brownian_path(self):
        """
        Generates a 1D Geometric Brownian motion path with 
        Autocorrelation (AR(1)) and Kurtosis (Student's t-distribution).
        """
        # 1. Use Student's t-distribution for Kurtosis (Fat Tails)
        # df=self.df (e.g., 5) creates fatter tails than a normal distribution
        shocks = np.random.standard_t(df=self.df, size=self.num_steps) 
        
        # 2. Add Autocorrelation (AR(1) logic)
        autocorr_shocks = np.zeros(self.num_steps)
        # Initialize the first shock
        autocorr_shocks[0] = shocks[0]
        
        for t in range(1, self.num_steps):
            # Current shock is influenced by the previous one based on self.phi
            autocorr_shocks[t] = self.phi * autocorr_shocks[t-1] + shocks[t]
            
        # 3. Final Path Calculation (Geometric Brownian Motion style)
        # (mu - 0.5 * sigma^2) * dt
        drift_component = (self.drift - 0.5 * self.volatility**2) * self.dt
        
        # Combine drift and the auto-correlated stochastic component
        # S_t+dt = S_t * exp(drift_component + volatility * sqrt(dt) * Z_t)
        log_returns = drift_component + (self.volatility * np.sqrt(self.dt) * autocorr_shocks)

        # Initialize path with initial price
        # Using cumsum on log_returns provides a vectorized way to build the GBM path
        # We prepend a 0 to the cumsum to ensure the first price is exactly self.initial_price
        path = self.initial_price * np.exp(np.insert(np.cumsum(log_returns), 0, 0.0))

        # Safeguard: Ensure price doesn't go below a tiny positive value
        path[path < 1e-9] = 1e-9
        
        return path

    def _get_obs(self):
        # 1. Price Window (Normalized)
        # Ensure lookback is consistent. If history < window, pad with starting price.
        window_size = self.lookback_window
        prices = list(self.price_history)[-window_size:]
        if len(prices) < window_size:
            pad_value = self.brownian_path[0]
            prices = [pad_value] * (window_size - len(prices)) + prices
        
        # Normalize prices as % change from the current price
        norm_prices = np.array(prices) / self.current_price - 1.0

        # 2. Meta Features (Scaled to roughly -1.0 to 1.0)
        # Scaling cash/portfolio by the starting_cash helps the network stay stable
        norm_cash = self.cash / self.starting_cash
        norm_shares = self.shares / self.max_shares
        norm_portfolio = (self.cash + (self.shares * self.current_price)) / self.starting_cash

        # 3. Concatenate
        meta_features = np.array([
            norm_cash, 
            norm_shares, 
            norm_portfolio
        ], dtype=np.float32)

        return np.concatenate([norm_prices, meta_features]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Basic State Initialization
        self.current_step = 0
        self.cash = self.starting_cash 
        self.shares = 0
        self.peak_price = self.initial_price
        
        # 2. Regime Randomization (Momentum vs Mean Reversion)
        if np.random.rand() > 0.5:
            self.phi = np.random.uniform(0.1, 0.5)
        else:
            self.phi = np.random.uniform(-0.5, -0.1)
        
        # 3. Path Generation & Initial Price
        self.brownian_path = self._generate_brownian_path()
        self.current_price = self.brownian_path[self.current_step]
        
        # 4. Risk & Reward Memory (Sanitization)
        self.peak_portfolio_value = self.starting_cash
        self.returns_window = []

        # 5. History for Rendering (Start fresh to avoid flatlines)
        # We initialize as single-item lists so the plot grows from Step 0
        self.price_history = [self.current_price]
        self.shares_history = [self.shares]
        self.cash_history = [self.cash]
        
        initial_val = self.cash + (self.shares * self.current_price)
        self.portfolio_value_history = [initial_val]
        self.action_history = [] 

        # 6. Observation and Initial Frame
        # Note: Ensure _get_obs() handles padding internally if history is < lookback
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info


    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.action_history.append(action)

        # 1. Capture state BEFORE action and price move
        prev_price = self.current_price
        prev_portfolio_value = self.cash + (self.shares * self.current_price)

        # 2. Execute Action & Friction Logic
        friction_penalty = 0.0
        if action == 1:  # Buy
            cost_to_buy = self.current_price * (1 + self.transaction_cost_pct)
            if self.cash >= cost_to_buy and self.shares < self.max_shares:
                self.cash -= cost_to_buy
                self.shares += 1
                friction_penalty = self.current_price * self.transaction_cost_pct
            else:
                friction_penalty = 0.05  # invalid buy
                
        elif action == 2:  # Sell
            if self.shares > 0:
                revenue = self.current_price * (1 - self.transaction_cost_pct)
                self.cash += revenue
                self.shares -= 1
                friction_penalty = self.current_price * self.transaction_cost_pct
            else:
                friction_penalty = 0.05  # invalid sell

        # 3. Advance Time and Update Price
        self.current_step += 1
        if self.current_step < len(self.brownian_path):
            self.current_price = self.brownian_path[self.current_step]

        # 4. Trailing Stop Logic (from old version)
        if self.shares > 0:
            self.peak_price = max(self.peak_price, self.current_price)
        else:
            self.peak_price = self.current_price  # reset when flat

        trailing_stop_penalty = 0.0
        if self.shares > 0:
            drop_from_peak = (self.peak_price - self.current_price) / self.peak_price
            if drop_from_peak > 0.02:
                trailing_stop_penalty = (drop_from_peak ** 2) * 100.0

        # 5. OLD Reward Logic
        # A. Position P&L: did the price move work in your favour?
        price_delta = self.current_price - prev_price
        movement_reward = self.shares * price_delta

        # B. Idle penalty: cost for sitting out or holding without action
        idle_penalty = 0.0
        if self.shares == 0:
            idle_penalty = 0.005
        elif action == 0 and self.shares != 0:
            idle_penalty = 0.001 * abs(self.shares)

        # C. Compose final reward
        final_reward = (movement_reward - friction_penalty - idle_penalty - trailing_stop_penalty) / self.initial_price
        self.last_friction_leak = friction_penalty

        # 6. Track returns for downside risk (keep your fix)
        current_portfolio_value = self.cash + (self.shares * self.current_price)
        step_return = (current_portfolio_value - prev_portfolio_value) / max(prev_portfolio_value, 1e-9)
        self.returns_window.append(step_return)
        if len(self.returns_window) > 50:
            self.returns_window.pop(0)

        # 7. Termination
        terminated = self.current_step >= self.num_steps
        truncated = False

        # 8. History Updates
        self.price_history.append(self.current_price)
        self.cash_history.append(self.cash)
        self.shares_history.append(self.shares)
        self.portfolio_value_history.append(current_portfolio_value)

        return self._get_obs(), final_reward, terminated, truncated, self._get_info()

    def _get_info(self):
        return {
            "price": self.current_price,           # Used for your print statement
            "shares": self.shares,                 # Used for your print statement
            "friction_leak": getattr(self, 'last_friction_leak', 0.0),
            "cash": self.cash,
            "step": self.current_step
        }

    def render(self):
        """Renders the environment."""
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        if len(self.price_history) < 2:
            return  # Wait until we have at least two points to draw a line

        n = self.num_assets
        cmap = plt.cm.get_cmap('tab10', max(n, 1))
        colors = [cmap(i) for i in range(n)]
        labels = [f"Asset {i}" for i in range(n)]

        n_panes = 4
        if self.fig is None:
            plt.ion() 
            # 3 Panes: Price/Actions, Returns %, and Inventory
            self.fig, axes = plt.subplots(n_panes, 1, figsize=(12, 13), sharex=True)
            self.ax = list(axes)
            plt.tight_layout(pad=3.0)

        # 1. Clear previous frames
        for ax in self.ax:
            ax.clear()
            ax.grid(True, alpha=0.3)
        
        time_steps = np.arange(len(self.price_history))

        # --- Pane 1: Price Action & Agent Decisions ---
        self.ax[0].plot(self.price_history, color='blue', lw=1.5)
        
        # Identify Buy/Sell actions for markers
        # Assuming Action 1 = Buy, Action 2 = Sell (adjust based on your actual action space)
        buys = [i +1 for i, a in enumerate(self.action_history) if a == 1]
        sells = [i +1 for i, a in enumerate(self.action_history) if a == 2]
        
        if buys:
            self.ax[0].scatter(buys, [self.price_history[i] for i in buys], 
                               marker='^', color='green', label='Buy', zorder=5, s=100)
        if sells:
            self.ax[0].scatter(sells, [self.price_history[i] for i in sells], 
                               marker='v', color='red', label='Sell', zorder=5, s=100)
            
        self.ax[0].set_title(f"Episode Analysis | Phi: {self.phi:.2f} | Vol: {self.volatility:.2f}")
        self.ax[0].set_ylabel("Price ($)")
        handles, labels = self.ax[0].get_legend_handles_labels()
        if handles:
            self.ax[0].legend(loc='upper left')

        # --- Pane 2: Portfolio Performance (Normalized %) ---
        if len(self.portfolio_value_history) > 0:
            initial_val = self.portfolio_value_history[0]
            returns = [(v - initial_val) / initial_val * 100 for v in self.portfolio_value_history]
            
            self.ax[1].plot(time_steps, returns, color='green', label='Net Return %')
            self.ax[1].axhline(0, color='black', lw=1, ls='--')
            
            # Dynamic scaling for returns
            r_min, r_max = min(returns), max(returns)
            self.ax[1].set_ylim(r_min - 0.5, r_max + 0.5)
            
        self.ax[1].set_ylabel("Return (%)")
        self.ax[1].legend(loc='upper left')

        # --- Pane 3: Inventory Level ---
        # Use .step() instead of .fill_between() for discrete shares—it looks much more professional
        self.ax[2].step(time_steps, self.shares_history, where='post', color='purple', label='Shares')
        self.ax[2].fill_between(time_steps, self.shares_history, step='post', color='purple', alpha=0.2)
        
        self.ax[2].set_ylabel("Shares Held")
        self.ax[2].set_xlabel("Time Step")
        
        # Set a fixed height based on max_shares so the plot doesn't jump around
        self.ax[2].set_ylim(0, self.max_shares + 1) 
        self.ax[2].set_xlim(0, self.num_steps)

        # Replace draw_idle() with a forced draw
        self.fig.canvas.draw() 
        self.fig.canvas.flush_events()
        
        plt.pause(0.01) # This is non-negotiable for live updates

    def close(self):
        """"Clean up matplotlib resources. Required by the Gymnasium interface."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax  = None