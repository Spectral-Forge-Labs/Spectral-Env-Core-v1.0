# Spectral-Env-Core — Lite vs Pro Product Design

## Overview

Two separate packages distributed from separate repositories:

- **Spectral-Env-Lite** — Free, open-source (MIT or similar permissive license). Public GitHub repo. Serves as a "try before you buy" funnel for academics, hobbyists, and researchers evaluating the platform.
- **Spectral-Env-Core (Pro)** — Source-available commercial license via Polar.sh. Private/gated repo. Full feature set for serious quant research and deployment.

Both packages share the same Gymnasium registration ID (`SpectralEnv-v1`) and API surface so upgrading from lite to pro requires only a package swap — no code changes in user training scripts.

---

## Distribution & Repo Structure

### Lite (Public Repo)
```
Spectral-Env-Lite/
├── spectral_env_lite/
│   ├── __init__.py           # Exports SpectralTradingEnv, registers SpectralEnv-v1
│   └── engine.py             # Single-file environment (rendering included)
├── examples/
│   └── train_basic.ipynb     # Simple SB3 PPO training notebook (single asset)
├── pyproject.toml
├── README.md
└── LICENSE                   # MIT
```

### Pro (Private/Gated Repo) — Current Codebase
```
Spectral-Env-Core/
├── spectral_env_core/
│   ├── __init__.py
│   ├── engine2.py            # Full environment (pickle-safe, multi-asset, fractional)
│   ├── wrappers.py           # SpectralRenderWrapper (decoupled rendering)
│   ├── est_env_params.py     # Parameter estimation from real market data
│   └── history/
│       └── engine.py         # Legacy
├── train.py                  # Full parallel training with diagnostics
├── alpaca_rl_trader.py       # Live/paper trading bridge
├── logs/
│   └── examine.ipynb         # Full diagnostic dashboard
├── pyproject.toml
├── README.md
└── LICENSE.txt               # Source-available commercial
```

---

## Feature Comparison Matrix

| Category | Feature | Lite | Pro |
|---|---|---|---|
| **Assets** | Single asset | ✅ | ✅ |
| | Multi-asset (N simultaneous) | ❌ | ✅ |
| | Inter-asset correlation (Cholesky) | ❌ | ✅ |
| **Position Model** | Integer shares (equities) | ✅ | ✅ |
| | Fractional positions (crypto) | ❌ | ✅ |
| **Price Dynamics** | Geometric Brownian Motion | ✅ | ✅ |
| | Student-t fat tails | ✅ | ✅ |
| | AR(1) autocorrelation | ✅ | ✅ |
| | GARCH(1,1) conditional volatility | ✅ | ✅ |
| | Markov regime switching (2 regimes) | ✅ | ✅ |
| | Markov regime switching (N regimes) | ❌ | ✅ |
| | Merton jump diffusion | ❌ | ✅ |
| **Fee Model** | Symmetric (buy + sell) | ✅ | ❌ |
| | Asymmetric (sell-only) | ❌ | ✅ |
| **Risk Controls** | Bankruptcy threshold (episode truncation) | ✅ | ✅ |
| | Per-asset trailing stop penalty | ❌ | ✅ |
| | Hard trailing stop (forced liquidation) | ❌ | ✅ |
| | Price floor / delisting detection | ❌ | ✅ |
| **Terminal Handling** | Mark-to-market (no forced sell) | ✅ | ❌ |
| | Forced terminal liquidation with sell fees | ❌ | ✅ |
| **Observation Space** | Normalised price lookback window | ✅ | ✅ |
| | Cash / shares / portfolio value | ✅ | ✅ |
| | Unrealised exit cost | ❌ | ✅ |
| | Time remaining in episode | ❌ | ✅ |
| **Reward Signal** | Absolute PnL (vs. starting cash) | ✅ | ❌ |
| | Alpha-relative PnL (vs. buy-and-hold benchmark) | ❌ | ✅ |
| **Rendering** | Built-in matplotlib (single-pane price + portfolio) | ✅ | ❌ |
| | Decoupled SpectralRenderWrapper (4-pane, pickle-safe) | ❌ | ✅ |
| **Parallelism** | SubprocVecEnv / DummyVecEnv compatible | ❌ (rendering blocks pickle) | ✅ |
| | VecNormalize integration | ❌ | ✅ |
| **Tooling** | Parameter estimation from real data (est_env_params.py) | ❌ | ✅ |
| | Full training script (train.py) | ❌ | ✅ |
| | Diagnostic dashboard (examine.ipynb) | ❌ | ✅ |
| | Live/paper trading bridge (alpaca_rl_trader.py) | ❌ | ✅ |
| **Training Support** | Basic example notebook | ✅ (simple SB3 loop) | ✅ |
| | SpectralExtractor (custom feature extractor) | ❌ | ✅ |
| | DiagnosticsCallback | ❌ | ✅ |
| | EntropyCoefficientCallback | ❌ | ✅ |

---

## Lite Engine Design (`spectral_env_lite/engine.py`)

### Constructor Signature
```python
class SpectralTradingEnv(gym.Env):
    def __init__(
        self,
        num_steps: int = 252,
        time_total: float = 1.0,
        initial_price: float = 100.0,
        volatility: float = 0.2,
        drift: float = 0.1,
        transaction_cost_pct: float = 0.001,
        starting_cash: float = 10_000.0,
        max_shares: int = 100,
        max_trade_size: int = 10,
        lookback_window: int = 30,
        phi: float = 0.0,
        df: int = 15,
        # GARCH
        garch_alpha: float = 0.05,
        garch_beta: float = 0.90,
        # Regime switching (2 regimes only)
        regime_drift_mults: tuple = (1.5, -0.5),
        regime_vol_mults: tuple = (0.7, 1.8),
        regime_switch_prob: float = 0.05,
        # Risk
        bankruptcy_threshold: float = 0.1,
        # AR(1)
        randomize_phi: bool = True,
        render_mode: str = None,  # 'human' or None
    ):
```

### Key Simplifications
1. **Single asset only** — no `num_assets` param, no per-asset arrays, no Cholesky
2. **No jump diffusion** — `_generate_brownian_path()` skips Merton component entirely
3. **Fixed 2 regimes** — `n_regimes` is not exposed; always 2
4. **Symmetric fees** — both buy and sell incur `transaction_cost_pct`
5. **No trailing stops** — no `trailing_stop_pct`, `hard_trailing_stop`, or `price_floor_pct`
6. **No terminal liquidation** — episode ends at step `num_steps`, final portfolio is mark-to-market with unrealised gains (no exit fee applied)
7. **Absolute reward** — `reward = pnl / starting_cash * 100` (no benchmark subtraction)
8. **Simple observation** — `[price_window(30), norm_cash, norm_shares, norm_portfolio]` → obs_dim = 33
9. **Rendering baked in** — `self.fig`, `self.ax` live on the class; not pickle-safe; simple 2-pane chart (price + portfolio return)
10. **Integer shares only** — `self.shares` is `int`, actions are rounded

### What Makes Lite Still Compelling
- Student-t fat tails → realistic tail events
- GARCH(1,1) → volatility clustering
- 2-regime Markov switching → bull/bear transitions
- AR(1) with randomised phi → momentum + mean-reversion training diversity
- Fully Gymnasium-compatible → drop-in with SB3/RLlib
- Infinite episodes (stochastic generation, never repeats)
- All core parameters exposed and tunable

---

## Upgrade Friction Points (What Lite Users Will Want)

These are deliberately excluded features that create natural demand for the pro version:

### 1. Multi-Asset Portfolio Training
**What they'll experience:** After training a single-asset agent, users will immediately want to test portfolio allocation across correlated assets. Lite forces `num_assets=1` with no escape.

**Pro unlock:** `num_assets=N`, correlation matrix, Cholesky-correlated paths, per-asset position management.

### 2. Parallel Training Speed
**What they'll experience:** Training at ~1,600 fps on a single env is painfully slow. 252-step episodes × 7M timesteps = hours of wall-clock time on a single core.

**Pro unlock:** Decoupled rendering → pickle-safe → `SubprocVecEnv` with 8+ workers → 5-8× speedup.

### 3. Realistic Fee Model
**What they'll experience:** Symmetric fees punish every action equally, making it hard to learn meaningful entry/exit timing. The agent learns "trade less" rather than "trade smarter."

**Pro unlock:** Sell-only fees + forced terminal liquidation = realistic broker economics that reward strategic position management.

### 4. Parameter Calibration to Real Markets
**What they'll experience:** Users will manually tune drift/vol/GARCH from their own data or guesswork. Tedious and error-prone.

**Pro unlock:** `est_env_params.py` — one command fits all parameters from real tickers via yfinance.

### 5. Jump Diffusion (Tail Events)
**What they'll experience:** Student-t gives moderate fat tails, but real markets have discrete crash events (flash crashes, earnings surprises). Lite agents are brittle to these.

**Pro unlock:** Merton jump diffusion produces realistic discontinuous price events for robustness training.

### 6. Training Diagnostics
**What they'll experience:** After training, users can only see final reward. No visibility into whether the agent is churning, over-leveraging, or exploiting a regime.

**Pro unlock:** Full diagnostic dashboard with friction tracking, action analysis, explained variance monitoring, regime exposure.

### 7. Crypto / Fractional Trading
**What they'll experience:** Integer shares only. Can't simulate $100 of BTC.

**Pro unlock:** `fractional=True` enables notional-based position sizing for crypto and fractional share platforms.

---

## API Compatibility Contract

Both lite and pro must satisfy this interface so user code works with either:

```python
import gymnasium as gym

# Works with either package installed (not both simultaneously)
env = gym.make('SpectralEnv-v1', **kwargs)
obs, info = env.reset(seed=42)

for _ in range(env.num_steps):
    action = env.action_space.sample()  # shape: (1,) for lite, (N,) for pro
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Shared Interface Guarantees
- `env.action_space` → `Box([-1, 1], shape=(num_assets,))`
- `env.observation_space` → `Box(shape=(obs_dim,))`
- `env.reset(seed=...)` → `(obs, info)`
- `env.step(action)` → `(obs, reward, terminated, truncated, info)`
- `info` dict always contains: `prices`, `shares`, `cash`, `step`, `regime`, `friction_leak`
- `env.render()` works in both (human mode)
- `env.close()` cleans up resources

### Breaking Difference (Documented)
- Lite `action_space.shape = (1,)` always; pro can be `(N,)` for N > 1
- Lite `obs_dim = lookback_window + 3`; pro `obs_dim = num_assets * lookback_window + num_assets + 4`
- Pro requires `SpectralRenderWrapper` for rendering; lite renders directly

---

## Lite Example Notebook (`examples/train_basic.ipynb`)

Minimal contents:
1. Install and import
2. Configure env with sensible defaults
3. Train PPO agent (single env, ~500K steps, no VecEnv)
4. Plot training reward curve
5. Run one eval episode with `render_mode='human'`
6. Print final info dict

No custom extractor, no callbacks, no diagnostics. Just enough to prove the environment works and the agent learns. Ends with a "Want more?" section pointing to the pro version's features.

---

## README Strategy for Lite Repo

The lite README should:
1. Lead with the value proposition: "Infinite realistic market data for RL research. Free."
2. Show a 5-line quick start that produces a working agent
3. Display a rendered episode (screenshot/gif)
4. List features present (emphasise what's there, not what's missing)
5. Mention limitations briefly and tastefully: "Single-asset training environment. For multi-asset portfolios, fractional positions, jump diffusion, parallel training, and calibration tools → see Spectral-Env-Core Pro"
6. Link to the pro repo / purchase page

---

## Implementation Priority

1. **Design document** ← this file
2. **Lite engine** — simplified single-file `engine.py` derived from `engine2.py`
3. **Lite `__init__.py`** — registration + export
4. **Lite `pyproject.toml`** — minimal deps (gymnasium, numpy, matplotlib)
5. **Lite example notebook** — basic training loop
6. **Lite README** — marketing-oriented
7. **Validate** — ensure `gym.make('SpectralEnv-v1')` works identically in both packages
