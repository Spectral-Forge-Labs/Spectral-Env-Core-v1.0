# Spectral-Env-Core

**High-fidelity stochastic market simulation for reinforcement learning research.**

Spectral-Env-Core generates unlimited, statistically accurate financial data for training RL trading agents. Every episode is unique — calibrated from real market dynamics but never repeating. Train agents that generalise to live markets, not agents that memorise historical price sequences.

Developed by [Spectral Forge Labs](https://github.com/Spectral-Forge-Labs).

---

## Why This Exists

Fixed historical datasets produce brittle agents. 20 years of daily data gives you ~5,000 samples — your agent memorises the sequence, overfits to a single regime, and collapses the moment markets shift.

Spectral-Env-Core solves this with a calibrated stochastic process that combines six layers of realistic market structure:

| Layer | What It Models | Why It Matters for RL |
|---|---|---|
| **Student-t fat tails** | Extreme events at realistic frequency | Agents learn risk management |
| **GARCH(1,1)** | Volatility clustering (calm/turbulent persistence) | Position sizing adapts to vol regime |
| **AR(1) autocorrelation** | Momentum and mean-reversion | Timing strategies become learnable |
| **N-regime Markov switching** | Bull/bear/sideways state transitions | Regime detection and adaptation |
| **Merton jump diffusion** | Flash crashes, earnings gaps | Robustness to discontinuous events |
| **Cholesky correlation** | Realistic multi-asset co-movement | Portfolio allocation and hedging |

---

## Quick Start

```bash
pip install -e .
pip install stable-baselines3 yfinance arch  # optional extras
```

```python
from spectral_env_core import SpectralTradingEnv, estimate_params

# Calibrate to a real ticker
params = estimate_params("NVDA", period="3y")
env = SpectralTradingEnv(**params, num_steps=252, starting_cash=50_000, max_shares=200, max_trade_size=20)

# Train
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

---

## Features

### Multi-Asset Portfolios
```python
from spectral_env_core import estimate_params_multi

params = estimate_params_multi(["AAPL", "MSFT", "NVDA", "JPM", "SPY"], period="3y")
env = SpectralTradingEnv(**params)
# Agent controls 5 continuous actions simultaneously — portfolio allocation
```

### Fractional Positions (Crypto)
```python
env = SpectralTradingEnv(
    initial_price=[67_500.0, 3_200.0],  # BTC, ETH
    max_shares=25_000,                   # max $25K position value per asset
    max_trade_size=5_000,                # max $5K notional per trade
    fractional=True,                     # buy 0.00312 BTC
)
```

### Production Training Pipeline
```python
from spectral_env_core import (
    SpectralTradingEnv,
    SpectralExtractor,          # shared-encoder multi-asset architecture
    DiagnosticsCallback,        # friction, action stats, explained variance
    EntropyCoefficientCallback, # controlled exploration decay
    estimate_params_multi,
)
```

### Parallel Training
The environment is pickle-safe — no matplotlib state in the core class. Works with `SubprocVecEnv` out of the box for 5-8× speedup.

```bash
python train.py  # 8 parallel envs, full diagnostic pipeline
```

---

## Package API

```python
from spectral_env_core import (
    # Environment
    SpectralTradingEnv,          # core RL environment
    SpectralRenderWrapper,       # matplotlib rendering (eval only)

    # Feature extraction
    SpectralExtractor,           # shared-encoder for multi-asset obs

    # Training callbacks
    DiagnosticsCallback,         # granular metrics + VecNormalize sync
    EntropyCoefficientCallback,  # linear entropy decay

    # Parameter calibration
    estimate_params,             # single ticker → env_kwargs
    estimate_params_multi,       # N tickers → env_kwargs with correlation
)
```

---

## Tutorials

| # | Topic | What You'll Learn |
|---|---|---|
| 01 | [Infinite Data Generation](tutorials/01_infinite_data.ipynb) | Why stochastic > historical for RL |
| 02 | [Market Realism](tutorials/02_market_realism.ipynb) | How each stochastic layer creates exploitable structure |
| 03 | [Multi-Asset Portfolios](tutorials/03_multi_asset_portfolios.ipynb) | Correlated assets and portfolio allocation |
| 04 | [Crypto & Fractional](tutorials/04_crypto_fractional.ipynb) | Dollar-based position sizing for BTC/ETH |
| 05 | [Parameter Calibration](tutorials/05_parameter_calibration.ipynb) | Ticker → calibrated env in one line |
| 06 | [Parallel Training](tutorials/06_parallel_training.ipynb) | SubprocVecEnv speedup + gradient quality |
| 07 | [Training Toolkit](tutorials/07_training_toolkit.ipynb) | SpectralExtractor + DiagnosticsCallback deep dive |

---

## Environment Design Highlights

- **Sell-only transaction fees** — buys are free, sells incur cost. Matches real broker economics and creates a clean incentive for strategic position management.
- **Forced terminal liquidation** — all positions are sold at episode end with fees applied. Prevents the "hold forever" reward hack.
- **Alpha-relative reward** — measured against a passive equal-weight buy-and-hold benchmark. The agent is rewarded for *outperforming*, not just riding the market up.
- **Unrealised exit cost + time remaining in observation** — the agent can see its deferred liability and episode clock, enabling the value function to price the terminal liquidation cliff.
- **Randomised AR(1) per episode** — each episode has a different momentum/reversion character, forcing the agent to generalise rather than memorise a single regime.

---

## Project Structure

```
spectral_env_core/
├── __init__.py           # Package exports + Gymnasium registration
├── engine2.py            # Core environment (pickle-safe, multi-asset, fractional)
├── extractors.py         # SpectralExtractor (feature extractor for SB3)
├── callbacks.py          # DiagnosticsCallback + EntropyCoefficientCallback
├── wrappers.py           # SpectralRenderWrapper (4-pane matplotlib dashboard)
├── est_env_params.py     # Parameter calibration from real market data
└── history/
    └── engine.py         # Legacy (do not use)

train.py                  # Reference parallel training script
tutorials/                # 7 Jupyter notebooks
```

---

## Requirements

- Python 3.9+
- Core: `gymnasium`, `numpy`, `scipy`
- Training: `stable-baselines3`, `matplotlib`
- Calibration: `yfinance`, `arch` (optional, improves GARCH fitting)

---

## License

MIT — use freely for research, education, or commercial applications.

---

*Disclaimer: Spectral Forge Labs does not provide financial advice. This is a simulation tool for data science research only.*
