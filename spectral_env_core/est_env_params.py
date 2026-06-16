import sys
import json
import warnings
import argparse
import numpy as np
from scipy import stats


warnings.filterwarnings("ignore")

TRADING_DAYS = 252

#--------------------------------------------
# Data Fetching
#--------------------------------------------

def fetch_prices(ticker: str, period: str = "3y") -> np.ndarray:
    """Download adjusted close prices."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}. Check the ticker symbol.")
    prices = df['Close'].dropna().values.flatten()
    return prices.astype(float)

def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


# -------------------------------------------
# Individual Estimators
#--------------------------------------------

def estimate_drift_vol(returns: np.ndarray) -> tuple:
    daily_mean = float(returns.mean())
    daily_std  = float(returns.std(ddof=1))
    vol        = daily_std * np.sqrt(TRADING_DAYS)
    drift      = daily_mean * TRADING_DAYS + 0.5 * vol ** 2
    return round(drift, 4), round(vol, 4)

def estimate_phi(returns: np.ndarray) -> float:
    y = returns[1:]
    x = returns[:-1]
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    phi = float(np.dot(x, y) / denom)
    return round(float(np.clip(phi, -0.95, 0.95)), 4)

def ar1_residuals(returns: np.ndarray, phi: float) -> np.ndarray:
    return returns[1:] - phi * returns[:-1]

def estimate_df(residuals: np.ndarray) -> int:  # Fix: was "estimage_df"
    std = float(residuals.std(ddof=1))
    normed = residuals / std if std > 0 else residuals
    try:
        df_fit, _, _ = stats.t.fit(normed, floc=0, fscale=1)
        return int(np.clip(round(df_fit), 3, 30))
    except Exception:
        return 15

def estimate_garch(returns: np.ndarray) -> tuple:
    try:
        from arch import arch_model
        am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        alpha = float(res.params.get("alpha[1]", 0.05))
        beta  = float(res.params.get("beta[1]", 0.90))
        # Enforce stationarity: alpha + beta < 1
        if alpha + beta >= 1.0:
            scale = 0.95 / (alpha + beta)
            alpha *= scale
            beta  *= scale
        return round(alpha, 4), round(beta, 4)
    except ImportError:
        print(" [warn] 'arch' package not found - using default GARCH params.")
        print("  Install with: pip install arch")
        return 0.05, 0.90
    except Exception as e:
        print(f" [warn] GARCH fitting failed ({e}) - using defaults.")
        return 0.05, 0.90

def estimate_jumps(returns: np.ndarray, threshold_sigma: float = 3.5) -> tuple:
    sigma  = float(returns.std(ddof=1))
    mask   = np.abs(returns) > threshold_sigma * sigma
    n_jumps = int(mask.sum())
    n_years = len(returns) / TRADING_DAYS

    if n_jumps < 3:
        # Too few events - treat as jump-free
        return 0.0, -0.03, 0.05

    jump_rets = returns[mask]
    intensity = round(n_jumps / n_years, 2)
    mean      = round(float(jump_rets.mean()), 4)
    std       = round(float(jump_rets.std(ddof=1)) if n_jumps > 1 else 0.05, 4)
    return intensity, mean, std


#--------------------------------------------
# Per-Ticker Estimator
#--------------------------------------------

def estimate_single(ticker: str, period: str = "3y", verbose: bool = True) -> dict:
    if verbose:
        print(f"\n{'='*54}")
        print(f" Estimating parameters for: {ticker.upper()}")
        print(f"{'='*54}")

    prices  = fetch_prices(ticker, period)
    returns = log_returns(prices)

    if verbose:
        years = len(returns) / TRADING_DAYS
        print(f" Prices: {len(prices):,} | Returns: {len(returns):,} | ~{years:.1f} yrs")

    drift, vol       = estimate_drift_vol(returns)
    phi              = estimate_phi(returns)
    residuals        = ar1_residuals(returns, phi)
    df               = estimate_df(residuals)          # Fix: corrected function name
    garch_a, garch_b = estimate_garch(returns)
    j_lam, j_mu, j_sig = estimate_jumps(returns)

    initial_price = round(float(prices[-1]), 2)

    env_kwargs = {
        "initial_price":  initial_price,
        "drift":          drift,
        "volatility":     vol,          # Fix: was "volattility"
        "phi":            phi,
        "randomize_phi":  False,
        "df":             df,
        "garch_alpha":    garch_a,
        "garch_beta":     garch_b,
        "jump_intensity": j_lam,
        "jump_mean":      j_mu,
        "jump_std":       j_sig,
    }

    return env_kwargs


#--------------------------------------------
# Multi-Ticker Estimator
#--------------------------------------------

def estimate_multi(tickers: list, period: str = "3y") -> dict:
    all_returns = {}
    per_asset   = {}

    for t in tickers:
        prices         = fetch_prices(t, period)
        r              = log_returns(prices)
        all_returns[t] = r
        per_asset[t]   = estimate_single(t, period, verbose=True)

    min_len = min(len(v) for v in all_returns.values())
    matrix  = np.column_stack([all_returns[t][-min_len:] for t in tickers])
    corr    = np.corrcoef(matrix.T)

    # Helpers
    def per_asset_list(key):
        return [per_asset[t][key] for t in tickers]

    def scalar_avg(key):
        vals = [per_asset[t][key] for t in tickers]
        return round(float(sum(vals) / len(vals)), 4)

    # Build env_kwargs
    env_kwargs = {
        "num_assets":     len(tickers),
        "initial_price":  per_asset_list("initial_price"),  # Fix: was "initial price"
        "drift":          per_asset_list("drift"),
        "volatility":     per_asset_list("volatility"),     # Fix: was "vol" / "volattility"
        "phi":            0.0,
        "randomize_phi":  True,
        "df":             int(round(scalar_avg("df"))),
        "garch_alpha":    scalar_avg("garch_alpha"),        # Fix: was "garch_a"
        "garch_beta":     scalar_avg("garch_beta"),         # Fix: was "garch_b"
        "jump_intensity": scalar_avg("jump_intensity"),
        "jump_mean":      scalar_avg("jump_mean"),
        "jump_std":       scalar_avg("jump_std"),           # Fix: was "jumpt_std"
        "correlation":    corr,
    }

    return env_kwargs


#--------------------------------------------
# Pretty-Print Copy/Paste Usage Block
#--------------------------------------------

def _serialize(v):                          # Fix: was "_seralize"
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, list):
        return [_serialize(x) for x in v]
    return v

def print_usage_block(env_kwargs: dict, tickers: list) -> None:
    cleaned  = {k: _serialize(v) for k, v in env_kwargs.items()}
    corr_list = cleaned.pop("correlation", None)

    print(f"\n # Parameters fitted from: {',  '.join(t.upper() for t in tickers)}")
    print("env_kwargs = " + json.dumps(cleaned, indent=4))

    if corr_list is not None:
        n    = len(corr_list)
        rows = ",\n     ".join(
            "[" + ", ".join(f"{corr_list[i][j]:+.4f}" for j in range(n)) + "]"
            for i in range(n)
        )
        print(f'\nenv_kwargs["correlation"] = np.array([\n      {rows}\n])')


#--------------------------------------------
# CLI Entry Point
#--------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Spectral Env parameters from historical market data."
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="One or more ticker symbols (e.g. NVDA, AAPL, MSFT)",
    )
    parser.add_argument(                    # Fix: --period was missing entirely
        "--period",
        default="3y",
        help="Lookback period passed to yfinance (e.g. 1y, 3y, 5y). Default: 3y",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

    if len(tickers) == 1:
        env_kwargs = estimate_single(tickers[0], period=args.period)
    else:
        env_kwargs = estimate_multi(tickers, period=args.period)

    print_usage_block(env_kwargs, tickers)


if __name__ == "__main__":      # Fix: was "if __main__ == "__main__":"
    main()