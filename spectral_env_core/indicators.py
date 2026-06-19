"""
spectral_env_core.indicators
==============================
Plug-in technical indicators for the SpectralTradingEnv observation space.

Indicators are passed to the environment at construction time and automatically
expand the observation space. Each indicator computes normalised values from
the pre-generated price path at every step.

Usage
-----
    from spectral_env_core import SpectralTradingEnv
    from spectral_env_core.indicators import RSI, MACD, BollingerBands

    env = SpectralTradingEnv(
        ...,
        indicators=[
            RSI(period=14),
            MACD(fast=12, slow=26, signal=9),
            BollingerBands(period=20, std_dev=2),
        ],
    )
    # Observation space automatically includes indicator values

Writing Custom Indicators
--------------------------
    from spectral_env_core.indicators import Indicator

    class MyIndicator(Indicator):
        @property
        def name(self) -> str:
            return "my_indicator"

        @property
        def n_features(self) -> int:
            return 1  # how many values per asset (or total if per_asset=False)

        def compute(self, prices: np.ndarray, step: int) -> np.ndarray:
            # prices shape: (num_steps+1, num_assets)
            # Return flat array of normalised values
            ...
"""

from abc import ABC, abstractmethod
import numpy as np


class Indicator(ABC):
    """
    Base class for observation space indicator plugins.

    Subclasses must implement:
      - name: unique identifier string
      - n_features: number of float values produced per call
      - compute(prices, step): returns normalised indicator values

    Properties:
      - per_asset: if True (default), n_features is multiplied by num_assets
                   in the total observation dim. If False, n_features is the
                   total regardless of num_assets.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this indicator (used in logging/debugging)."""
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of values produced per asset (or total if per_asset=False)."""
        ...

    @property
    def per_asset(self) -> bool:
        """If True, compute() returns n_features values per asset."""
        return True

    @abstractmethod
    def compute(self, prices: np.ndarray, step: int) -> np.ndarray:
        """
        Compute indicator value(s) at the given step.

        Parameters
        ----------
        prices : np.ndarray, shape (num_steps+1, num_assets)
            Full pre-generated price path for the episode.
        step : int
            Current step index (0 to num_steps).

        Returns
        -------
        np.ndarray, dtype float32
            Flat array of normalised indicator values.
            Length = n_features * num_assets (if per_asset=True)
                   or n_features (if per_asset=False).
            Values should be roughly in [-1, 1] or [0, 1] range for
            stable network training.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_features={self.n_features})"


# ---------------------------------------------------------------------------
# Built-in indicators
# ---------------------------------------------------------------------------

class RSI(Indicator):
    """
    Relative Strength Index.

    Normalised to [-1, 1] range: (RSI - 50) / 50.
    At value 0.0 the market is neutral; +1.0 is extremely overbought;
    -1.0 is extremely oversold.

    Parameters
    ----------
    period : int
        Lookback period for RSI computation. Default 14.
    """

    def __init__(self, period: int = 14):
        self.period = period

    @property
    def name(self) -> str:
        return f"rsi_{self.period}"

    @property
    def n_features(self) -> int:
        return 1

    def compute(self, prices: np.ndarray, step: int) -> np.ndarray:
        num_assets = prices.shape[1]
        result = np.zeros(num_assets, dtype=np.float32)

        for i in range(num_assets):
            start = max(0, step - self.period)
            window = prices[start:step + 1, i]

            if len(window) < 2:
                continue

            deltas = np.diff(window)
            gains = np.maximum(deltas, 0.0)
            losses = np.maximum(-deltas, 0.0)

            avg_gain = gains.mean() if len(gains) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0

            if avg_loss == 0.0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

            # Normalise to [-1, 1]
            result[i] = (rsi - 50.0) / 50.0

        return result


class MACD(Indicator):
    """
    Moving Average Convergence Divergence.

    Returns 2 features per asset:
      - MACD line (normalised by price)
      - Signal line (normalised by price)

    Parameters
    ----------
    fast : int
        Fast EMA period. Default 12.
    slow : int
        Slow EMA period. Default 26.
    signal : int
        Signal EMA period. Default 9.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"macd_{self.fast}_{self.slow}_{self.signal}"

    @property
    def n_features(self) -> int:
        return 2  # MACD line + signal line

    def compute(self, prices: np.ndarray, step: int) -> np.ndarray:
        num_assets = prices.shape[1]
        result = np.zeros(num_assets * 2, dtype=np.float32)

        for i in range(num_assets):
            series = prices[:step + 1, i]
            current_price = series[-1] if len(series) > 0 else 1.0

            if len(series) < self.slow + self.signal:
                continue

            # EMA computation
            ema_fast = self._ema(series, self.fast)
            ema_slow = self._ema(series, self.slow)
            macd_line = ema_fast - ema_slow

            # Signal line (EMA of MACD)
            if len(macd_line) >= self.signal:
                signal_line = self._ema(macd_line, self.signal)
                macd_val   = macd_line[-1]
                signal_val = signal_line[-1]
            else:
                macd_val   = 0.0
                signal_val = 0.0

            # Normalise by current price (makes it scale-invariant)
            result[i * 2]     = macd_val / current_price * 100
            result[i * 2 + 1] = signal_val / current_price * 100

        return result

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        alpha = 2.0 / (period + 1)
        ema = np.empty(len(data))
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
        return ema


class BollingerBands(Indicator):
    """
    Bollinger Bands — returns the normalised position of price within the bands.

    Returns 1 feature per asset: (price - middle) / (upper - middle).
    Value of +1.0 means price is at the upper band; -1.0 at the lower band;
    0.0 at the middle (SMA).

    Parameters
    ----------
    period : int
        SMA/std lookback period. Default 20.
    std_dev : float
        Number of standard deviations for band width. Default 2.0.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period  = period
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        return f"bbands_{self.period}_{self.std_dev}"

    @property
    def n_features(self) -> int:
        return 1

    def compute(self, prices: np.ndarray, step: int) -> np.ndarray:
        num_assets = prices.shape[1]
        result = np.zeros(num_assets, dtype=np.float32)

        for i in range(num_assets):
            start = max(0, step - self.period + 1)
            window = prices[start:step + 1, i]

            if len(window) < 2:
                continue

            sma = window.mean()
            std = window.std(ddof=0)

            if std < 1e-10:
                result[i] = 0.0
            else:
                band_width = self.std_dev * std
                current = prices[step, i]
                result[i] = np.clip((current - sma) / band_width, -2.0, 2.0)

        return result


class ATR(Indicator):
    """
    Average True Range — measures volatility.

    Returns 1 feature per asset: ATR normalised by current price.
    Higher values = more volatile conditions.

    Parameters
    ----------
    period : int
        ATR lookback period. Default 14.
    """

    def __init__(self, period: int = 14):
        self.period = period

    @property
    def name(self) -> str:
        return f"atr_{self.period}"

    @property
    def n_features(self) -> int:
        return 1

    def compute(self, prices: np.ndarray, step: int) -> np.ndarray:
        num_assets = prices.shape[1]
        result = np.zeros(num_assets, dtype=np.float32)

        for i in range(num_assets):
            start = max(0, step - self.period)
            window = prices[start:step + 1, i]

            if len(window) < 2:
                continue

            # True range approximation (no high/low available, use close-to-close)
            true_ranges = np.abs(np.diff(window))
            atr = true_ranges.mean()
            current = prices[step, i]

            # Normalise by current price (percentage ATR)
            result[i] = (atr / current) * 10  # scale so typical values are ~0.1-1.0

        return result


class SMA_Crossover(Indicator):
    """
    Simple Moving Average crossover signal.

    Returns 1 feature per asset: (fast_SMA - slow_SMA) / current_price.
    Positive = bullish crossover; negative = bearish crossover.

    Parameters
    ----------
    fast : int
        Fast SMA period. Default 10.
    slow : int
        Slow SMA period. Default 30.
    """

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"sma_cross_{self.fast}_{self.slow}"

    @property
    def n_features(self) -> int:
        return 1

    def compute(self, prices: np.ndarray, step: int) -> np.ndarray:
        num_assets = prices.shape[1]
        result = np.zeros(num_assets, dtype=np.float32)

        for i in range(num_assets):
            series = prices[:step + 1, i]
            current = series[-1] if len(series) > 0 else 1.0

            if len(series) < self.slow:
                continue

            fast_sma = series[-self.fast:].mean()
            slow_sma = series[-self.slow:].mean()

            # Normalise by price — gives a percentage spread
            result[i] = (fast_sma - slow_sma) / current * 100

        return result
