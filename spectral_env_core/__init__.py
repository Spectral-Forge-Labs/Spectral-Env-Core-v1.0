from .engine2 import SpectralTradingEnv
from .wrappers import SpectralRenderWrapper
from .extractors import SpectralExtractor
from .callbacks import DiagnosticsCallback, EntropyCoefficientCallback
from .indicators import Indicator, RSI, MACD, BollingerBands, ATR, SMA_Crossover
from .backtest import backtest, backtest_historical, BacktestReport
from .journal import TradeJournal
from .est_env_params import estimate_single as estimate_params
from .est_env_params import estimate_multi as estimate_params_multi

from gymnasium.envs.registration import register

register(
    id='SpectralEnv-v1',
    entry_point='spectral_env_core.engine2:SpectralTradingEnv',
)
