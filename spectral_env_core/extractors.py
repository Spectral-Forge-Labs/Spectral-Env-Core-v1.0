"""
spectral_env_core.extractors
=============================
Custom feature extractors designed for Spectral-Env-Core's observation space.

Usage
-----
    from spectral_env_core import SpectralExtractor

    policy_kwargs = dict(
        features_extractor_class=SpectralExtractor,
        features_extractor_kwargs=dict(
            num_assets=5,
            lookback_window=30,
            asset_embed_dim=32,
            meta_embed_dim=16,
        ),
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
"""

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpectralExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-asset trading observations.

    Designed specifically for SpectralTradingEnv's flat observation layout:
        [price_window_asset_0, ..., price_window_asset_N, meta_features]

    Architecture
    ------------
    price_net : shared 2-layer MLP applied identically to every asset's
                normalised price window. Weight sharing means the network
                learns generic temporal patterns (momentum, reversion,
                volatility clustering) applicable across all assets.
                Windows are already normalised as % deviation from the
                asset's current price, making weight sharing statistically
                sound across assets with different price levels.

    meta_net  : single linear projection for portfolio metadata
                (norm_cash, norm_shares × N, norm_portfolio_value,
                unrealised_exit_cost, time_remaining).
                Projects raw scalars into the same activation scale as
                the price embeddings before concatenation, preventing
                gradient scale mismatch.

    Output dim (auto-computed from architecture):
        (num_assets × asset_embed_dim) + meta_embed_dim

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box
    num_assets        : number of assets in the environment
    lookback_window   : number of price steps in each asset's history window
    asset_embed_dim   : output dimension of the shared price encoder per asset
    meta_embed_dim    : output dimension of the metadata projection
    """

    def __init__(
        self,
        observation_space,
        num_assets: int = 1,
        lookback_window: int = 30,
        asset_embed_dim: int = 32,
        meta_embed_dim: int = 16,
    ):
        self.num_assets         = num_assets
        self.lookback_window    = lookback_window
        self.price_features_dim = num_assets * lookback_window
        self.asset_embed_dim    = asset_embed_dim

        # Meta features: cash(1) + shares(N) + portfolio(1) + exit_cost(1) + time_remaining(1)
        base_meta_dim = 1 + num_assets + 1 + 1 + 1

        # Any remaining features are from indicators
        obs_total = observation_space.shape[0]
        self.meta_dim = obs_total - self.price_features_dim  # includes base meta + indicators
        self.meta_embed_dim = meta_embed_dim

        # Output dim derived from architecture
        total_output_dim = (num_assets * asset_embed_dim) + meta_embed_dim
        super().__init__(observation_space, features_dim=total_output_dim)

        # Shared price encoder — 2-layer MLP captures non-linear temporal
        # structure (momentum, mean-reversion, GARCH clusters) that a single
        # linear layer cannot represent.
        self.price_net = nn.Sequential(
            nn.Linear(lookback_window, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, asset_embed_dim),
            nn.LayerNorm(asset_embed_dim),
            nn.ReLU(),
        )

        # Metadata + indicator projection — aligns gradient scale with price embeddings
        self.meta_net = nn.Sequential(
            nn.Linear(self.meta_dim, meta_embed_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]

        # Split into price windows and portfolio metadata
        price_flat = observations[:, :self.price_features_dim]
        meta_raw   = observations[:, self.price_features_dim:]

        # Process all assets in one batched forward pass
        price_reshaped = price_flat.reshape(batch_size * self.num_assets, self.lookback_window)
        price_encoded  = self.price_net(price_reshaped)
        price_features = price_encoded.reshape(batch_size, self.num_assets * self.asset_embed_dim)

        # Project metadata
        meta_features = self.meta_net(meta_raw)

        # Concatenate — both branches share the same activation scale
        return th.cat([price_features, meta_features], dim=1)
