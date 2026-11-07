"""Macro regime conditioning for the Spectral RL environment.

Self-contained module that generates per-episode drift/vol modifiers from a
named Federal Reserve stress scenario. Unlike the batch-oriented SpectralForge
generator, this works one episode at a time — suited to RL's reset() loop.

Pipeline:
    1. Build a deterministic quarterly skeleton for equity_price_index and vix.
    2. Add seeded stochastic noise for episode diversity.
    3. Interpolate to daily resolution (num_steps).
    4. Derive drift_modifier (additive) and vol_modifier (multiplicative).

Regime targets are calibrated to published Fed DFAST/CCAR scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Regime definitions (equity + vix only — the two variables that drive the
# price engine modifiers). Values calibrated to Fed stress scenarios.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeSpec:
    """Equity + VIX targets for a named scenario.

    Attributes:
        equity_trough: cumulative % change in equity index at trough (e.g. -0.54).
        equity_q: quarter index of equity trough.
        equity_frontload: fraction of equity move in Q1 (None = linear).
        vix_jump_off: starting VIX level (decimal, e.g. 0.20).
        vix_peak: peak VIX level (decimal).
        vix_q: quarter index of VIX peak.
    """

    equity_trough: float
    equity_q: int
    equity_frontload: float | None
    vix_jump_off: float
    vix_peak: float
    vix_q: int


# Calibrated to Fed published scenarios (equity + VIX components).
REGIME_SPECS: dict[str, RegimeSpec] = {
    "severely_adverse_2020": RegimeSpec(-0.50, 4, 0.60, 0.14, 0.70, 2),
    "severely_adverse_2021": RegimeSpec(-0.55, 4, 0.60, 0.31, 0.70, 2),
    "severely_adverse_2022": RegimeSpec(-0.55, 4, 0.60, 0.215, 0.70, 2),
    "severely_adverse_2023": RegimeSpec(-0.45, 4, 0.65, 0.336, 0.75, 2),
    "severely_adverse_2024": RegimeSpec(-0.54, 3, 0.60, 0.20, 0.72, 2),
    "severely_adverse_2025": RegimeSpec(-0.50, 4, 0.60, 0.276, 0.65, 2),
    "severely_adverse_2026": RegimeSpec(-0.54, 3, 0.67, 0.232, 0.72, 2),
    "baseline_2024": RegimeSpec(0.0, 13, None, 0.20, 0.22, 7),
    "baseline_2025": RegimeSpec(0.0, 13, None, 0.276, 0.28, 7),
    "baseline_2026": RegimeSpec(0.043, 4, None, 0.232, 0.25, 7),
}

BASELINE_VIX = 0.20
VOL_MODIFIER_MIN = 0.25
VOL_MODIFIER_MAX = 4.0
N_QUARTERS = 13


def available_regimes() -> list[str]:
    """Return the list of regime names available for conditioning."""
    return sorted(REGIME_SPECS)


def _skeleton(jump_off: float, extreme: float, q_extreme: int,
              frontload: float | None, n_quarters: int) -> np.ndarray:
    """Build a deterministic quarterly skeleton path."""
    skel = np.empty(n_quarters)
    skel[0] = jump_off
    total = extreme - jump_off
    q_ext = min(q_extreme, n_quarters - 1)

    if q_ext == 0:
        skel[:] = extreme
        return skel

    if frontload is not None and q_ext >= 2:
        q1 = jump_off + total * frontload
        rem = total - total * frontload
        skel[1] = q1
        for q in range(2, q_ext + 1):
            skel[q] = q1 + rem * ((q - 1) / (q_ext - 1))
    else:
        for q in range(1, q_ext + 1):
            skel[q] = jump_off + total * (q / q_ext)

    for q in range(q_ext + 1, n_quarters):
        skel[q] = extreme
    return skel


def compute_episode_modifiers(
    regime_name: str,
    num_steps: int,
    np_random: np.random.Generator,
    noise_scale: float = 0.15,
    severity_scalar: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate drift and vol modifiers for a single RL episode.

    Parameters
    ----------
    regime_name : str
        Named regime (see available_regimes()).
    num_steps : int
        Number of daily steps in the episode.
    np_random : np.random.Generator
        Episode RNG (from the env) for reproducible per-episode noise.
    noise_scale : float
        Stochastic noise as a fraction of the skeleton move.
    severity_scalar : float
        Scales the regime targets proportionally.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (drift_modifier, vol_modifier), each shape (num_steps,).
        drift_modifier is additive (annualized), vol_modifier multiplicative.
    """
    if regime_name not in REGIME_SPECS:
        raise ValueError(
            f"Unknown regime '{regime_name}'. Available: {available_regimes()}"
        )
    spec = REGIME_SPECS[regime_name]

    # --- Equity skeleton (cumulative % change) ---
    equity_extreme = spec.equity_trough * severity_scalar
    equity_skel = _skeleton(0.0, equity_extreme, spec.equity_q,
                            spec.equity_frontload, N_QUARTERS)
    equity_noise_mag = noise_scale * abs(equity_extreme)
    equity_q = equity_skel + np_random.standard_normal(N_QUARTERS) * equity_noise_mag
    equity_q[0] = 0.0  # anchor jump-off

    # --- VIX skeleton (level) ---
    vix_extreme = spec.vix_jump_off + (spec.vix_peak - spec.vix_jump_off) * severity_scalar
    vix_skel = _skeleton(spec.vix_jump_off, vix_extreme, spec.vix_q, None, N_QUARTERS)
    vix_noise_mag = noise_scale * abs(vix_extreme - spec.vix_jump_off)
    vix_q = vix_skel + np_random.standard_normal(N_QUARTERS) * vix_noise_mag
    vix_q[0] = spec.vix_jump_off
    vix_q = np.clip(vix_q, 0.05, None)  # VIX can't go negative

    # --- Interpolate quarterly → daily ---
    days_per_q = num_steps / N_QUARTERS
    q_days = np.array([min(int(q * days_per_q), num_steps) for q in range(N_QUARTERS)])
    daily_idx = np.arange(num_steps)

    equity_daily = CubicSpline(q_days, equity_q, extrapolate=True)(daily_idx)
    vix_daily    = CubicSpline(q_days, vix_q,    extrapolate=True)(daily_idx)

    # --- Drift modifier (additive, annualized) ---
    # equity is cumulative %; daily log-return × num_steps = annualized drift
    log_levels = np.log(1.0 + np.clip(equity_daily, -0.99, None))
    daily_log_ret = np.diff(log_levels, prepend=log_levels[0])
    drift_modifier = daily_log_ret * num_steps

    # --- Vol modifier (multiplicative, clamped) ---
    vol_modifier = np.clip(vix_daily / BASELINE_VIX, VOL_MODIFIER_MIN, VOL_MODIFIER_MAX)

    return drift_modifier, vol_modifier
