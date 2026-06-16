"""
train.py — Spectral RL Agent Training Script
=============================================
Parallel PPO training using SubprocVecEnv + VecNormalize.

Changes from v1 (based on eval log analysis):
  - VecNormalize stats synced from train → eval env before each evaluation
    via SyncVecNormalizeCallback. Fixes the catastrophic early divergence
    caused by the eval env running with stale (zero-initialised) normalisation
    stats while the training env rapidly updated its own.
  - target_kl raised 0.05 → 0.15 so n_epochs=20 is actually consumed.
    With a noisy value function, high-variance advantages push KL up fast,
    causing early exit after 1-2 passes at the tighter threshold.
  - clip_range raised 0.1 → 0.2 (standard PPO default). 0.1 was
    over-conservative and slowed policy updates unnecessarily.
  - ent_coef linear decay 0.05 → 0.001. Higher entropy early encourages
    exploration past the 0.05 action dead zone; decay lets the policy
    sharpen as it converges.
  - Learning rate unchanged at 3e-5.

Usage
-----
    python train.py

Outputs
-------
    ./models/best_model/best_model.zip   — best checkpoint by eval reward
    ./models/final_model.zip             — model at end of training
    ./logs/vecnormalize.pkl              — VecNormalize stats for inference
    ./logs/                              — TensorBoard logs
"""

import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from spectral_env_core import (
    SpectralTradingEnv,
    SpectralExtractor,
    DiagnosticsCallback,
    EntropyCoefficientCallback,
)


# ---------------------------------------------------------------------------
# Environment parameters — fitted from AAPL, JPM, XOM, JNJ, SPY (3y)
# ---------------------------------------------------------------------------

ENV_KWARGS = {
    "num_assets":          5,
    "num_steps":           252,          # one trading year per episode
    "initial_price":       [311.42, 298.85, 148.29, 231.03, 752.82],
    "drift":               [0.2266, 0.3093, 0.1792, 0.1796, 0.2199],
    "volatility":          [0.2566, 0.2272, 0.2306, 0.1725, 0.1506],
    "phi":                 0.0,
    "randomize_phi":       True,
    "df":                  11,
    "garch_alpha":         0.1033,
    "garch_beta":          0.6441,
    "jump_intensity":      1.81,
    "jump_mean":           -0.011,
    "jump_std":            0.0642,
    "starting_cash":       100_000,
    "max_shares":          10_000,
    "max_trade_size":      1_000,
    "transaction_cost_pct": 0.0005,
}

ENV_KWARGS["correlation"] = np.array([
    [+1.0000, +0.3026, +0.1235, +0.0469, +0.6617],
    [+0.3026, +1.0000, +0.2405, +0.1131, +0.6013],
    [+0.1235, +0.2405, +1.0000, +0.1369, +0.1880],
    [+0.0469, +0.1131, +0.1369, +1.0000, +0.0533],
    [+0.6617, +0.6013, +0.1880, +0.0533, +1.0000],
])

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

N_ENVS          = 8          # parallel workers — set to cpu_count() if needed
TOTAL_TIMESTEPS = 7_000_000
N_STEPS         = 4096       # doubled — more episodes per update for better critic statistics
BATCH_SIZE      = 128        # medium =  less gradient steps per rollout
N_EPOCHS        = 10         # passes over rollout data per update
GAMMA           = 0.995      # higher discount for 252-step episodes
GAE_LAMBDA      = 0.95       # GAE smooths advantage estimates
LEARNING_RATE   = 3e-5
CLIP_RANGE      = 0.2        # standard PPO default (was 0.1, too conservative)
TARGET_KL       = 0.05       # relaxed so n_epochs=20 is actually consumed
ENT_COEF_START  = 0.005      # reduced from 0.05 — policy stays cautious while critic learns friction cost
ENT_COEF_END    = 0.001      # decays to near-zero as policy converges
VF_COEF         = 0.75       # upweights critic loss to fix explained_variance

EVAL_FREQ       = 8_192      # timesteps between evaluations (wall-clock, not per-env)
N_EVAL_EPISODES = 25

NUM_ASSETS      = ENV_KWARGS["num_assets"]
LOOKBACK_WINDOW = 30         # must match SpectralTradingEnv default

OUTPUT_DIR      = "./models"
LOG_DIR         = "./logs"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    # --- Training environments (parallel) ---
    print(f"Spawning {N_ENVS} training environments...")
    train_vec = make_vec_env(
        SpectralTradingEnv,
        n_envs=N_ENVS,
        env_kwargs=ENV_KWARGS,
        vec_env_cls=SubprocVecEnv,
    )
    # Normalise observations and rewards to stabilise critic training.
    # clip_obs=10 prevents extreme values from rare jump/GARCH events.
    train_vec = VecNormalize(
        train_vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=GAMMA,
    )

    # --- Evaluation environment (single, same seed distribution) ---
    eval_env = make_vec_env(
        SpectralTradingEnv,
        n_envs=1,
        env_kwargs=ENV_KWARGS,
    )
    # Share normalisation stats from training env — do NOT update them
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # eval rewards should be unnormalised for readability
        clip_obs=10.0,
        gamma=GAMMA,
        training=False,      # stats are read-only during eval
    )

    # --- Callbacks ---
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=200,
        min_evals=100,
        verbose=1,
    )

    # Granular diagnostics — fires every EVAL_FREQ timesteps independently
    # of EvalCallback, writes logs/diagnostics.npz after each checkpoint
    diag_callback = DiagnosticsCallback(
        eval_env=eval_env,
        train_env=train_vec,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        log_path=LOG_DIR,
    )

    # Linearly decays ent_coef each rollout (callable ent_coef not supported
    # by SB3 MlpPolicy — must be updated directly on the model object)
    ent_callback = EntropyCoefficientCallback(
        start=ENT_COEF_START,
        end=ENT_COEF_END,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(OUTPUT_DIR, "best_model"),
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),  # convert to per-env steps
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        callback_on_new_best=None,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    # --- Policy architecture ---
    # Critic gets a deeper network than the actor — the value function in a
    # stochastic, regime-switching market is harder to approximate than the
    # policy, so it needs more representational capacity.
    policy_kwargs = dict(
        log_std_init=-1.0,               # std ≈ 0.37 — enough to push past the 0.05 action dead zone
        features_extractor_class=SpectralExtractor,
        features_extractor_kwargs=dict(
            num_assets=NUM_ASSETS,
            lookback_window=LOOKBACK_WINDOW,
            asset_embed_dim=32,
            meta_embed_dim=16,
        ),
        net_arch=dict(
            pi=[64, 64],                 # actor
            vf=[128, 128, 64],           # deeper critic for noisy value landscape
        ),
    )

    # --- Model ---
    model = PPO(
        "MlpPolicy",
        train_vec,
        verbose=1,
        device="cpu",                    # MLP policies run faster on CPU than GPU
        tensorboard_log=LOG_DIR,
        # Rollout
        n_steps=N_STEPS,
        # Optimisation
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        # PPO clipping
        clip_range=CLIP_RANGE,
        target_kl=TARGET_KL,
        # Returns / advantage
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        # Loss coefficients — ent_coef starts at ENT_COEF_START and is
        # decayed each rollout by EntropyCoefficientCallback
        ent_coef=ENT_COEF_START,
        vf_coef=VF_COEF,
        # Architecture
        policy_kwargs=policy_kwargs,
    )

    print(f"\nModel policy:\n{model.policy}\n")
    print(f"Extractor output dim : {NUM_ASSETS * 32 + 16}")  # (5×32) price embeddings + 16 meta_embed_dim = 176
    print(f"Total parameters     : {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} envs...\n")

    # --- Train ---
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[ent_callback, diag_callback, eval_callback],
        progress_bar=True,
    )

    # --- Save final model and normalisation stats ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save(final_model_path)
    train_vec.save(os.path.join(LOG_DIR, "vecnormalize.pkl"))
    print(f"\nFinal model saved to {final_model_path}.zip")
    print(f"VecNormalize stats saved to {LOG_DIR}/vecnormalize.pkl")

    train_vec.close()
    eval_env.close()


if __name__ == "__main__":
    main()
