"""
spectral_env_core.callbacks
============================
Stable Baselines3 callbacks designed for SpectralTradingEnv training pipelines.

Available callbacks:
    - DiagnosticsCallback  : granular training diagnostics with VecNormalize sync
    - EntropyCoefficientCallback : linear entropy decay (workaround for SB3 limitation)

Usage
-----
    from spectral_env_core.callbacks import DiagnosticsCallback, EntropyCoefficientCallback

    diag = DiagnosticsCallback(
        eval_env=eval_env,
        train_env=train_vec,
        eval_freq=8192,
    )
    ent = EntropyCoefficientCallback(start=0.01, end=0.001)
    model.learn(total_timesteps=7_000_000, callback=[ent, diag, eval_callback])
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class EntropyCoefficientCallback(BaseCallback):
    """
    Linearly decays the PPO entropy coefficient from `start` to `end`
    over the course of training.

    Required because SB3's PPO does not accept a callable for ent_coef —
    it must be a float. This callback updates model.ent_coef directly
    at the start of each rollout.

    Parameters
    ----------
    start : float
        Initial entropy coefficient (at timestep 0).
    end : float
        Final entropy coefficient (at total_timesteps).

    Example
    -------
        ent_cb = EntropyCoefficientCallback(start=0.01, end=0.001)
        model.learn(callback=[ent_cb])
    """

    def __init__(self, start: float = 0.01, end: float = 0.001):
        super().__init__(verbose=0)
        self.start = start
        self.end   = end

    def _on_rollout_start(self) -> None:
        progress = 1.0 - self.num_timesteps / self.model._total_timesteps
        self.model.ent_coef = float(self.end + (self.start - self.end) * progress)

    def _on_step(self) -> bool:
        return True


class DiagnosticsCallback(BaseCallback):
    """
    Periodic training diagnostics for SpectralTradingEnv.

    Runs deterministic eval episodes at fixed intervals and logs granular
    metrics to a compressed `.npz` file. Also handles VecNormalize stat
    synchronisation between training and eval environments automatically.

    Metrics captured per checkpoint:
      - Mean / std / min / max episode reward
      - Episode truncation rate (bankruptcy / hard stop / delisting)
      - Mean cumulative friction (transaction costs) per episode
      - Action statistics: mean |action|, std, fraction zeroed by dead zone
      - Regime exposure distribution
      - Explained variance (from SB3 logger)
      - Current entropy coefficient

    Parameters
    ----------
    eval_env : VecNormalize
        Evaluation environment wrapped in VecNormalize.
    train_env : VecNormalize
        Training environment wrapped in VecNormalize. Stats are synced
        from here to eval_env before each diagnostic run.
    n_eval_episodes : int
        Number of episodes to run per diagnostic checkpoint.
    eval_freq : int
        Run diagnostics every `eval_freq` total timesteps.
    log_path : str
        Directory to write `diagnostics.npz`.
    verbose : int
        If > 0, prints a summary line per checkpoint.

    Example
    -------
        from spectral_env_core.callbacks import DiagnosticsCallback

        diag = DiagnosticsCallback(
            eval_env=eval_env,
            train_env=train_vec,
            eval_freq=8192,
            log_path="./logs",
        )
        model.learn(callback=[diag])

        # After training:
        data = np.load("./logs/diagnostics.npz")
        print(data["mean_reward"])
    """

    def __init__(
        self,
        eval_env: VecNormalize,
        train_env: VecNormalize,
        n_eval_episodes: int = 25,
        eval_freq: int = 8_192,
        log_path: str = "./logs",
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env        = eval_env
        self.train_env       = train_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq       = eval_freq
        self.log_path        = log_path

        # Accumulated metrics
        self._timesteps:       list = []
        self._mean_reward:     list = []
        self._std_reward:      list = []
        self._min_reward:      list = []
        self._max_reward:      list = []
        self._truncation_rate: list = []
        self._mean_friction:   list = []
        self._mean_abs_action: list = []
        self._std_action:      list = []
        self._dead_zone_frac:  list = []
        self._regime_fracs:    list = []
        self._ent_coef:        list = []
        self._explained_var:   list = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._run_diagnostics()
        return True

    def _sync_normalisation(self) -> None:
        """Copy running obs/reward stats from training to eval VecNormalize."""
        if hasattr(self.train_env, 'obs_rms'):
            self.eval_env.obs_rms = self.train_env.obs_rms
        if hasattr(self.train_env, 'ret_rms'):
            self.eval_env.ret_rms = self.train_env.ret_rms

    def _run_diagnostics(self) -> None:
        """Run eval episodes and collect all diagnostic metrics."""
        self._sync_normalisation()

        episode_rewards   = []
        episode_truncated = []
        episode_friction  = []
        episode_actions   = []
        episode_regimes   = []

        obs = self.eval_env.reset()
        ep_reward   = 0.0
        ep_friction = 0.0
        ep_actions  = []
        ep_regimes  = []
        eps_done    = 0

        while eps_done < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)

            ep_reward   += float(reward[0])
            ep_friction += float(info[0].get("friction_leak", 0.0))
            ep_actions.append(action[0].copy())
            ep_regimes.append(int(info[0].get("regime", 0)))

            if done[0]:
                step = int(info[0].get("step", 252))
                episode_truncated.append(step < 252)
                episode_rewards.append(ep_reward)
                episode_friction.append(ep_friction)
                episode_actions.append(np.vstack(ep_actions))
                episode_regimes.append(ep_regimes[:])

                ep_reward   = 0.0
                ep_friction = 0.0
                ep_actions  = []
                ep_regimes  = []
                eps_done   += 1
                obs = self.eval_env.reset()

        # --- Aggregate ---
        rewards     = np.array(episode_rewards)
        all_actions = np.vstack(episode_actions)
        all_regimes = np.concatenate(episode_regimes)

        abs_actions    = np.abs(all_actions)
        dead_zone_frac = float((abs_actions < 0.05).mean())

        n_regimes    = max(int(all_regimes.max()) + 1, 2)
        regime_fracs = np.array([
            float((all_regimes == r).mean()) for r in range(n_regimes)
        ])

        ent_coef = float(self.model.ent_coef)
        ev = self.model.logger.name_to_value.get("train/explained_variance", float("nan"))

        # --- Store ---
        self._timesteps.append(self.num_timesteps)
        self._mean_reward.append(float(rewards.mean()))
        self._std_reward.append(float(rewards.std()))
        self._min_reward.append(float(rewards.min()))
        self._max_reward.append(float(rewards.max()))
        self._truncation_rate.append(float(np.mean(episode_truncated)))
        self._mean_friction.append(float(np.mean(episode_friction)))
        self._mean_abs_action.append(float(abs_actions.mean()))
        self._std_action.append(float(all_actions.std()))
        self._dead_zone_frac.append(dead_zone_frac)
        self._regime_fracs.append(regime_fracs)
        self._ent_coef.append(ent_coef)
        self._explained_var.append(float(ev))

        if self.verbose:
            print(
                f"[Diagnostics] step={self.num_timesteps:>8,} | "
                f"reward={rewards.mean():>8.2f} ± {rewards.std():.2f} | "
                f"trunc={np.mean(episode_truncated)*100:.1f}% | "
                f"dead_zone={dead_zone_frac*100:.1f}% | "
                f"exp_var={float(ev):.4f} | "
                f"ent={ent_coef:.5f}"
            )

        # --- Save ---
        os.makedirs(self.log_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.log_path, "diagnostics.npz"),
            timesteps       = np.array(self._timesteps),
            mean_reward     = np.array(self._mean_reward),
            std_reward      = np.array(self._std_reward),
            min_reward      = np.array(self._min_reward),
            max_reward      = np.array(self._max_reward),
            truncation_rate = np.array(self._truncation_rate),
            mean_friction   = np.array(self._mean_friction),
            mean_abs_action = np.array(self._mean_abs_action),
            std_action      = np.array(self._std_action),
            dead_zone_frac  = np.array(self._dead_zone_frac),
            regime_fracs    = np.array(self._regime_fracs),
            ent_coef        = np.array(self._ent_coef),
            explained_var   = np.array(self._explained_var),
        )
