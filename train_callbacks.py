"""Shared SB3 callbacks: checkpoint + deterministic eval + best model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env


class ProgressLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.laps = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("lap_completed"):
                self.laps += 1
                print(
                    f"[lap] n={self.laps} time={info.get('lap_time')} "
                    f"prog={info.get('progress')} map={info.get('track_name')}"
                )
        return True


class TrainStatsLogger(BaseCallback):
    """Print ent_coef / losses periodically."""

    def __init__(self, every: int = 2000):
        super().__init__()
        self.every = every

    def _on_step(self) -> bool:
        if self.n_calls % self.every != 0:
            return True
        ent = getattr(self.model, "ent_coef", None)
        if callable(ent):
            try:
                ent = float(ent().detach().cpu().item())
            except Exception:
                ent = None
        elif hasattr(self.model, "log_ent_coef"):
            try:
                ent = float(self.model.log_ent_coef.exp().detach().cpu().item())
            except Exception:
                ent = ent
        print(
            f"[stats] steps={self.num_timesteps} ent_coef~{ent} "
            f"(see rollout/train tables for critic/actor loss)"
        )
        return True


class LengthAwareEvalCallback(EvalCallback):
    """Save best by mean episode length (tie-break: mean reward).

    Pure reward ranking collapses when longer rollouts accumulate more
    negative CTE terms than short crashes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_length = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        prev_path = self.best_model_save_path
        prev_best_r = self.best_mean_reward
        self.best_model_save_path = None
        continue_training = super()._on_step()
        self.best_model_save_path = prev_path
        self.best_mean_reward = prev_best_r

        if len(self.evaluations_length) == 0:
            return continue_training

        mean_reward = float(self.last_mean_reward)
        mean_length = float(np.mean(self.evaluations_length[-1]))
        improved = mean_length > self.best_mean_length + 1e-6 or (
            abs(mean_length - self.best_mean_length) < 1e-6
            and mean_reward > self.best_mean_reward
        )
        if improved:
            self.best_mean_length = mean_length
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(
                    f"New best mean ep_length={mean_length:.1f} "
                    f"(reward={mean_reward:.2f})!"
                )
            if self.best_model_save_path is not None:
                self.model.save(str(Path(self.best_model_save_path) / "best_model"))
        return continue_training


def make_train_callbacks(
    env_fn,
    log_dir: str | Path,
    n_envs: int,
    seed: int,
    eval_freq: int = 5000,
    ckpt_freq: int = 10_000,
    n_eval_episodes: int = 5,
) -> CallbackList:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir = log_dir / "best"
    ckpt_dir = log_dir / "checkpoints"
    best_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    eval_env = make_vec_env(env_fn, n_envs=1, seed=seed + 1000)
    eval_cb = LengthAwareEvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=max(eval_freq // max(n_envs, 1), 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(ckpt_freq // max(n_envs, 1), 1),
        save_path=str(ckpt_dir),
        name_prefix="sac",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    return CallbackList([ProgressLogger(), TrainStatsLogger(), eval_cb, ckpt_cb])
