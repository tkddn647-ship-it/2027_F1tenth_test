"""Shared SB3 callbacks: checkpoint + deterministic eval + best model."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


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
    """Print ent_coef / losses periodically (원인 4·2 진단용)."""

    def __init__(self, every: int = 2000):
        super().__init__()
        self.every = every

    def _on_step(self) -> bool:
        if self.n_calls % self.every != 0:
            return True
        logger = self.model.logger
        # SB3 dumps to logger; also peek model attrs
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
            f"[stats] steps={self.num_timesteps} ent_coef≈{ent} "
            f"(see rollout/train tables for critic/actor loss)"
        )
        return True


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
    eval_cb = EvalCallback(
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
