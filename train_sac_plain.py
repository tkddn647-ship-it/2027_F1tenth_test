"""Plain MLP SAC baseline — verify env is learnable without connectome."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from f1tenth_mapless_env import F1TenthMaplessEnv, OBS_DIM


class ProgressLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.laps = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("lap_completed"):
                self.laps += 1
                print(f"[lap] n={self.laps} time={info.get('lap_time')}")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=str, default="Spielberg")
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--min-speed", type=float, default=2.0)
    p.add_argument("--max-speed", type=float, default=3.5)
    p.add_argument("--max-steer", type=float, default=0.30)
    args = p.parse_args()

    def _env_fn():
        return Monitor(
            F1TenthMaplessEnv(
                map_name=args.map,
                seed=args.seed,
                min_speed=args.min_speed,
                max_speed=args.max_speed,
                max_steer=args.max_steer,
            )
        )

    env = make_vec_env(_env_fn, n_envs=args.n_envs, seed=args.seed)
    save_path = f"plain_sac_f1tenth_{args.map}.zip"
    print(f"[plain-sac] obs={OBS_DIM} map={args.map} speed=[{args.min_speed},{args.max_speed}]")

    if args.fresh and Path(save_path).exists():
        Path(save_path).replace(Path(save_path).with_suffix(".bak.zip"))

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=2_000,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy=-1.0,
        verbose=1,
        seed=args.seed,
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps, callback=ProgressLogger())
    model.save(save_path)
    print(f"[plain-sac] saved {save_path}")


if __name__ == "__main__":
    main()
