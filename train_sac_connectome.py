"""
train_sac_connectome.py
=======================
SAC + LiDAR/IMU encoders + ConnectomeRNN temporal memory.

  python train_sac_connectome.py --use-cache --map Spielberg --timesteps 300000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from connectome_loader import (
    build_adjacency,
    build_synthetic_hemibrain,
    identify_io_neurons,
    load_hemibrain_cache,
    load_hemibrain_real,
    select_speed_relevant_subcircuit,
)
from f1tenth_mapless_env import F1TenthMaplessEnv, OBS_DIM, try_official_f1tenth_gym
from policy_sac_connectome import ConnectomeTemporalFeatures, make_sac_policy_kwargs


def build_brain(n_neurons, real_token, use_cache, cache_dir, seed, use_subcircuit, refresh_cache):
    if use_cache:
        print("[train] hemibrain cache")
        neurons_df, conn_df = load_hemibrain_cache(cache_dir)
    elif real_token:
        print("[train] hemibrain download")
        neurons_df, conn_df = load_hemibrain_real(real_token)
    else:
        print(f"[train] synthetic connectome n={n_neurons}")
        neurons_df, conn_df = build_synthetic_hemibrain(n_neurons=n_neurons, seed=seed)

    if use_subcircuit:
        neurons_df, conn_df = select_speed_relevant_subcircuit(neurons_df, conn_df)
        neurons_df = neurons_df.reset_index(drop=True)

    input_idx, output_idx = identify_io_neurons(neurons_df)
    if len(input_idx) == 0 or len(output_idx) == 0:
        raise SystemExit(f"I/O empty in={len(input_idx)} out={len(output_idx)}")
    A_signed = build_adjacency(neurons_df, conn_df, signed=True)
    print(
        f"[train] neurons={len(neurons_df)} in={len(input_idx)} out={len(output_idx)} "
        f"density={100.0 * (A_signed != 0).mean():.4f}%"
    )
    return A_signed, input_idx, output_idx


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
                    f"map={info.get('track_name')}"
                )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="hemibrain_cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--real-token", type=str, default=None)
    parser.add_argument("--n-neurons", type=int, default=800)
    parser.add_argument("--no-subcircuit", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=3_000)
    parser.add_argument("--min-speed", type=float, default=2.0)
    parser.add_argument("--max-speed", type=float, default=4.0)
    parser.add_argument("--max-steer", type=float, default=0.30)
    args = parser.parse_args()

    print(f"[train] official_f1tenth_gym={try_official_f1tenth_gym()} (fallback racetracks OK)")
    print(f"[train] obs_dim={OBS_DIM} map={args.map} algo=SAC")
    print(f"[train] speed=[{args.min_speed},{args.max_speed}] steer={args.max_steer}")

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

    A_signed, input_idx, output_idx = build_brain(
        args.n_neurons,
        args.real_token,
        args.use_cache,
        args.cache_dir,
        args.seed,
        use_subcircuit=not args.no_subcircuit,
        refresh_cache=args.refresh_cache,
    )
    policy_kwargs = make_sac_policy_kwargs(A_signed, input_idx, output_idx)
    custom_objects = {"ConnectomeTemporalFeatures": ConnectomeTemporalFeatures}

    save_path = args.save_path or f"connectome_sac_f1tenth_{args.map}.zip"
    resume_path = None if args.fresh else args.resume

    if resume_path and Path(resume_path).exists():
        print(f"[train] resume {resume_path}")
        model = SAC.load(
            resume_path,
            env=env,
            custom_objects=custom_objects,
            device="auto",
        )
        model.set_env(env)
        reset_ts = False
    else:
        if args.fresh and Path(save_path).exists():
            bak = Path(save_path).with_suffix(".bak.zip")
            Path(save_path).replace(bak)
            print(f"[train] fresh: moved old -> {bak.name}")
        print("[train] fresh SAC start")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
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
        reset_ts = True

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[train] params={n_params:,} timesteps={args.timesteps}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=ProgressLogger(),
        reset_num_timesteps=reset_ts,
    )
    model.save(save_path)
    print(f"[train] saved {save_path}")
    print(f"watch: python watch_sac_f1tenth.py --model {save_path} --map {args.map} --use-cache")


if __name__ == "__main__":
    main()
