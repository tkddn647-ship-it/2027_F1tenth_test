"""
train_sac_connectome.py
=======================
SAC + LiDAR/IMU encoders + ConnectomeRNN temporal memory (학습).

  python train_sac_connectome.py --use-cache --map Spielberg --timesteps 200000 --fresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import SAC
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
from train_callbacks import make_train_callbacks


def resolve_device(name: str) -> str:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[train] WARN: CUDA 요청했지만 사용 불가 → cpu")
        return "cpu"
    return name


def build_brain(
    n_neurons,
    real_token,
    use_cache,
    cache_dir,
    seed,
    use_subcircuit,
    max_neurons,
):
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
        neurons_df, conn_df = select_speed_relevant_subcircuit(
            neurons_df, conn_df, max_neurons=max_neurons
        )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="hemibrain_cache")
    parser.add_argument("--real-token", type=str, default=None)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--max-neurons", type=int, default=256)
    parser.add_argument("--no-subcircuit", action="store_true")
    parser.add_argument("--freeze-connectome", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="default: 512 on CUDA, 256 on CPU",
    )
    parser.add_argument("--learning-starts", type=int, default=3_000)
    parser.add_argument("--min-speed", type=float, default=2.0)
    parser.add_argument("--max-speed", type=float, default=7.0)
    parser.add_argument("--max-steer", type=float, default=0.3735)
    parser.add_argument("--physics", type=str, default="st", choices=["st", "kinematic"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto|cuda|cpu|mps — Jetson/PC GPU면 cuda 권장",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    batch_size = args.batch_size if args.batch_size is not None else (
        512 if device == "cuda" else 256
    )
    learn_connectome = not args.freeze_connectome
    print(f"[train] official_f1tenth_gym={try_official_f1tenth_gym()} (fallback OK)")
    print(f"[train] obs_dim={OBS_DIM} map={args.map} algo=SAC device={device}")
    if device == "cuda":
        print(f"[train] GPU={torch.cuda.get_device_name(0)}")
    print(
        f"[train] speed=[{args.min_speed},{args.max_speed}] steer={args.max_steer} "
        f"max_neurons={args.max_neurons} learn_connectome={learn_connectome} "
        f"batch={batch_size}"
    )

    def _env_fn():
        return Monitor(
            F1TenthMaplessEnv(
                map_name=args.map,
                seed=args.seed,
                min_speed=args.min_speed,
                max_speed=args.max_speed,
                max_steer=args.max_steer,
                physics=args.physics,
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
        max_neurons=args.max_neurons,
    )
    policy_kwargs = make_sac_policy_kwargs(
        A_signed, input_idx, output_idx, learn_connectome=learn_connectome
    )
    custom_objects = {"ConnectomeTemporalFeatures": ConnectomeTemporalFeatures}

    save_path = args.save_path or f"connectome_sac_f1tenth_{args.map}.zip"
    resume_path = None if args.fresh else args.resume

    if resume_path and Path(resume_path).exists():
        print(f"[train] resume {resume_path}")
        model = SAC.load(
            resume_path,
            env=env,
            custom_objects=custom_objects,
            device=device,
        )
        model.set_env(env)
        reset_ts = False
    else:
        if args.fresh and Path(save_path).exists():
            bak = Path(save_path).with_suffix(".bak.zip")
            Path(save_path).replace(bak)
            print(f"[train] fresh: moved old -> {bak.name}")
        print("[train] fresh SAC + temporal ConnectomeRNN")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=batch_size,
            learning_starts=args.learning_starts,
            gamma=0.99,
            tau=0.005,
            train_freq=4,
            gradient_steps=4,
            ent_coef="auto",
            target_entropy=-2.0,
            verbose=1,
            seed=args.seed,
            device=device,
        )
        reset_ts = True

    log_dir = Path(f"runs/connectome_{args.map}")
    callbacks = make_train_callbacks(
        _env_fn, log_dir, n_envs=args.n_envs, seed=args.seed
    )
    n_train = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.policy.parameters())
    print(f"[train] params trainable={n_train:,} / all={n_all:,} timesteps={args.timesteps}")
    print(f"[train] logs/best → {log_dir}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_ts,
    )
    model.save(save_path)
    best = log_dir / "best" / "best_model.zip"
    if best.exists():
        print(f"[train] best eval model: {best}")
    print(f"[train] saved last {save_path}")
    print(f"watch: python watch_sac_f1tenth.py --model {best if best.exists() else save_path} --map {args.map} --use-cache")


if __name__ == "__main__":
    main()
