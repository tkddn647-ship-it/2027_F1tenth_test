"""
lidarfly.train
==============
LiDAR 1080빔 × 3프레임 + 초파리 커넥톰 정책 SAC.

  python -m lidarfly.train --timesteps 1000000 --n-envs 4 --subproc \\
      --save-dir runs/lidarfly --resume auto --v-max 7

actor = 커넥톰 DN + state 를 선형으로 읽기 (--pi-net "" 기본).
critic 은 AsymSACPolicy 가 privileged 27 을 붙여 MLP 로 학습한다.
저장: last_model.zip (종료·주기), best_model.zip + best.json (평가 최고).
이어 학습: --resume auto → save-dir 안 정상 zip.

TODO: torch 없는 numpy 추론(np_actor)은 이번 범위 밖.
학습 본체는 코랩에서 돌린다.  이 파일은 여기서 긴 학습을 실행하지 않는 것을 전제로 한다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mapless40.config import GAMMA_40HZ
from mapless40.policy import AsymSACPolicy
from mapless40.train import (LapEvalCallback, TimeLimitCallback, _PolicyWarmupSAC, _find_resume,
                             _safe_save)

from .config import DEFAULT_MAPS, V_MAX, lidarfly_config
from .env import LidarFlyEnv, make_env
from .policy import LidarFlyFeatures


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=DEFAULT_MAPS)
    p.add_argument("--eval-maps", default="ifac,roboracer_0817")
    p.add_argument("--pi-net", default="", help="actor 은닉층 (기본 없음 = 커넥톰 DN 을 선형으로 읽기)")
    p.add_argument("--qf-net", default="256,256")
    p.add_argument("--n-dn", type=int, default=48)
    p.add_argument("--v-max", type=float, default=V_MAX, help="최고 속도 [m/s] (기본 7, 최저 2)")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer-size", type=int, default=150_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--gradient-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=GAMMA_40HZ,
                   help="기본 0.99**0.25 (40 Hz 에서 0.1 s 당 0.99)")
    p.add_argument("--target-entropy", type=float, default=-1.0)
    p.add_argument("--max-episode-s", type=float, default=60.0)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--ckpt-freq", type=int, default=100_000)
    p.add_argument("--save-every-min", type=float, default=10.0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None, help="이어 학습할 zip, 또는 auto (save-dir 안 최신 모델)")
    p.add_argument("--time-limit-min", type=float, default=0)
    return p


def _net_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def apply_train_cfg(args):
    cfg = lidarfly_config()
    cfg.action.v_max = float(args.v_max)
    cfg.raceline.v_max = float(args.v_max)
    cfg.max_episode_s = float(args.max_episode_s)
    if cfg.action.v_max < cfg.action.v_min:
        raise SystemExit(f"--v-max {cfg.action.v_max} < v_min {cfg.action.v_min}")
    return cfg


def policy_kwargs(args, cfg) -> dict:
    return dict(
        features_extractor_class=LidarFlyFeatures,
        features_extractor_kwargs=dict(n_dn=args.n_dn, range_max=float(cfg.lidar.range_max)),
        net_arch=dict(pi=_net_list(args.pi_net), qf=_net_list(args.qf_net)),
    )


def main():
    args = build_parser().parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = apply_train_cfg(args)
    maps = [m for m in args.maps.split(",") if m]
    eval_maps = [m for m in args.eval_maps.split(",") if m]
    save_dir = Path(args.save_dir or f"runs/lidarfly_{time.strftime('%Y%m%d_%H%M%S')}")
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps({"args": vars(args)}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(make_env(maps, seed=args.seed, cfg=cfg), n_envs=args.n_envs, seed=args.seed,
                       vec_env_cls=vec_cls)
    kw = policy_kwargs(args, cfg)

    resume = args.resume
    if resume == "auto":
        found = _find_resume(save_dir)
        resume = str(found) if found else None
        print(f"[train] resume auto → {resume or '없음, 새로 시작'}")
    if resume:
        model = SAC.load(resume, env=env, device=device)
        model.learning_starts = model.num_timesteps + min(args.learning_starts, 10_000)
        model.__class__ = _PolicyWarmupSAC
        reset_ts = False
    else:
        model = SAC(AsymSACPolicy, env, learning_rate=args.lr, buffer_size=args.buffer_size,
                    batch_size=args.batch_size, learning_starts=args.learning_starts, gamma=args.gamma,
                    tau=0.005, train_freq=1, gradient_steps=args.gradient_steps, ent_coef="auto",
                    target_entropy=args.target_entropy, policy_kwargs=kw, verbose=1,
                    seed=args.seed, device=device)
        reset_ts = True
    actor = model.policy.actor
    n_act = sum(p.numel() for p in actor.parameters())
    n_brain = sum(p.numel() for p in actor.features_extractor.parameters())
    print(f"[train] lidarfly (LiDAR + 커넥톰) pi={args.pi_net or '선형'} device={model.device} "
          f"actor_params={n_act:,} (brain {n_brain:,}) maps={maps} eval={eval_maps} gamma={args.gamma:.4f} "
          f"v={cfg.action.v_min:g}~{cfg.action.v_max:g} m/s")

    cbs = CallbackList([
        LapEvalCallback(eval_maps, cfg, save_dir, args.eval_freq, env_cls=LidarFlyEnv, env_cfg=cfg),
        CheckpointCallback(max(args.ckpt_freq // args.n_envs, 1), str(save_dir / "checkpoints"), name_prefix="sac"),
        TimeLimitCallback(args.time_limit_min, save_dir / "last_model", args.save_every_min),
    ])
    try:
        model.learn(total_timesteps=args.timesteps, callback=cbs, reset_num_timesteps=reset_ts)
    finally:
        if _safe_save(model, save_dir / "last_model.zip"):
            print(f"[train] saved {save_dir / 'last_model.zip'}")


if __name__ == "__main__":
    main()
