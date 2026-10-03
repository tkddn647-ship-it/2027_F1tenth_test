"""
camfly.train
============
카메라(파리 눈 + depth) + 초파리 커넥톰 정책 SAC 학습.

  python -m camfly.train --encoder fly --timesteps 1000000 --n-envs 4 --subproc \
      --save-dir runs/camfly_fly --resume auto --time-limit-min 150

mapless40.train 의 평가·저장·이어 학습 콜백을 그대로 쓴다.
기본 actor = 커넥톰 DN + state 를 선형으로 읽기 (--pi-net "" ).  비교용 --encoder cnn --pi-net 256,256.
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

from mapless40.policy import AsymSACPolicy
from mapless40.train import (LapEvalCallback, TimeLimitCallback, _PolicyWarmupSAC, _find_resume,
                             _safe_save)

from .config import CamFlyConfig
from .env import CamFlyEnv, make_env
from .policy import CamFeatures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default="ifac:3,roboracer_0817:3,Spielberg:1,Silverstone:1,Monza:1")
    p.add_argument("--eval-maps", default="ifac,roboracer_0817")
    p.add_argument("--encoder", choices=["fly", "cnn"], default="fly")
    p.add_argument("--pi-net", default="", help="actor 은닉층 (기본 없음 = 커넥톰 DN 을 선형으로 읽기)")
    p.add_argument("--qf-net", default="256,256")
    p.add_argument("--n-dn", type=int, default=48)
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
    p.add_argument("--gamma", type=float, default=0.99 ** (40.0 / 30.0 / 4.0))   # 0.1 s 당 0.99 (30 Hz)
    p.add_argument("--target-entropy", type=float, default=-1.0)
    p.add_argument("--max-episode-s", type=float, default=60.0)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--ckpt-freq", type=int, default=100_000)
    p.add_argument("--save-every-min", type=float, default=10.0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--time-limit-min", type=float, default=0)
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cf = CamFlyConfig()
    cf.env.max_episode_s = args.max_episode_s
    maps = [m for m in args.maps.split(",") if m]
    eval_maps = [m for m in args.eval_maps.split(",") if m]
    save_dir = Path(args.save_dir or f"runs/camfly_{args.encoder}_{time.strftime('%Y%m%d_%H%M%S')}")
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps({"args": vars(args)}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(make_env(maps, seed=args.seed, cfg=cf), n_envs=args.n_envs, seed=args.seed,
                       vec_env_cls=vec_cls)
    net = lambda s: [int(x) for x in s.split(",") if x.strip()]  # noqa: E731
    policy_kwargs = dict(features_extractor_class=CamFeatures,
                         features_extractor_kwargs=dict(encoder=args.encoder, n_dn=args.n_dn),
                         net_arch=dict(pi=net(args.pi_net), qf=net(args.qf_net)))

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
                    target_entropy=args.target_entropy, policy_kwargs=policy_kwargs, verbose=1,
                    seed=args.seed, device=device)
        reset_ts = True
    actor = model.policy.actor
    n_act = sum(p.numel() for p in actor.parameters())
    n_brain = sum(p.numel() for p in actor.features_extractor.parameters())
    print(f"[train] camfly encoder={args.encoder} pi={args.pi_net or '선형'} device={model.device} "
          f"actor_params={n_act:,} (brain {n_brain:,}) maps={maps} eval={eval_maps} gamma={args.gamma:.4f}")

    cbs = CallbackList([
        LapEvalCallback(eval_maps, cf.env, save_dir, args.eval_freq, env_cls=CamFlyEnv, env_cfg=cf),
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
