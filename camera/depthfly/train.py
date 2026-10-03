"""
depthfly.train
============
depth 카메라 + 초파리 커넥톰만 쓰는 정책 SAC 학습.

  python -m camera.depthfly.train --timesteps 1000000 --n-envs 4 --subproc \
      --save-dir runs/depthfly --resume auto --time-limit-min 150

mapless40.train 의 평가·저장·이어 학습 콜백을 그대로 쓴다.
actor = 커넥톰 DN + state 를 선형으로 읽기 (--pi-net "" 기본).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mapless40.policy import AsymSACPolicy
from mapless40.train import (LapEvalCallback, TimeLimitCallback, _PolicyWarmupSAC, _find_resume,
                             _safe_save)

from .config import DepthFlyConfig, add_cfg_args
from .env import DepthFlyEnv, make_env
from .policy import NearFeatures


def load_bc_init(model, path, cf, log_std: float) -> None:
    """bc.py 결과를 SAC 정책에 넣는다: 회로(actor·critic·target 모두) + actor 읽기층 μ, 잡음 σ 는 작게."""
    import numpy as np
    from .brain import PARAM_KEYS
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    if abs(float(meta.get("v_max", cf.env.action.v_max)) - cf.env.action.v_max) > 1e-6:
        raise SystemExit(f"[bc-init] v_max 불일치: npz {meta.get('v_max')} vs 학습 {cf.env.action.v_max} — 같은 --v-max 로")
    pol = model.policy
    with torch.no_grad():
        for fe in (pol.actor.features_extractor, pol.critic.features_extractor):
            b = fe.brain
            for k in PARAM_KEYS:
                getattr(b, k).copy_(torch.from_numpy(d[f"brain.{k}"]))
            b.mask.copy_(torch.from_numpy(d["mask"])); b.sign.copy_(torch.from_numpy(d["sign"]))
        pol.actor.mu.weight.copy_(torch.from_numpy(d["mu_w"])); pol.actor.mu.bias.copy_(torch.from_numpy(d["mu_b"]))
        pol.actor.log_std.weight.zero_(); pol.actor.log_std.bias.fill_(log_std)
    pol.critic_target.load_state_dict(pol.critic.state_dict())
    print(f"[bc-init] {path} (모방학습 반복 {meta.get('iter')}, 시험 {meta.get('eval', meta.get('lap_test'))})")


class ActorFreeze(BaseCallback):
    """처음 until 스텝까지 actor 파라미터를 얼려 critic 만 배우게 한다 (grad 없음 → Adam 이 건너뜀)."""

    def __init__(self, until: int):
        super().__init__()
        self.until, self.frozen = until, None

    def _set(self, frozen: bool):
        if self.frozen is frozen:
            return
        for q in self.model.policy.actor.parameters():
            q.requires_grad_(not frozen)
        self.frozen = frozen
        print(f"[bc-init] actor {'고정 (critic 먼저)' if frozen else '학습 시작'} @ {self.num_timesteps:,}")

    def _on_training_start(self):
        self._set(self.model.num_timesteps < self.until)

    def _on_step(self) -> bool:
        self._set(self.num_timesteps < self.until)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default="ifac:3,roboracer_0817:3,Spielberg:1,Silverstone:1,Monza:1")
    p.add_argument("--eval-maps", default="ifac,roboracer_0817")
    p.add_argument("--pi-net", default="", help="actor 은닉층 (기본 없음 = 커넥톰 DN 을 선형으로 읽기)")
    p.add_argument("--qf-net", default="256,256")
    p.add_argument("--n-dn", type=int, default=48)
    p.add_argument("--fps", type=float, default=30.0, help="Gemini 2L depth 최대 30 fps")
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
    p.add_argument("--gamma", type=float, default=None, help="기본: 0.1 s 당 0.99 = 0.99^(10/fps)")
    p.add_argument("--target-entropy", type=float, default=None,
                   help="기본: -1 (무작위 시작) / -3.5 (--bc-init: σ≈0.1 정책의 엔트로피 ≈ -3.2 보다 낮게 → 엔트로피 계수가 커지지 않게)")
    p.add_argument("--max-episode-s", type=float, default=60.0)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--ckpt-freq", type=int, default=100_000)
    p.add_argument("--save-every-min", type=float, default=10.0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--time-limit-min", type=float, default=0)
    p.add_argument("--bc-init", default=None, help="모방학습 결과 npz (camera.depthfly.bc) 로 actor·critic 회로를 시작")
    p.add_argument("--actor-freeze-steps", type=int, default=30_000,
                   help="--bc-init 일 때 처음 이만큼은 critic 만 학습 (엉터리 critic 이 모방한 정책을 망가뜨리는 것 방지)")
    p.add_argument("--ent-init", type=float, default=0.02, help="--bc-init 일 때 엔트로피 계수 시작값 (기본 auto 는 1.0)")
    p.add_argument("--bc-log-std", type=float, default=-2.3,
                   help="--bc-init 일 때 행동 잡음 log σ 시작값 (σ≈0.1. σ 0.2 면 모방한 정책이 8초 안에 충돌 — 시뮬 확인)")
    p.add_argument("--wiring", default="bilateral", choices=["bilateral", "random"], help="DN 배선 (v2 까지 random)")
    add_cfg_args(p)
    args = p.parse_args()
    if args.target_entropy is None:
        args.target_entropy = -3.5 if args.bc_init else -1.0

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cf = DepthFlyConfig(fps=args.fps, v_max=args.v_max, brake=args.brake)
    if args.gamma is None:
        args.gamma = 0.99 ** (10.0 / args.fps)
    cf.env.max_episode_s = args.max_episode_s
    maps = [m for m in args.maps.split(",") if m]
    eval_maps = [m for m in args.eval_maps.split(",") if m]
    save_dir = Path(args.save_dir or f"runs/depthfly_{time.strftime('%Y%m%d_%H%M%S')}")
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps({"args": vars(args)}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(make_env(maps, seed=args.seed, cfg=cf), n_envs=args.n_envs, seed=args.seed,
                       vec_env_cls=vec_cls)
    net = lambda s: [int(x) for x in s.split(",") if x.strip()]  # noqa: E731
    policy_kwargs = dict(features_extractor_class=NearFeatures,
                         features_extractor_kwargs=dict(n_dn=args.n_dn, wiring=args.wiring),
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
                    tau=0.005, train_freq=1, gradient_steps=args.gradient_steps,
                    ent_coef=f"auto_{args.ent_init}" if args.bc_init else "auto",
                    target_entropy=args.target_entropy, policy_kwargs=policy_kwargs, verbose=1,
                    seed=args.seed, device=device)
        reset_ts = True
        if args.bc_init:
            load_bc_init(model, args.bc_init, cf, args.bc_log_std)
            model.__class__ = _PolicyWarmupSAC          # 버퍼 채우는 동안에도 무작위 대신 모방한 정책으로 운전
    actor = model.policy.actor
    n_act = sum(p.numel() for p in actor.parameters())
    n_brain = sum(p.numel() for p in actor.features_extractor.parameters())
    print(f"[train] depthfly (depth + 커넥톰) pi={args.pi_net or '선형'} device={model.device} "
          f"actor_params={n_act:,} (brain {n_brain:,}) maps={maps} eval={eval_maps} gamma={args.gamma:.4f} "
          f"v={cf.env.action.v_min:g}~{cf.env.action.v_max:g} m/s brake={cf.env.act.brake_enabled}")

    cb_list = []
    if args.bc_init and not resume:
        cb_list.append(ActorFreeze(model.learning_starts + args.actor_freeze_steps))
    cbs = CallbackList(cb_list + [
        LapEvalCallback(eval_maps, cf.env, save_dir, args.eval_freq, env_cls=DepthFlyEnv, env_cfg=cf),
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
