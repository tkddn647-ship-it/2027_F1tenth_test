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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mapless40.config import GAMMA_40HZ
from mapless40.policy import AsymSACPolicy
from mapless40.train import (LapEvalCallback, TimeLimitCallback, _PolicyWarmupSAC, _find_resume,
                             _safe_save)


def _has_tb() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False

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
    p.add_argument("--target-entropy", type=float, default=None,
                   help="기본: -1 (무작위 시작) / -3.5 (--bc-init: σ≈0.1 정책 엔트로피보다 낮게, depthfly v3 와 같음)")
    p.add_argument("--max-episode-s", type=float, default=60.0)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--ckpt-freq", type=int, default=100_000)
    p.add_argument("--save-every-min", type=float, default=10.0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None, help="이어 학습할 zip, 또는 auto (save-dir 안 최신 모델)")
    p.add_argument("--time-limit-min", type=float, default=0)
    p.add_argument("--bc-init", default=None, help="모방학습 결과 npz (lidarfly.bc) 로 actor·critic 회로를 시작")
    p.add_argument("--actor-freeze-steps", type=int, default=30_000,
                   help="--bc-init 일 때 처음 이만큼은 critic 만 학습 (엉터리 critic 이 모방한 정책을 망가뜨리는 것 방지)")
    p.add_argument("--ent-init", type=float, default=0.02, help="--bc-init 일 때 엔트로피 계수 시작값 (auto 기본은 1.0)")
    p.add_argument("--bc-log-std", type=float, default=-2.3, help="--bc-init 일 때 행동 잡음 log σ 시작값 (σ≈0.1)")
    p.add_argument("--compile", action="store_true",
                   help="커넥톰 회로 forward 를 torch.compile (작은 커널을 합쳐 GPU 업데이트 약 2배, 알고리즘 동일)")
    return p


def _ensure_python_headers() -> None:
    """torch.compile(inductor) 가 C 확장을 빌드할 때 Python.h 가 필요하다.
    시스템에 python3-dev 가 없으면 uv 가 받은 같은 마이너 버전 Python 의 헤더를 CPATH 에 넣는다."""
    import os
    import sys
    import sysconfig
    inc = Path(sysconfig.get_paths()["include"])
    if (inc / "Python.h").exists():
        return
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for h in sorted(Path.home().glob(f".local/share/uv/python/cpython-{sys.version_info.major}."
                                     f"{sys.version_info.minor}*/include/{ver}/Python.h")):
        os.environ["CPATH"] = f"{h.parent}:{os.environ['CPATH']}" if os.environ.get("CPATH") else str(h.parent)
        print(f"[train] Python.h 없음 → {h.parent} 사용", flush=True)
        return
    raise SystemExit("--compile: Python.h 가 없다. python3-dev 를 설치하거나 `uv python install "
                     f"{sys.version_info.major}.{sys.version_info.minor}` 후 다시 실행")


def load_bc_init(model, path, cfg, log_std: float) -> None:
    """lidarfly.bc 결과를 SAC 정책에 넣는다: 회로(actor·critic·target) + actor 읽기층 μ, 잡음 σ 는 작게."""
    import numpy as np
    from camera.depthfly.brain import PARAM_KEYS
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    if meta.get("sensor") != "lidar1080":
        raise SystemExit(f"[bc-init] {path} 는 lidarfly.bc 결과가 아니다 (sensor={meta.get('sensor')})")
    if abs(float(meta["v_max"]) - cfg.action.v_max) > 1e-6:
        raise SystemExit(f"[bc-init] v_max 불일치: npz {meta['v_max']} vs 학습 {cfg.action.v_max} — 같은 --v-max 로")
    if abs(float(meta["range_max"]) - cfg.lidar.range_max) > 1e-6:
        raise SystemExit(f"[bc-init] range_max 불일치: npz {meta['range_max']} vs 학습 {cfg.lidar.range_max}")
    pol = model.policy
    with torch.no_grad():
        for fe in (pol.actor.features_extractor, pol.critic.features_extractor):
            core = fe.brain.core
            for k in PARAM_KEYS:
                getattr(core, k).copy_(torch.from_numpy(d[f"brain.{k}"]))
            if not (torch.equal(core.mask.cpu(), torch.from_numpy(d["mask"]))
                    and torch.equal(core.sign.cpu(), torch.from_numpy(d["sign"]))):
                raise SystemExit("[bc-init] DN 배선(mask/sign)이 다르다 — n_dn·fan_in·seed 확인")
        pol.actor.mu.weight.copy_(torch.from_numpy(d["mu_w"])); pol.actor.mu.bias.copy_(torch.from_numpy(d["mu_b"]))
        pol.actor.log_std.weight.zero_(); pol.actor.log_std.bias.fill_(log_std)
    pol.critic_target.load_state_dict(pol.critic.state_dict())
    print(f"[bc-init] {path} (모방학습 반복 {meta.get('iter')}, 시험 {meta.get('eval')})", flush=True)


def compile_brains(model) -> None:
    """actor·critic·target critic 의 커넥톰 회로 forward 를 torch.compile. 가중치·저장 형식은 그대로."""
    _ensure_python_headers()
    for net in (model.policy.actor, model.policy.critic, model.policy.critic_target):
        brain = net.features_extractor.brain
        brain.forward = torch.compile(brain.forward, dynamic=False)
    print("[train] torch.compile: 커넥톰 회로 (actor·critic·target)", flush=True)


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
    if args.target_entropy is None:
        args.target_entropy = -3.5 if args.bc_init else -1.0
    cfg = apply_train_cfg(args)
    maps = [m for m in args.maps.split(",") if m]
    eval_maps = [m for m in args.eval_maps.split(",") if m]
    save_dir = Path(args.save_dir or f"runs/lidarfly_{time.strftime('%Y%m%d_%H%M%S')}")
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps({"args": vars(args)}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv

    def _train_env(rank: int):
        factory = make_env(maps, seed=args.seed, cfg=cfg, rank=rank)

        def _init():
            env_i = factory()
            env_i.action_space.seed(args.seed + rank)
            return Monitor(env_i)

        return _init

    env = vec_cls([_train_env(i) for i in range(args.n_envs)])
    seed_lo = int(args.seed)
    seed_hi = seed_lo + int(args.n_envs) - 1
    print(f"[train] 병렬 env {args.n_envs}개, 시드 {seed_lo}..{seed_hi}, random 맵은 그 에피소드가 뽑혔을 때만 재생성")
    kw = policy_kwargs(args, cfg)

    tb_log = str(save_dir / "tb") if _has_tb() else None
    resume = args.resume
    if resume == "auto":
        found = _find_resume(save_dir)
        resume = str(found) if found else None
        print(f"[train] resume auto → {resume or '없음, 새로 시작'}")
    if resume:
        model = SAC.load(resume, env=env, device=device)
        model.learning_starts = model.num_timesteps + min(args.learning_starts, 10_000)
        model.__class__ = _PolicyWarmupSAC
        if tb_log:
            model.tensorboard_log = tb_log
        reset_ts = False
    else:
        model = SAC(AsymSACPolicy, env, learning_rate=args.lr, buffer_size=args.buffer_size,
                    batch_size=args.batch_size, learning_starts=args.learning_starts, gamma=args.gamma,
                    tau=0.005, train_freq=1, gradient_steps=args.gradient_steps,
                    ent_coef=f"auto_{args.ent_init}" if args.bc_init else "auto",
                    target_entropy=args.target_entropy, policy_kwargs=kw, verbose=1,
                    seed=args.seed, device=device, tensorboard_log=tb_log)
        reset_ts = True
        if args.bc_init:
            load_bc_init(model, args.bc_init, cfg, args.bc_log_std)
            model.__class__ = _PolicyWarmupSAC          # 버퍼 채우는 동안에도 무작위 대신 모방한 정책으로 운전
    if args.compile:
        compile_brains(model)
    actor = model.policy.actor
    n_act = sum(p.numel() for p in actor.parameters())
    n_brain = sum(p.numel() for p in actor.features_extractor.parameters())
    print(f"[train] lidarfly (LiDAR + 커넥톰) pi={args.pi_net or '선형'} device={model.device} "
          f"actor_params={n_act:,} (brain {n_brain:,}) maps={maps} eval={eval_maps} gamma={args.gamma:.4f} "
          f"v={cfg.action.v_min:g}~{cfg.action.v_max:g} m/s")

    cb_list = []
    if args.bc_init and not resume:
        from camera.depthfly.train import ActorFreeze
        cb_list.append(ActorFreeze(model.learning_starts + args.actor_freeze_steps))
    cbs = CallbackList(cb_list + [
        LapEvalCallback(eval_maps, cfg, save_dir, args.eval_freq, env_cls=LidarFlyEnv, env_cfg=cfg),
        CheckpointCallback(max(args.ckpt_freq // args.n_envs, 1), str(save_dir / "checkpoints"), name_prefix="sac"),
        TimeLimitCallback(args.time_limit_min, save_dir / "last_model", args.save_every_min),
    ])
    try:
        model.learn(total_timesteps=args.timesteps, callback=cbs, reset_num_timesteps=reset_ts)
    finally:
        if _safe_save(model, save_dir / "last_model.zip"):
            print(f"[train] saved {save_dir / 'last_model.zip'}")


# Windows spawn 에서 SubprocVecEnv 자식이 학습을 다시 시작하지 않게 가드.
# 문제가 나면 --subproc 없이 DummyVecEnv 로 실행한다.
if __name__ == "__main__":
    main()
