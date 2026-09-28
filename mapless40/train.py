"""
mapless40.train
===============
40 Hz mapless asymmetric SAC 학습.

예)
  python -m mapless40.train --maps Spielberg,Silverstone,Monza --eval-maps Budapest \
      --encoder conv1d --timesteps 2000000 --n-envs 8 --subproc --device cuda

  python -m mapless40.train --encoder bev ...      # BEV 2D CNN 비교 실험

산출물: runs/mapless40_<encoder>_<시각>/
  config.json, best_model.zip, last_model.zip, checkpoints/, eval.csv, actor.ts.pt(export)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .config import GAMMA_40HZ, EnvConfig
from .env import MaplessRaceEnv40, make_env
from .policy import AsymFeatures, AsymSACPolicy


def _has_tb() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


class LapEvalCallback(BaseCallback):
    """결정적 정책으로 고정 스폰에서 주행 → 진행률(랩 단위)·랩타임·충돌률 기록.

    점수 = 평균 진행률(eval 시간 안에 몇 바퀴 갔나). 빠르고 안 죽을수록 높다.
    """

    def __init__(self, eval_maps, cfg: EnvConfig, save_dir: Path, eval_freq: int,
                 n_spawns: int = 3, seed: int = 1234):
        super().__init__()
        self.env = MaplessRaceEnv40(maps=eval_maps, cfg=cfg, seed=seed,
                                    sensor_noise=True, randomize=False)
        self.maps = [t.name for t in self.env.tracks]
        self.save_dir, self.eval_freq, self.n_spawns = save_dir, eval_freq, n_spawns
        self.best = -np.inf
        self.last_eval = 0
        self.csv = save_dir / "eval.csv"
        with self.csv.open("w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "map", "spawn", "progress", "laps",
                                    "best_lap", "collided", "v_mean", "sim_t"])

    def _run(self, m: str, k: int) -> dict:
        line_len = next(t for t in self.env.tracks if t.name == m).line.length
        obs, _ = self.env.reset(seed=1000 + k, options={
            "map": m, "s0": line_len * k / self.n_spawns, "lat": 0.0, "dyaw": 0.0, "v0": 1.5})
        vs = []
        while True:
            a, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = self.env.step(a)
            vs.append(info["speed"])
            if term or trunc:
                break
        return dict(progress=info["progress"], laps=info["laps"],
                    best_lap=min(info["lap_times"]) if info["lap_times"] else np.nan,
                    collided=info["collided"], v_mean=float(np.mean(vs)), sim_t=info["t"])

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        rows, prog = [], []
        for m in self.maps:
            for k in range(self.n_spawns):
                r = self._run(m, k)
                rows.append([self.num_timesteps, m, k, r["progress"], r["laps"], r["best_lap"],
                             int(r["collided"]), r["v_mean"], r["sim_t"]])
                prog.append(r["progress"])
                self.logger.record(f"eval/{m}_progress_{k}", r["progress"])
        with self.csv.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        score = float(np.mean(prog))
        crash = float(np.mean([r[6] for r in rows]))
        laps = [r[5] for r in rows if not np.isnan(r[5])]
        self.logger.record("eval/progress_mean", score)
        self.logger.record("eval/crash_rate", crash)
        if laps:
            self.logger.record("eval/best_lap_s", float(np.min(laps)))
        print(f"[eval] t={self.num_timesteps} progress={score:.3f} crash={crash:.2f} "
              f"best_lap={np.min(laps) if laps else '-'}")
        if score > self.best:
            self.best = score
            self.model.save(str(self.save_dir / "best_model"))
            print(f"[eval] new best → {self.save_dir / 'best_model.zip'}")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default="Spielberg,Silverstone,Monza,Catalunya")
    p.add_argument("--eval-maps", default="Budapest")
    p.add_argument("--encoder", choices=["conv1d", "bev"], default="conv1d")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--subproc", action="store_true", help="env 를 프로세스로 병렬 실행")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer-size", type=int, default=200_000,
                   help="전이당 약 18 KB (scan float16 ×2) → 20만 ≈ 3.6 GB RAM")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--gradient-steps", type=int, default=2,
                   help="vec step 마다 gradient step 수 (n_envs=8 이면 UTD 0.25)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=GAMMA_40HZ)
    p.add_argument("--net", default="256,256")
    p.add_argument("--max-episode-s", type=float, default=60.0)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--ckpt-freq", type=int, default=200_000)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None, help="이어 학습할 zip")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = EnvConfig(max_episode_s=args.max_episode_s)
    maps = [m for m in args.maps.split(",") if m]
    eval_maps = [m for m in args.eval_maps.split(",") if m]
    save_dir = Path(args.save_dir or f"runs/mapless40_{args.encoder}_{time.strftime('%Y%m%d_%H%M%S')}")
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(
        {"args": vars(args), "env": cfg.to_dict()}, indent=2, ensure_ascii=False), encoding="utf-8")

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(make_env(maps, seed=args.seed, cfg=cfg), n_envs=args.n_envs,
                       seed=args.seed, vec_env_cls=vec_cls)

    net = [int(x) for x in args.net.split(",")]
    policy_kwargs = dict(
        features_extractor_class=AsymFeatures,
        features_extractor_kwargs=dict(
            encoder=args.encoder,
            lidar_cfg=cfg.to_dict()["lidar"],
            norm_cfg=cfg.to_dict()["norm"],
        ),
        net_arch=dict(pi=net, qf=net),
    )

    if args.resume:
        model = SAC.load(args.resume, env=env, device=device)
        reset_ts = False
    else:
        model = SAC(
            AsymSACPolicy, env,
            learning_rate=args.lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
            learning_starts=args.learning_starts, gamma=args.gamma, tau=0.005,
            train_freq=1, gradient_steps=args.gradient_steps,
            ent_coef="auto", target_entropy=-2.0,
            policy_kwargs=policy_kwargs, verbose=1, seed=args.seed, device=device,
            tensorboard_log=str(save_dir / "tb") if _has_tb() else None,
        )
        reset_ts = True

    n_act = sum(p.numel() for p in model.policy.actor.parameters())
    n_crit = sum(p.numel() for p in model.policy.critic.parameters())
    print(f"[train] encoder={args.encoder} maps={maps} eval={eval_maps} device={device} "
          f"gamma={args.gamma:.4f} actor_params={n_act:,} critic_params={n_crit:,}")

    cbs = CallbackList([
        LapEvalCallback(eval_maps, cfg, save_dir, args.eval_freq),
        CheckpointCallback(max(args.ckpt_freq // args.n_envs, 1), str(save_dir / "checkpoints"),
                           name_prefix="sac"),
    ])
    try:
        model.learn(total_timesteps=args.timesteps, callback=cbs, reset_num_timesteps=reset_ts)
    finally:
        model.save(str(save_dir / "last_model"))
        print(f"[train] saved {save_dir / 'last_model.zip'}")

    best = save_dir / "best_model.zip"
    try:
        from .export import export_actor
        export_actor(best if best.exists() else save_dir / "last_model.zip", save_dir)
    except Exception as e:  # pragma: no cover
        print(f"[train] export 실패 (수동으로 python -m mapless40.export 실행): {e}")


if __name__ == "__main__":
    main()
