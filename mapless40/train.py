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

    점수 = 평균 '라인 기준 전진 속도' [m/s] = 진행 거리 / 에피소드 시간.
    충돌하면 거기서 멈추므로 빠르고 안 죽을수록 높다. 맵 길이가 달라도 비교 가능.
    이어 학습(--resume) 시 best.json 에서 이전 최고 점수를 읽어 덮어쓰기를 막는다.
    """

    def __init__(self, eval_maps, cfg: EnvConfig, save_dir: Path, eval_freq: int,
                 n_spawns: int = 3, seed: int = 1234):
        super().__init__()
        self.env = MaplessRaceEnv40(maps=eval_maps, cfg=cfg, seed=seed,
                                    sensor_noise=True, randomize=False)
        self.maps = [t.name for t in self.env.tracks]
        self.save_dir, self.eval_freq, self.n_spawns = save_dir, eval_freq, n_spawns
        self.best_file = save_dir / "best.json"
        self.best = -np.inf
        if self.best_file.exists():
            self.best = float(json.loads(self.best_file.read_text())["score"])
        self.last_eval = 0
        self.csv = save_dir / "eval.csv"
        if not self.csv.exists():
            with self.csv.open("w", newline="") as f:
                csv.writer(f).writerow(["timesteps", "map", "spawn", "progress", "laps",
                                        "best_lap", "collided", "v_mean", "sim_t", "score_mps"])

    def _run(self, m: str, k: int) -> dict:
        line_len = next(t for t in self.env.tracks if t.name == m).line.length
        obs, _ = self.env.reset(seed=1000 + k, options={
            "map": m, "s0": line_len * k / self.n_spawns, "lat": 0.0, "dyaw": 0.0, "v0": 1.5,
            "n_obstacles": 0 if k == 0 else 2})       # 출발0 = 순수 랩타임, 나머지 = 장애물 2개
        vs = []
        while True:
            a, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = self.env.step(a)
            vs.append(info["speed"])
            if term or trunc:
                break
        return dict(progress=info["progress"], laps=info["laps"],
                    score=info["progress"] * line_len / self.env.cfg.max_episode_s,
                    best_lap=min(info["lap_times"]) if info["lap_times"] else np.nan,
                    collided=info["collided"], v_mean=float(np.mean(vs)), sim_t=info["t"])

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        rows, scores = [], []
        for m in self.maps:
            for k in range(self.n_spawns):
                r = self._run(m, k)
                rows.append([self.num_timesteps, m, k, r["progress"], r["laps"], r["best_lap"],
                             int(r["collided"]), r["v_mean"], r["sim_t"], r["score"]])
                scores.append(r["score"])
                self.logger.record(f"eval/{m}_progress_{k}", r["progress"])
        with self.csv.open("a", newline="") as f:
            csv.writer(f).writerows(rows)
        score = float(np.mean(scores))
        crash = float(np.mean([r[6] for r in rows]))
        laps = [r[5] for r in rows if not np.isnan(r[5])]
        self.logger.record("eval/score_mps", score)
        self.logger.record("eval/crash_rate", crash)
        if laps:
            self.logger.record("eval/best_lap_s", float(np.min(laps)))
        per_map = "  ".join(f"{m}: 진행 {np.mean([r[3] for r in rows if r[1] == m]):.2f}바퀴"
                            for m in self.maps)
        print(f"[eval] t={self.num_timesteps:,} 점수={score:.2f} m/s  충돌률={crash:.2f}  "
              f"최고랩={np.min(laps) if laps else '-'}  | {per_map}", flush=True)
        if score > self.best:
            self.best = score
            self.model.save(str(self.save_dir / "best_model"))
            self.best_file.write_text(json.dumps({"score": score, "timesteps": int(self.num_timesteps)}))
            print(f"[eval] new best → {self.save_dir / 'best_model.zip'}", flush=True)
        return True


class TimeLimitCallback(BaseCallback):
    """지정 시간이 지나면 학습을 멈추고 저장 (코랩 세션 끊기기 전에 안전하게 종료)."""

    def __init__(self, minutes: float, save_path: Path | None = None, save_every_min: float = 10.0):
        super().__init__()
        self.deadline = time.time() + minutes * 60 if minutes > 0 else None
        self._t, self._n = time.time(), None
        # 코랩 런타임이 끊기면 finally 의 last_model 저장이 안 돈다 → 주기적으로 덮어쓴다
        self.save_path, self.save_every = save_path, save_every_min * 60
        self._t_save = time.time()
        # 시간 분해: rollout(env 스텝 + 행동 추론) vs 그 사이(= SAC 업데이트)
        self._t_roll = self._t_end = None
        self._acc_env = self._acc_upd = 0.0

    def _on_rollout_start(self) -> None:
        now = time.time()
        if self._t_end is not None:
            self._acc_upd += now - self._t_end
        self._t_roll = now

    def _on_rollout_end(self) -> None:
        now = time.time()
        if self._t_roll is not None:
            self._acc_env += now - self._t_roll
        self._t_end = now

    def _on_step(self) -> bool:
        now = time.time()
        if self._n is None:
            self._t, self._n = now, self.num_timesteps
        elif now - self._t >= 60.0:          # SB3 fps 는 누적 평균이라 지금 속도를 따로 찍는다
            rate = (self.num_timesteps - self._n) / (now - self._t)
            left = getattr(self.model, "_total_timesteps", 0) - self.num_timesteps
            eta = f", 남은 {left / max(rate, 1e-6) / 3600:.1f} h" if left > 0 else ""
            phase = "학습 중" if self.num_timesteps > self.model.learning_starts else "데이터 모으는 중(업데이트 전)"
            tot = max(self._acc_env + self._acc_upd, 1e-9)
            print(f"[speed] 최근 1분 {rate:.0f} steps/s [{phase}, {self.model.device}] "
                  f"env {100 * self._acc_env / tot:.0f}% / 업데이트 {100 * self._acc_upd / tot:.0f}% "
                  f"(step {self.num_timesteps:,}{eta})", flush=True)
            self._acc_env = self._acc_upd = 0.0
            self._t, self._n = now, self.num_timesteps
        if self.save_path is not None and now - self._t_save >= self.save_every:
            self.model.save(str(self.save_path))
            self._t_save = now
            print(f"[save] step {self.num_timesteps:,} → {self.save_path}.zip", flush=True)
        if self.deadline and now > self.deadline:
            print("[train] 시간 제한 도달 → 저장 후 종료", flush=True)
            return False
        return True


class _PolicyWarmupSAC(SAC):
    """이어 학습용: learning_starts 전(버퍼 다시 모으는 구간)에도 무작위 대신 현재 정책으로 행동."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)


def _find_resume(save_dir: Path) -> Path | None:
    """save_dir 안에서 가장 최근 모델(last_model 또는 최신 체크포인트)을 찾는다."""
    cands = list((save_dir / "checkpoints").glob("*.zip")) + [save_dir / "last_model.zip"]
    cands = [c for c in cands if c.exists()]
    return max(cands, key=lambda c: c.stat().st_mtime) if cands else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default="Spielberg,Silverstone,Monza,Catalunya")
    p.add_argument("--eval-maps", default="Budapest")
    p.add_argument("--encoder", choices=["conv1d", "bev", "both"], default="conv1d")
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
    p.add_argument("--ckpt-freq", type=int, default=100_000)
    p.add_argument("--save-every-min", type=float, default=10.0, help="last_model 을 이 간격(분)마다 덮어씀")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None, help="이어 학습할 zip, 또는 'auto' (save-dir 안 최신 모델)")
    p.add_argument("--time-limit-min", type=float, default=0, help="이 시간(분) 뒤 저장하고 종료 (0=제한 없음)")
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

    resume = args.resume
    if resume == "auto":
        found = _find_resume(save_dir)
        resume = str(found) if found else None
        print(f"[train] resume auto → {resume or '없음, 새로 시작'}")
    if resume:
        model = SAC.load(resume, env=env, device=device)
        # 리플레이 버퍼는 저장하지 않으므로 1만 스텝을 다시 모은 뒤 학습 재개.
        # SB3 기본은 이 구간을 **무작위 행동**으로 채운다 → 버퍼가 충돌 데이터로 가득 차고,
        # 그걸로 바로 업데이트하면 이미 배운 정책이 무너진다 (ep_len 837 → 114 관찰).
        # → 이어 학습 때는 불러온 정책(확률적 SAC 행동)으로 모은다.
        model.learning_starts = model.num_timesteps + min(args.learning_starts, 10_000)
        # 인스턴스에 함수를 붙이면 model.save 가 그걸 pickle 하다 실패한다 → 클래스만 바꾼다
        model.__class__ = _PolicyWarmupSAC
        print(f"[train] 이어 학습: step {model.num_timesteps:,} 부터, 정책 행동으로 1만 스텝 모은 뒤 업데이트 재개")
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
    real_dev = str(model.device)
    if device == "cuda" and not real_dev.startswith("cuda"):
        print("!" * 70 + "\n[train] 경고: GPU 없음 → CPU 로 학습 중 (매우 느림). 코랩 GPU 할당량 소진 가능성.\n"
              "        런타임 → 런타임 유형 변경 → T4 GPU 확인, 안 되면 할당량이 풀린 뒤 다시.\n" + "!" * 70, flush=True)
    gpu = torch.cuda.get_device_name(0) if real_dev.startswith("cuda") else "-"
    print(f"[train] encoder={args.encoder} maps={maps} eval={eval_maps} device={real_dev} ({gpu}) "
          f"gamma={args.gamma:.4f} actor_params={n_act:,} critic_params={n_crit:,}")

    cbs = CallbackList([
        LapEvalCallback(eval_maps, cfg, save_dir, args.eval_freq),
        CheckpointCallback(max(args.ckpt_freq // args.n_envs, 1), str(save_dir / "checkpoints"),
                           name_prefix="sac"),
        TimeLimitCallback(args.time_limit_min, save_dir / "last_model", args.save_every_min),
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
