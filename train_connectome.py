"""
train_connectome.py
=====================
레이싱 목표: MultiTrackEnv에서 랩타임을 줄이는 정책 학습.

- 기본: 초파리 속도관련 서브서킷(ConnectomeRNN) + PPO
- --baseline: 동일 환경에서 표준 MLP 비교
- --use-cache: hemibrain_cache 사용 (없으면 합성 T4/T5→CX→DN)

예:
  python train_connectome.py --use-cache --timesteps 200000
  python train_connectome.py --baseline --timesteps 200000
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np

from connectome_loader import (
    DEFAULT_CACHE_DIR,
    build_synthetic_hemibrain,
    load_hemibrain_real,
    load_hemibrain_cache,
    identify_io_neurons,
    select_speed_relevant_subcircuit,
    build_adjacency,
    resolve_token,
    save_hemibrain_cache,
)
from multitrack_env import MultiTrackEnv
from lidar_race_env import LidarRaceEnv


def build_brain(n_neurons: int, real_token: str | None, use_cache: bool,
                cache_dir: str, seed: int, use_subcircuit: bool,
                refresh_cache: bool):
    if use_cache and not refresh_cache:
        print(f"[train] 캐시 로드: {cache_dir}")
        neurons_df, conn_df = load_hemibrain_cache(cache_dir)
    elif real_token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS") or refresh_cache:
        token = resolve_token(real_token)
        print("[train] hemibrain 다운로드...")
        neurons_df, conn_df = load_hemibrain_real(token)
        save_hemibrain_cache(neurons_df, conn_df, cache_dir=cache_dir)
    else:
        print(f"[train] 합성 레이싱 커넥톰 n={n_neurons}")
        neurons_df, conn_df = build_synthetic_hemibrain(n_neurons=n_neurons, seed=seed)

    if use_subcircuit:
        neurons_df, conn_df = select_speed_relevant_subcircuit(neurons_df, conn_df)
        neurons_df = neurons_df.reset_index(drop=True)

    input_idx, output_idx = identify_io_neurons(neurons_df)
    if len(input_idx) == 0 or len(output_idx) == 0:
        raise SystemExit(
            f"I/O 뉴런 없음 (in={len(input_idx)}, out={len(output_idx)}). "
            "타입 prefix / 서브서킷 필터를 확인하세요."
        )

    A_signed = build_adjacency(neurons_df, conn_df)
    print(f"[train] 뉴런 {len(neurons_df)} | in {len(input_idx)} | out {len(output_idx)} | "
          f"밀도 {(A_signed != 0).mean():.4%}")
    return A_signed, input_idx, output_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-neurons", type=int, default=800, help="합성 커넥톰 크기")
    parser.add_argument("--real-token", type=str, default=None)
    parser.add_argument("--use-cache", action="store_true",
                        help="hemibrain_cache 사용")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-subcircuit", action="store_true",
                        help="서브서킷 필터 끄기")
    parser.add_argument("--baseline", action="store_true",
                        help="표준 MLP (랩타임 비교용)")
    parser.add_argument("--use-f1tenth-gym", action="store_true")
    parser.add_argument("--env", type=str, default="lidar",
                        choices=["multitrack", "lidar"],
                        help="기본 lidar=실차맵 occupancy+LiDAR (권장)")
    parser.add_argument("--map", type=str, default="ajou",
                        help="lidar 맵: ajou(Roboracer Cartographer) | Spielberg | vegas | yaml경로")
    parser.add_argument("--tracks", type=str, default="oval,wavy_loop,asymmetric_oval",
                        help="쉼표구분 트랙명 (procedural 또는 Spielberg 등)")
    parser.add_argument("--racetracks-dir", type=str, default=None,
                        help="multitrack + 실제 F1 centerline 풀 사용 시")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습할 PPO zip (명시할 때만 resume)")
    parser.add_argument("--fresh", action="store_true",
                        help="기존 zip 무시하고 처음부터 학습")
    parser.add_argument("--unlock-after-laps", type=int, default=0,
                        help="누적 완주 N회 후 spawn=random. 0=비활성(직진 스폰 유지, 기본)")
    args = parser.parse_args()

    try:
        import torch.nn as nn
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as e:
        raise SystemExit(
            "pip install torch gymnasium stable-baselines3\n"
            f"(원본: {e})"
        )

    track_names = [t.strip() for t in args.tracks.split(",") if t.strip()]

    def _env_fn():
        if args.use_f1tenth_gym:
            from f1tenth_gym_wrapper import F1TenthGymMultiMapEnv
            return F1TenthGymMultiMapEnv(
                args.racetracks_dir or "f1tenth_racetracks", seed=args.seed
            )
        if args.env == "lidar":
            return LidarRaceEnv(map_name=args.map, seed=args.seed)
        return MultiTrackEnv(
            track_names=track_names,
            seed=args.seed,
            racetracks_dir=args.racetracks_dir,
        )

    env = make_vec_env(_env_fn, n_envs=args.n_envs, seed=args.seed)

    class LapTimeLogger(BaseCallback):
        """완주 로그 + 커리큘럼: N회 완주 후 전 env spawn_mode=random."""

        def __init__(self, unlock_after_laps: int = 0):
            super().__init__()
            self.lap_times_by_track: dict[str, list[float]] = {}
            self.unlock_after_laps = int(unlock_after_laps)
            self._unlocked = False

        def _set_all_spawn(self, mode: str) -> None:
            venv = self.training_env
            envs = getattr(venv, "envs", None)
            if envs is None:
                return
            for e in envs:
                base = e
                while hasattr(base, "env"):
                    base = base.env
                if hasattr(base, "set_spawn_mode"):
                    base.set_spawn_mode(mode)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if info.get("lap_completed") and info.get("lap_time") is not None:
                    track = info.get("track_name", "unknown")
                    self.lap_times_by_track.setdefault(track, []).append(float(info["lap_time"]))
            total = sum(len(v) for v in self.lap_times_by_track.values())
            if (
                self.unlock_after_laps > 0
                and (not self._unlocked)
                and total >= self.unlock_after_laps
            ):
                self._unlocked = True
                self._set_all_spawn("random")
                print(
                    f"[curriculum] 누적 완주 {total}회 ≥ {self.unlock_after_laps} "
                    f"→ spawn_mode=random"
                )
            if total > 0 and total % 20 == 0:
                print(f"[LapTimeLogger] 누적 완주 {total}회")
                for track, times in self.lap_times_by_track.items():
                    recent = times[-10:]
                    print(f"    {track:16s}: n={len(times):4d}, "
                          f"평균={np.mean(recent):.2f}s, 최고={min(recent):.2f}s")
            return True

    if args.baseline:
        print("[train] BASELINE MLP")
        policy_kwargs = dict(net_arch=[64, 64])
        if args.env == "lidar":
            save_path = args.save_path or f"baseline_mlp_lidar_{args.map}.zip"
        else:
            save_path = args.save_path or "baseline_mlp_racetrack.zip"
        custom_objects = None
    else:
        from connectome_rnn import ConnectomeRNN

        A_signed, input_idx, output_idx = build_brain(
            args.n_neurons, args.real_token, args.use_cache, args.cache_dir,
            args.seed, use_subcircuit=not args.no_subcircuit,
            refresh_cache=args.refresh_cache,
        )

        class ConnectomeFeaturesExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, n_inner_steps: int = 2):
                super().__init__(observation_space, features_dim=len(output_idx))
                self.brain = ConnectomeRNN(
                    A_signed, input_idx, output_idx,
                    n_obs=observation_space.shape[0], n_act=len(output_idx),
                    dt=1.0, tau=3.0,
                )
                self.brain.W_out = nn.Identity()
                self._features_dim = len(output_idx)
                self.n_inner_steps = n_inner_steps

            def forward(self, observations):
                y, _h = self.brain(observations, n_steps=self.n_inner_steps)
                return y

        policy_kwargs = dict(
            features_extractor_class=ConnectomeFeaturesExtractor,
            features_extractor_kwargs=dict(n_inner_steps=2),
            net_arch=[],
        )
        custom_objects = {"ConnectomeFeaturesExtractor": ConnectomeFeaturesExtractor}
        tag = "cache" if args.use_cache else "synth"
        if args.env == "lidar":
            save_path = args.save_path or f"connectome_ppo_lidar_{args.map}.zip"
        else:
            save_path = args.save_path or f"connectome_ppo_racetrack_{tag}.zip"

    resume_path = None if args.fresh else args.resume

    if resume_path and Path(resume_path).exists():
        print(f"[train] resume: {resume_path} (+{args.timesteps} steps)")
        model = PPO.load(
            resume_path,
            env=env,
            custom_objects=custom_objects,
            device="cpu",
        )
        model.set_env(env)
        reset_ts = False
    else:
        if args.fresh and Path(save_path).exists():
            bak = Path(save_path).with_suffix(".bak.zip")
            Path(save_path).replace(bak)
            print(f"[train] fresh: moved old weights -> {bak.name}")
        print("[train] fresh start (no resume)")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            n_steps=512,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.005,
        )
        reset_ts = True

    print(f"[train] timesteps={args.timesteps}  tracks={track_names}")
    print(f"[train] params={sum(p.numel() for p in model.policy.parameters()):,}")
    logger = LapTimeLogger(unlock_after_laps=args.unlock_after_laps)
    model.learn(total_timesteps=args.timesteps, callback=logger, reset_num_timesteps=reset_ts)
    model.save(save_path)
    print(f"[train] 저장: {save_path}")
    if logger.lap_times_by_track:
        print("[train] 최종 트랙별 랩타임 요약:")
        for track, times in logger.lap_times_by_track.items():
            print(f"  {track:16s}: n={len(times)}, 평균={np.mean(times):.2f}s, "
                  f"최고={min(times):.2f}s")
    print(
        "비교:\n"
        "  python train_connectome.py --use-cache --timesteps 200000\n"
        "  python train_connectome.py --baseline --timesteps 200000"
    )


if __name__ == "__main__":
    main()
