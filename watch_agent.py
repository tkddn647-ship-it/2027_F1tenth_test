"""
watch_agent.py
================
학습된 connectome PPO를 toy 환경에서 돌려보고,
주행 GIF + 궤적 PNG로 저장한 뒤 기본 뷰어로 연다.

  python watch_agent.py --model connectome_ppo_real.zip --use-cache
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow

from connectome_loader import (
    DEFAULT_CACHE_DIR, load_hemibrain_cache, build_synthetic_hemibrain,
    identify_io_neurons, build_adjacency,
)
from toy_obstacle_env import ToyObstacleEnv


def load_brain(use_cache: bool, cache_dir: str, n_neurons: int, seed: int):
    if use_cache:
        neurons_df, conn_df = load_hemibrain_cache(cache_dir)
    else:
        neurons_df, conn_df = build_synthetic_hemibrain(n_neurons=n_neurons, seed=seed)
    input_idx, output_idx = identify_io_neurons(neurons_df)
    A_signed = build_adjacency(neurons_df, conn_df)
    return A_signed, input_idx, output_idx


def make_extractor_class(A_signed, input_idx, output_idx):
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from connectome_rnn import ConnectomeRNN

    class ConnectomeFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim: int = 32, n_inner_steps: int = 3):
            super().__init__(observation_space, features_dim=len(output_idx))
            n_obs = observation_space.shape[0]
            self.brain = ConnectomeRNN(
                A_signed, input_idx, output_idx,
                n_obs=n_obs, n_act=len(output_idx), dt=1.0, tau=5.0,
            )
            self.brain.W_out = nn.Identity()
            self._features_dim = len(output_idx)
            self.n_inner_steps = n_inner_steps

        def forward(self, observations):
            y, _h = self.brain(observations, n_steps=self.n_inner_steps)
            return y

    return ConnectomeFeaturesExtractor


def draw_frame(env: ToyObstacleEnv, ax, step: int, reward_sum: float, collided: bool):
    ax.clear()
    ax.set_aspect("equal")
    R = env.ARENA_RADIUS
    ax.set_xlim(-R - 0.5, R + 0.5)
    ax.set_ylim(-R - 0.5, R + 0.5)
    ax.add_patch(Circle((0, 0), R, fill=False, lw=2, color="#222"))
    for ox, oy in env.obstacles:
        ax.add_patch(Circle((ox, oy), env.OBSTACLE_RADIUS, color="#c44", alpha=0.85))

    x, y = env.pos
    hx, hy = np.cos(env.heading), np.sin(env.heading)
    ax.add_patch(Circle((x, y), 0.25, color="#1a7" if not collided else "#333"))
    ax.add_patch(FancyArrow(
        x, y, hx * 0.55, hy * 0.55,
        width=0.08, head_width=0.28, head_length=0.22,
        color="#0a5", length_includes_head=True,
    ))

    # LiDAR rays
    obs = env._get_obs() * env.RAY_RANGE
    angles = env.heading + np.linspace(-np.pi / 2, np.pi / 2, env.N_RAYS)
    for d, a in zip(obs, angles):
        ax.plot(
            [x, x + np.cos(a) * d],
            [y, y + np.sin(a) * d],
            color="#4af", alpha=0.35, lw=0.8,
        )

    status = "CRASH" if collided else "OK"
    ax.set_title(f"hemibrain agent  |  step {step}  |  reward {reward_sum:.1f}  |  {status}")
    ax.set_xticks([])
    ax.set_yticks([])


def run_episode(model, env, seed: int):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    positions = [env.pos.copy()]
    frames_meta = []
    for t in range(env.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        positions.append(env.pos.copy())
        frames_meta.append((env.pos.copy(), env.heading, env.obstacles.copy(),
                            env.speed, bool(info["collided"]), total, t + 1))
        if terminated or truncated:
            break
    return total, positions, frames_meta, bool(info["collided"])


def save_gif(env_template: ToyObstacleEnv, frames_meta, out_path: Path, fps: int = 12):
    from PIL import Image
    import io

    images = []
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    for pos, heading, obstacles, speed, collided, total, step in frames_meta:
        # 임시로 env 상태 주입해서 그리기
        env_template.pos = pos
        env_template.heading = heading
        env_template.obstacles = obstacles
        env_template.speed = speed
        draw_frame(env_template, ax, step, total, collided)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB").copy())
        buf.close()
    plt.close(fig)

    if not images:
        raise SystemExit("프레임이 비었습니다.")
    images[0].save(
        out_path, save_all=True, append_images=images[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"[watch] GIF 저장: {out_path.resolve()}")


def save_trajectory(positions, obstacles, collided, out_path: Path, arena_r: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), arena_r, fill=False, lw=2, color="#222"))
    for ox, oy in obstacles:
        ax.add_patch(Circle((ox, oy), 0.4, color="#c44", alpha=0.8))
    pts = np.array(positions)
    ax.plot(pts[:, 0], pts[:, 1], color="#1a7", lw=2)
    ax.scatter(pts[0, 0], pts[0, 1], c="blue", s=60, zorder=5, label="start")
    ax.scatter(pts[-1, 0], pts[-1, 1], c="black" if collided else "orange",
               s=60, zorder=5, label="end")
    ax.legend(loc="upper right")
    ax.set_title("Trajectory")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[watch] 궤적 저장: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="connectome_ppo_real.zip")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--n-neurons", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="watch_out")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model_path = Path(args.model)
    if not model_path.exists():
        # 합성 학습본 fallback
        alt = Path("connectome_ppo.zip")
        if alt.exists():
            print(f"[watch] {model_path} 없음 → {alt} 사용")
            model_path = alt
        else:
            raise SystemExit(f"모델 파일 없음: {args.model}")

    A_signed, input_idx, output_idx = load_brain(
        args.use_cache, args.cache_dir, args.n_neurons, seed=0,
    )
    Extractor = make_extractor_class(A_signed, input_idx, output_idx)

    env = ToyObstacleEnv(seed=args.seed)
    print(f"[watch] 모델 로드: {model_path}")
    model = PPO.load(
        str(model_path),
        env=env,
        custom_objects={"ConnectomeFeaturesExtractor": Extractor},
        device="cpu",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 여러 에피소드 점수
    rewards, lens, crashes = [], [], 0
    for i in range(args.n_eval):
        total, positions, frames, collided = run_episode(model, env, seed=args.seed + i)
        rewards.append(total)
        lens.append(len(positions) - 1)
        crashes += int(collided)
        print(f"  ep{i}: reward={total:.2f}, steps={lens[-1]}, crash={collided}")

    print(f"[watch] 평균 보상 {np.mean(rewards):.2f} ± {np.std(rewards):.2f}  |  "
          f"평균 길이 {np.mean(lens):.1f}  |  충돌 {crashes}/{args.n_eval}")

    # 보기 좋은 시드로 GIF 1개
    total, positions, frames, collided = run_episode(model, env, seed=args.seed)
    gif_path = out_dir / "drive.gif"
    traj_path = out_dir / "trajectory.png"
    save_gif(env, frames, gif_path)
    save_trajectory(positions, frames[0][2], collided, traj_path, env.ARENA_RADIUS)

    # Windows에서 기본 앱으로 열기
    try:
        import os
        os.startfile(str(gif_path.resolve()))
        os.startfile(str(traj_path.resolve()))
    except Exception as e:
        print(f"[watch] 자동 열기 실패 (파일은 저장됨): {e}")


if __name__ == "__main__":
    main()
