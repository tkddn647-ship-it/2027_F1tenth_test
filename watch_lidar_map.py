"""
watch_lidar_map.py
==================
실제 레이싱 맵 occupancy 위에서 LiDAR 주행을 시각화.

  python watch_lidar_map.py --map ajou
  python watch_lidar_map.py --map ajou --model connectome_ppo_lidar_ajou.zip --use-cache
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from lidar_race_env import LidarRaceEnv


def render_frame(env: LidarRaceEnv, xs, ys, title: str) -> np.ndarray:
    """전체 맵을 프레임에 맞춤 (예전 pad=20m → 흰 여백만 커지던 버그 수정)."""
    ox, oy = float(env.origin[0]), float(env.origin[1])
    map_w = env.w * env.resolution
    map_h = env.h * env.resolution
    extent = [ox, ox + map_w, oy, oy + map_h]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    # origin=upper + extent: 픽셀(0,0)=맵 상단=높은 y
    ax.imshow(env.occ, cmap="gray_r", origin="upper", extent=extent, interpolation="nearest")

    scans = env._cast_lidar()
    half = env.FOV / 2.0
    angles = env.theta + np.linspace(-half, half, env.N_RAYS)
    for d, a in zip(scans, angles):
        ax.plot(
            [env.x, env.x + d * np.cos(a)],
            [env.y, env.y + d * np.sin(a)],
            color="#1af", alpha=0.55, lw=0.9,
        )

    if len(xs) > 1:
        ax.plot(xs, ys, color="#e22", lw=1.6, alpha=0.9)
    ax.scatter(env.x, env.y, c="#f80", s=50, zorder=5, edgecolors="k", lw=0.4)
    arrow_len = max(0.6, 0.04 * max(map_w, map_h))
    ax.arrow(
        env.x, env.y,
        arrow_len * np.cos(env.theta), arrow_len * np.sin(env.theta),
        head_width=arrow_len * 0.45, color="#fa0",
        length_includes_head=True, zorder=6,
    )

    margin = 0.4
    ax.set_xlim(ox - margin, ox + map_w + margin)
    ax.set_ylim(oy - margin, oy + map_h + margin)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def wall_follow(obs, n_rays: int, hist_len: int = 4) -> np.ndarray:
    """최신 LiDAR 프레임(히스토리 마지막)으로 조향·속도."""
    # obs = [lidar * hist_len | v | yaw]
    off = n_rays * (hist_len - 1)
    rays = obs[off: off + n_rays]
    left, mid, right = rays[:7].mean(), rays[7:13].mean(), rays[13:].mean()
    steer = float(np.clip((right - left) * 2.2, -1, 1))
    if mid < 0.22:
        steer = float(np.sign(right - left + 1e-6))
    if mid > 0.45:
        speed_u = 0.7
    elif mid > 0.28:
        speed_u = 0.35
    else:
        speed_u = 0.0
    return np.array([steer, speed_u], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="ajou",
                        help="ajou(실차카토) | Spielberg | vegas | yaml경로")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="watch_out_race")
    parser.add_argument("--max-sec", type=float, default=40.0,
                        help="최대 시뮬 시간 [s] (기본 40)")
    args = parser.parse_args()

    env = LidarRaceEnv(args.map, seed=args.seed)
    model = None
    if args.model:
        from stable_baselines3 import PPO
        custom = None
        if not args.baseline:
            from connectome_loader import (
                load_hemibrain_cache, build_synthetic_hemibrain,
                select_speed_relevant_subcircuit, identify_io_neurons, build_adjacency,
            )
            from watch_agent import make_extractor_class
            if args.use_cache:
                n, c = load_hemibrain_cache()
            else:
                n, c = build_synthetic_hemibrain(800, seed=0)
            n, c = select_speed_relevant_subcircuit(n, c)
            n = n.reset_index(drop=True)
            inp, out = identify_io_neurons(n)
            A = build_adjacency(n, c)
            custom = {"ConnectomeFeaturesExtractor": make_extractor_class(A, inp, out)}
        model = PPO.load(args.model, env=env, custom_objects=custom, device="cpu")

    obs, _ = env.reset(seed=args.seed)
    xs, ys = [env.x], [env.y]
    frames = []
    info = {}
    max_steps = min(env.MAX_STEPS, int(args.max_sec / env.DT))
    # 녹화: 0.1s 간격 (= 2 step), GIF duration=100ms → 거의 실시간
    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = wall_follow(obs, env.N_RAYS, env.HIST_LEN)
        obs, r, term, trunc, info = env.step(action)
        xs.append(env.x)
        ys.append(env.y)
        if step % 2 == 0 or term or trunc:
            st = "LAP" if info.get("lap_completed") else ("CRASH" if info.get("collided") else "RUN")
            t_sec = (step + 1) * env.DT
            title = (
                f"{args.map}  {st}  t={t_sec:.1f}s  "
                f"v={env.v:.1f}m/s  steer={np.degrees(env.last_steer):.0f}deg  "
                f"dist={info.get('dist_m', 0):.1f}m"
            )
            frames.append(Image.fromarray(render_frame(env, xs, ys, title)))
        if term or trunc:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "model" if args.model else "lidar"
    gif = out / f"lidar_{args.map}_{tag}.gif"
    if len(frames) > 500:
        frames = frames[:: max(1, len(frames) // 500)]
    if frames:
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    t_end = (step + 1) * env.DT if frames else 0.0
    print(
        f"[watch] {args.map} t={t_end:.1f}s "
        f"crash={info.get('collided')} dist={info.get('dist_m', 0):.1f}m "
        f"v_end={env.v:.1f} -> {gif.resolve()}"
    )
    try:
        import os
        os.startfile(str(gif.resolve()))
    except Exception:
        pass


if __name__ == "__main__":
    main()
