"""
watch_sac_f1tenth.py
====================
  python watch_sac_f1tenth.py --model connectome_sac_f1tenth_Spielberg.zip --map Spielberg --use-cache
  python watch_sac_f1tenth.py --model plain_sac_f1tenth_Spielberg_best.zip --map Spielberg --plain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3 import SAC

from connectome_loader import (
    build_adjacency,
    identify_io_neurons,
    load_hemibrain_cache,
    select_speed_relevant_subcircuit,
)
from f1tenth_mapless_env import F1TenthMaplessEnv, N_BEAMS
from policy_sac_connectome import ConnectomeTemporalFeatures


def render_frame(env: F1TenthMaplessEnv, xs, ys, title: str) -> np.ndarray:
    ox, oy = float(env.origin[0]), float(env.origin[1])
    map_w = env.w * env.resolution
    map_h = env.h * env.resolution
    extent = [ox, ox + map_w, oy, oy + map_h]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(env.occ, cmap="gray_r", origin="upper", extent=extent, interpolation="nearest")
    if env.centerline is not None:
        cl = env.centerline
        ax.plot(cl[:, 0], cl[:, 1], color="#888", lw=6, alpha=0.35)

    scans = env._cast_lidar()
    half = 4.71238898 / 2.0
    angles = env.theta + np.linspace(-half, half, N_BEAMS)
    step = max(1, N_BEAMS // 40)
    for d, a in zip(scans[::step], angles[::step]):
        ax.plot(
            [env.x, env.x + d * np.cos(a)],
            [env.y, env.y + d * np.sin(a)],
            color="#1af", alpha=0.35, lw=0.6,
        )
    if len(xs) > 1:
        ax.plot(xs, ys, color="#e22", lw=1.5)
    ax.scatter(env.x, env.y, c="#f80", s=40, zorder=5)
    ax.arrow(
        env.x, env.y,
        0.8 * np.cos(env.theta), 0.8 * np.sin(env.theta),
        head_width=0.3, color="#f80", length_includes_head=True,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--plain", action="store_true", help="MLP SAC (no connectome)")
    parser.add_argument("--cache-dir", type=str, default="hemibrain_cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sec", type=float, default=180.0)
    parser.add_argument("--min-speed", type=float, default=2.0)
    parser.add_argument("--max-speed", type=float, default=7.0)
    parser.add_argument("--max-steer", type=float, default=0.30)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="watch_out_race")
    args = parser.parse_args()

    env = F1TenthMaplessEnv(
        args.map,
        seed=args.seed,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        max_steer=args.max_steer,
    )
    custom = None
    if args.use_cache and not args.plain:
        n, c = load_hemibrain_cache(args.cache_dir)
        n, c = select_speed_relevant_subcircuit(n, c, max_neurons=256)
        n = n.reset_index(drop=True)
        i, o = identify_io_neurons(n)
        _ = build_adjacency(n, c)  # topology sanity; weights live in zip
        custom = {"ConnectomeTemporalFeatures": ConnectomeTemporalFeatures}

    model = SAC.load(args.model, env=env, custom_objects=custom, device="cpu")
    obs, _ = env.reset(seed=args.seed)
    xs, ys = [env.x], [env.y]
    frames = []
    info = {}
    dt = float(getattr(env, "dt_ctrl", 0.1))
    max_steps = min(env.MAX_STEPS, int(args.max_sec / dt))

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        xs.append(env.x)
        ys.append(env.y)
        if step % 2 == 0 or term or trunc:
            st = "LAP" if info.get("lap_completed") else ("CRASH" if info.get("collided") else "RUN")
            t_sec = (step + 1) * dt
            title = (
                f"{args.map} {st} t={t_sec:.1f}s v={env.v:.1f} "
                f"prog={info.get('progress', 0):.0%}"
            )
            frames.append(Image.fromarray(render_frame(env, xs, ys, title)))
        if term or trunc:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("connectome" if args.use_cache and not args.plain else "plain")
    gif = out / f"sac_{tag}_{args.map}.gif"
    if len(frames) > 350:
        frames = frames[:: max(1, len(frames) // 350)]
    if frames:
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50, loop=0)
        # still trajectory PNG
        png = out / f"sac_{tag}_{args.map}_traj.png"
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ox, oy = float(env.origin[0]), float(env.origin[1])
        ax.imshow(
            env.occ, cmap="gray_r", origin="upper",
            extent=[ox, ox + env.w * env.resolution, oy, oy + env.h * env.resolution],
        )
        if env.centerline is not None:
            ax.plot(env.centerline[:, 0], env.centerline[:, 1], color="#888", lw=5, alpha=0.3)
        ax.plot(xs, ys, color="#e22", lw=2)
        ax.scatter(xs[0], ys[0], c="#0a5", s=60, zorder=5, label="start")
        ax.scatter(xs[-1], ys[-1], c="#f80", s=60, zorder=5, label="end")
        ax.set_aspect("equal")
        ax.set_title(
            f"{tag} | lap={info.get('lap_completed')} "
            f"prog={info.get('progress', 0):.1%} "
            f"t={(step+1)*dt:.1f}s"
        )
        ax.legend(loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(png)
        plt.close(fig)
        print(f"[watch] traj -> {png.resolve()}")

    print(
        f"[watch] t={(step+1)*dt:.1f}s lap={info.get('lap_completed')} "
        f"crash={info.get('collided')} prog={info.get('progress')} -> {gif.resolve()}"
    )


if __name__ == "__main__":
    main()
