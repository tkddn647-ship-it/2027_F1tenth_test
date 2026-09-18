"""
watch_race_map.py
=================
실제 f1tenth_racetracks 맵(PNG) 위에서 완주 GIF 생성.

  python watch_race_map.py --model connectome_ppo_spielberg.zip --use-cache --map Spielberg
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import yaml

from multitrack_env import MultiTrackEnv


def map_extent(map_png: Path):
    """yaml origin/resolution으로 imshow extent (left,right,bottom,top)."""
    yml = map_png.with_name(map_png.name.replace("_map.png", "_map.yaml"))
    if not yml.exists():
        yml = next(map_png.parent.glob("*.yaml"), None)
    img = np.array(Image.open(map_png))
    h, w = img.shape[:2]
    if yml is None:
        return None, img
    meta = yaml.safe_load(yml.read_text(encoding="utf-8"))
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    # ROS: origin = lower-left of map in world
    extent = [ox, ox + w * res, oy, oy + h * res]
    return extent, img


def render_frame(env, xs, ys, title: str):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    if env.map_png is not None:
        extent, img = map_extent(Path(env.map_png))
        if extent is not None:
            ax.imshow(img, cmap="gray", origin="lower", extent=extent, alpha=0.95)
        else:
            ax.imshow(img, cmap="gray", origin="upper", alpha=0.9)
    cl = env.centerline
    ax.plot(cl[:, 0], cl[:, 1], color="#4af", lw=1.2, alpha=0.7, label="centerline")
    ax.plot(xs, ys, color="#e22", lw=2.0, label="ego")
    ax.scatter(env.pos[0], env.pos[1], c="#f80", s=40, zorder=5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    # zoom around car
    pad = 25.0
    ax.set_xlim(env.pos[0] - pad, env.pos[0] + pad)
    ax.set_ylim(env.pos[1] - pad, env.pos[1] + pad)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="없으면 휴리스틱(벽 따라가기)으로 완주 영상")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--map", type=str, default="Spielberg")
    parser.add_argument("--racetracks-dir", type=str, default="f1tenth_racetracks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="watch_out_race")
    args = parser.parse_args()

    env = MultiTrackEnv(
        track_names=[args.map],
        racetracks_dir=args.racetracks_dir,
        track_switch_every_episode=False,
        fixed_start=True,
        seed=args.seed,
    )

    model = None
    use_pp = args.model is None
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

    def pure_pursuit_action():
        """센터라인 look-ahead (영상용 완주). 곡률 크면 감속."""
        look = 18
        tgt = env.centerline[(env.cl_idx + look) % env.n_cl]
        near = env.centerline[(env.cl_idx + 5) % env.n_cl]
        to = tgt - env.pos
        desired = np.arctan2(to[1], to[0])
        err = (desired - env.heading + np.pi) % (2 * np.pi) - np.pi
        steer = float(np.clip(err / 0.45, -1, 1))
        # 급커브(헤딩 오차·근처 곡률)면 감속
        t1 = env.tangents[env.cl_idx]
        t2 = env.tangents[(env.cl_idx + 10) % env.n_cl]
        bend = float(1.0 - np.clip(np.dot(t1, t2), -1, 1))
        target_spd = 0.85 - 0.55 * bend - 0.25 * abs(err)
        target_spd = float(np.clip(target_spd, 0.25, 0.85))
        cur = float(obs[env.N_RAYS])
        accel = float(np.clip((target_spd - cur) * 2.5, -1, 1))
        return np.array([steer, accel], dtype=np.float32)

    obs, _ = env.reset(seed=args.seed)
    xs, ys = [float(env.pos[0])], [float(env.pos[1])]
    frames = []
    total = 0.0
    info = {}
    for t in range(env.MAX_STEPS):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif use_pp:
            action = pure_pursuit_action()
        else:
            rays = obs[: env.N_RAYS]
            left, right = rays[: env.N_RAYS // 2].mean(), rays[env.N_RAYS // 2 :].mean()
            steer = float(np.clip((right - left) * 3.0, -1, 1))
            accel = 0.65 if obs[env.N_RAYS] < 0.75 else 0.25
            action = np.array([steer, accel], dtype=np.float32)

        obs, r, term, trunc, info = env.step(action)
        total += float(r)
        xs.append(float(env.pos[0]))
        ys.append(float(env.pos[1]))
        if t % 4 == 0 or term or trunc:
            status = "LAP" if info.get("lap_completed") else (
                "OFF" if info.get("off_track") else "RUN"
            )
            lt = info.get("lap_time")
            title = f"{args.map}  {status}  t={t}  prog={info.get('progress',0):.0%}"
            if lt:
                title += f"  lap={lt:.1f}s"
            frames.append(Image.fromarray(render_frame(env, xs, ys, title)))
        if term or trunc:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "model" if args.model else "heuristic"
    gif = out / f"lap_{args.map}_{tag}.gif"
    if frames:
        # subsample if too long
        if len(frames) > 200:
            frames = frames[:: max(1, len(frames) // 180)]
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(
        f"[watch] map={args.map} lap={info.get('lap_completed')} "
        f"time={info.get('lap_time')} steps={t} -> {gif.resolve()}"
    )
    try:
        import os
        os.startfile(str(gif.resolve()))
    except Exception:
        pass


if __name__ == "__main__":
    main()
