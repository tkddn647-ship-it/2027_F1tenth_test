"""
watch_racetrack.py
==================
MultiTrackEnv에서 학습된 정책(커넥톰 또는 MLP) 주행을 GIF로 저장.

  python watch_racetrack.py --model connectome_ppo_racetrack_cache.zip --use-cache
  python watch_racetrack.py --model baseline_mlp_racetrack.zip --baseline
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from multitrack_env import MultiTrackEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="hemibrain_cache")
    parser.add_argument("--track", type=str, default="oval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="watch_out_race")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = MultiTrackEnv(track_names=[args.track], seed=args.seed,
                        track_switch_every_episode=False)
    custom = {}
    if not args.baseline:
        from connectome_loader import (
            load_hemibrain_cache, build_synthetic_hemibrain,
            select_speed_relevant_subcircuit, identify_io_neurons, build_adjacency,
        )
        from watch_agent import make_extractor_class
        if args.use_cache:
            neurons_df, conn_df = load_hemibrain_cache(args.cache_dir)
        else:
            neurons_df, conn_df = build_synthetic_hemibrain(800, seed=0)
        neurons_df, conn_df = select_speed_relevant_subcircuit(neurons_df, conn_df)
        neurons_df = neurons_df.reset_index(drop=True)
        inp, out = identify_io_neurons(neurons_df)
        A = build_adjacency(neurons_df, conn_df)
        custom = {"ConnectomeFeaturesExtractor": make_extractor_class(A, inp, out)}

    model = PPO.load(args.model, env=env, custom_objects=custom or None, device="cpu")

    obs, _ = env.reset(seed=args.seed)
    xs, ys = [env.pos[0]], [env.pos[1]]
    frames = []
    total = 0.0
    info = {}
    for t in range(env.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total += float(r)
        xs.append(env.pos[0]); ys.append(env.pos[1])
        if t % 3 == 0:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
            cl = env.centerline
            ax.plot(cl[:, 0], cl[:, 1], color="#888", lw=8, solid_capstyle="round", alpha=0.35)
            ax.plot(cl[:, 0], cl[:, 1], color="#ccc", lw=1)
            ax.plot(xs, ys, color="#1a7", lw=2)
            ax.scatter(env.pos[0], env.pos[1], c="#0a5", s=40)
            ax.set_aspect("equal")
            ax.set_title(f"{args.track}  t={t}  R={total:.1f}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(Image.fromarray(buf))
            plt.close(fig)
        if term or trunc:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gif = out / f"drive_{args.track}.gif"
    if frames:
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=60, loop=0)
    print(f"[watch] reward={total:.2f} lap={info.get('lap_completed')} "
          f"lap_time={info.get('lap_time')} → {gif}")
    try:
        import os
        os.startfile(str(gif.resolve()))
    except Exception:
        pass


if __name__ == "__main__":
    main()
