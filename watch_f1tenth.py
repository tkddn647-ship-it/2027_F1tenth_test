"""
watch_f1tenth.py
================
F1TENTH 맵에서 학습된 커넥톰 에이전트를 돌려 GIF/궤적으로 저장.

  python watch_f1tenth.py --model connectome_ppo_f1tenth_berlin.zip --map berlin
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome_loader import DEFAULT_CACHE_DIR, load_hemibrain_cache, identify_io_neurons, build_adjacency
from f1tenth_env import F1TenthEnv
from watch_agent import make_extractor_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="connectome_ppo_f1tenth_berlin.zip")
    parser.add_argument("--map", type=str, default="berlin")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--n-beams", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-eval", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="watch_out_f1tenth")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from PIL import Image

    neurons_df, conn_df = load_hemibrain_cache(args.cache_dir)
    input_idx, output_idx = identify_io_neurons(neurons_df)
    A_signed = build_adjacency(neurons_df, conn_df)
    Extractor = make_extractor_class(A_signed, input_idx, output_idx)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"모델 없음: {model_path}\n먼저 학습: "
                         f"python train_connectome.py --use-cache --env f1tenth --map {args.map}")

    env = F1TenthEnv(map_name=args.map, n_beams=args.n_beams, seed=args.seed)
    model = PPO.load(
        str(model_path), env=env,
        custom_objects={"ConnectomeFeaturesExtractor": Extractor},
        device="cpu",
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rewards = []
    for i in range(args.n_eval):
        obs, _ = env.reset(seed=args.seed + i)
        total, steps = 0.0, 0
        for steps in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += float(r)
            if term or trunc:
                break
        rewards.append(total)
        print(f"  ep{i}: reward={total:.2f} steps={steps+1} "
              f"crash={info['collided']} lap={info['lap_count']} path={info['path_len']:.1f}")

    print(f"[watch] 평균 보상 {np.mean(rewards):.2f}")

    # GIF — 매 프레임 render는 너무 느려서 간헐 캡처
    obs, _ = env.reset(seed=args.seed)
    frames = []
    total = 0.0
    xs, ys = [env.x], [env.y]
    for t in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total += float(r)
        xs.append(env.x)
        ys.append(env.y)
        if t % 4 == 0:
            frames.append(env.render())
        if term or trunc:
            break

    gif_path = out / f"drive_{args.map}.gif"
    images = [Image.fromarray(f) for f in frames]
    if len(images) > 120:
        images = images[:: max(1, len(images) // 100)]
    if images:
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=80, loop=0)
    print(f"[watch] GIF: {gif_path.resolve()}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        env.occ, cmap="gray_r", origin="upper",
        extent=[
            env.origin[0], env.origin[0] + env.w * env.resolution,
            env.origin[1], env.origin[1] + env.h * env.resolution,
        ],
    )
    ax.plot(xs, ys, color="#1a7", lw=2)
    ax.scatter(xs[0], ys[0], c="blue", s=40)
    ax.scatter(xs[-1], ys[-1], c="red", s=40)
    ax.set_title(f"{args.map}  reward={total:.1f}  path={info['path_len']:.1f}")
    ax.set_aspect("equal")
    traj_path = out / f"trajectory_{args.map}.png"
    fig.savefig(traj_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[watch] 궤적: {traj_path.resolve()}")

    try:
        import os
        os.startfile(str(gif_path.resolve()))
        os.startfile(str(traj_path.resolve()))
    except Exception:
        pass


if __name__ == "__main__":
    main()
