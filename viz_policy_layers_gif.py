"""
viz_policy_layers_gif.py
========================
Connectome SAC 주행 중 레이어/커넥톰 연속 활성화 GIF.

  python viz_policy_layers_gif.py --model connectome_sac_f1tenth_Spielberg_80k.zip --map Spielberg --use-cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from stable_baselines3 import SAC

from connectome_loader import (
    identify_io_neurons,
    load_hemibrain_cache,
    select_speed_relevant_subcircuit,
)
from f1tenth_mapless_env import F1TenthMaplessEnv, N_BEAMS
from policy_sac_connectome import ConnectomeTemporalFeatures, HIST_LEN


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def render_frame(
    env: F1TenthMaplessEnv,
    xs: list[float],
    ys: list[float],
    acts: dict,
    action: np.ndarray,
    info: dict,
    step: int,
    t_sec: float,
) -> np.ndarray:
    lidar = _to_np(acts["lidar"])[0]
    yaw = _to_np(acts["yaw"])[0]
    z = _to_np(acts["z"])[0]
    h_seq = _to_np(acts["h_seq"])[0]
    y_dn = _to_np(acts["y_dn"])[0]
    h_last = _to_np(acts["h"])[0]
    fuse = _to_np(acts["fuse"])[0]

    fig = plt.figure(figsize=(12, 7.2), dpi=90)
    fig.suptitle(
        f"step={step}  t={t_sec:.1f}s  steer={action[0]:+.2f} speed_u={action[1]:.2f}  "
        f"v={info.get('speed', env.v):.1f}  slip={info.get('slip_angle', 0):+.3f}  "
        f"prog={info.get('progress', 0):.0%}",
        fontsize=10,
    )

    # track
    ax = fig.add_subplot(2, 3, 1)
    ox, oy = float(env.origin[0]), float(env.origin[1])
    ax.imshow(
        env.occ,
        cmap="gray_r",
        origin="upper",
        extent=[ox, ox + env.w * env.resolution, oy, oy + env.h * env.resolution],
        interpolation="nearest",
    )
    if len(xs) > 1:
        ax.plot(xs, ys, color="#e22", lw=1.4)
    ax.scatter(env.x, env.y, c="#f80", s=28, zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Track")

    # lidar
    ax = fig.add_subplot(2, 3, 2)
    ang = np.linspace(-135, 135, N_BEAMS)
    ax.plot(ang, lidar[-1] * 40.0, color="#1f6feb", lw=1.0)
    ax.set_ylim(0, 40)
    ax.set_title("LiDAR (m)")
    ax.grid(True, alpha=0.25)

    # lidar hist
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(lidar, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    ax.set_title("LiDAR hist")
    ax.set_ylabel("t")

    # encoder z
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(z.T, aspect="auto", cmap="coolwarm", origin="lower", vmin=-1, vmax=1)
    ax.set_title("Encoder z_t")
    ax.set_xlabel("t")

    # connectome energy + DN
    ax = fig.add_subplot(2, 3, 5)
    energy = np.linalg.norm(h_seq, axis=1)
    ax.plot(np.arange(HIST_LEN), energy, "o-", color="#8250df", label="||h||")
    ax.set_ylabel("||h_t||", color="#8250df")
    ax2 = ax.twinx()
    ax2.bar(np.arange(len(y_dn)), y_dn, color="#1a7f37", alpha=0.55, width=0.7)
    ax2.set_ylabel("DN", color="#1a7f37")
    ax.set_title("Connectome ||h|| + DN")
    ax.set_xlabel("unroll t / DN idx")
    ax.grid(True, alpha=0.25)

    # top neurons + fuse energy
    ax = fig.add_subplot(2, 3, 6)
    top = np.argsort(np.abs(h_last))[::-1][:40]
    ax.bar(np.arange(len(top)), h_last[top], color="#8250df", alpha=0.85)
    ax.set_title(f"h top40  |  ||fuse||={np.linalg.norm(fuse):.2f}")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _aligned_start(env: F1TenthMaplessEnv, cl_idx: int = 10) -> np.ndarray:
    """Centerline-aligned pose so reverse-termination does not fire immediately."""
    cl = env.centerline
    idx = int(cl_idx) % len(cl)
    tang = env._cl_tangent(idx)
    th = float(np.arctan2(tang[1], tang[0]))
    return np.array([cl[idx, 0], cl[idx, 1], th], dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--map", type=str, default="Spielberg")
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--cache-dir", type=str, default="hemibrain_cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--stride", type=int, default=2, help="매 N 제어스텝마다 프레임")
    p.add_argument("--min-speed", type=float, default=2.0)
    p.add_argument("--max-speed", type=float, default=3.5)
    p.add_argument("--max-steer", type=float, default=0.30)
    p.add_argument(
        "--physics",
        type=str,
        default="kinematic",
        choices=["kinematic", "st"],
        help="80k connectome zip은 kinematic 학습",
    )
    p.add_argument("--start-idx", type=int, default=10, help="센터라인 출발 인덱스")
    p.add_argument("--duration-ms", type=int, default=90, help="GIF 프레임 간격(ms)")
    p.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="저장 프레임 상한 (초과 시 균등 샘플)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="docs/figures/policy_layers_activation.gif",
    )
    args = p.parse_args()

    if not args.use_cache:
        raise SystemExit("--use-cache required")

    env = F1TenthMaplessEnv(
        args.map,
        seed=args.seed,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        max_steer=args.max_steer,
        physics=args.physics,
    )
    n, c = load_hemibrain_cache(args.cache_dir)
    n, c = select_speed_relevant_subcircuit(n, c, max_neurons=256)
    n = n.reset_index(drop=True)
    _ = identify_io_neurons(n)
    custom = {"ConnectomeTemporalFeatures": ConnectomeTemporalFeatures}
    model = SAC.load(args.model, env=env, custom_objects=custom, device="cpu")
    fe = model.policy.actor.features_extractor
    if not hasattr(fe, "forward_intermediates"):
        raise SystemExit("forward_intermediates missing")

    pose = _aligned_start(env, args.start_idx)
    obs, _ = env.reset(seed=args.seed, options={"pose": pose})
    xs, ys = [env.x], [env.y]
    frames: list[Image.Image] = []
    dt = float(getattr(env, "dt_ctrl", 0.1))
    info: dict = {}

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        if step % args.stride == 0:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                acts = fe.forward_intermediates(obs_t)
            rgb = render_frame(
                env, xs, ys, acts, np.asarray(action), info, step, step * dt
            )
            frames.append(Image.fromarray(rgb))

        obs, r, term, trunc, info = env.step(action)
        xs.append(env.x)
        ys.append(env.y)
        if term or trunc:
            # last frame
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                acts = fe.forward_intermediates(obs_t)
            rgb = render_frame(
                env, xs, ys, acts, np.asarray(action), info, step, (step + 1) * dt
            )
            frames.append(Image.fromarray(rgb))
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) > args.max_frames:
        step = max(1, len(frames) // args.max_frames)
        frames = frames[::step]
    if frames:
        frames[0].save(
            out,
            save_all=True,
            append_images=frames[1:],
            duration=int(args.duration_ms),
            loop=0,
        )
    print(f"[viz-gif] frames={len(frames)} -> {out.resolve()}")
    print(
        f"[viz-gif] end lap={info.get('lap_completed')} crash={info.get('collided')} "
        f"prog={info.get('progress')}"
    )


if __name__ == "__main__":
    main()
