"""
viz_policy_layers_plain_gif.py
==============================
Physical-AI ST plain SAC 주행 중 MLP 레이어 + 슬립/ay 연속 활성화 GIF.

  python viz_policy_layers_plain_gif.py --model runs/plain_Spielberg/best/best_model.zip
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

from f1tenth_mapless_env import F1TenthMaplessEnv, N_BEAMS, HIST_LEN


def _aligned_start(env: F1TenthMaplessEnv, cl_idx: int = 10) -> np.ndarray:
    cl = env.centerline
    idx = int(cl_idx) % len(cl)
    tang = env._cl_tangent(idx)
    th = float(np.arctan2(tang[1], tang[0]))
    return np.array([cl[idx, 0], cl[idx, 1], th], dtype=np.float64)


def actor_intermediates(policy, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Capture MLP hidden ReLUs + mu for plain SAC actor."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    actor = policy.actor
    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def _fn(_m, _i, o):
            captured[name] = o.detach()

        return _fn

    h1 = actor.latent_pi[1].register_forward_hook(hook("h1"))
    h2 = actor.latent_pi[3].register_forward_hook(hook("h2"))
    with torch.no_grad():
        feat = actor.extract_features(obs_t, actor.features_extractor)
        latent = actor.latent_pi(feat)
        mu = actor.mu(latent)
    h1.remove()
    h2.remove()

    lidar = obs[: N_BEAMS * HIST_LEN].reshape(HIST_LEN, N_BEAMS)
    yaw = obs[N_BEAMS * HIST_LEN :]
    return {
        "lidar": lidar,
        "yaw": yaw,
        "h1": captured["h1"].cpu().numpy()[0],
        "h2": captured["h2"].cpu().numpy()[0],
        "mu": mu.cpu().numpy()[0],
        "latent": latent.cpu().numpy()[0],
    }


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
    lidar = acts["lidar"]
    h1, h2 = acts["h1"], acts["h2"]
    mu = acts["mu"]

    fig = plt.figure(figsize=(12, 7.2), dpi=90)
    fig.suptitle(
        f"ST plain  step={step}  t={t_sec:.1f}s  "
        f"steer={action[0]:+.2f} speed_u={action[1]:.2f}  "
        f"v={info.get('speed', env.v):.1f}  "
        f"slip={info.get('slip_angle', 0):+.3f}  "
        f"ay={info.get('ay', 0):+.1f}  "
        f"prog={info.get('progress', 0):.0%}",
        fontsize=10,
    )

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
    ax.set_title("Track (mapless obs)")

    ax = fig.add_subplot(2, 3, 2)
    ang = np.linspace(-135, 135, N_BEAMS)
    ax.plot(ang, lidar[-1] * 40.0, color="#1f6feb", lw=1.0)
    ax.set_ylim(0, 40)
    ax.set_title("LiDAR (m)")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(lidar, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    ax.set_title("LiDAR hist")
    ax.set_ylabel("t")

    ax = fig.add_subplot(2, 3, 4)
    top1 = np.argsort(np.abs(h1))[::-1][:48]
    ax.bar(np.arange(len(top1)), h1[top1], color="#0969da", alpha=0.85)
    ax.set_title(f"Actor h1 ReLU top48  ||h1||={np.linalg.norm(h1):.1f}")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 3, 5)
    top2 = np.argsort(np.abs(h2))[::-1][:48]
    ax.bar(np.arange(len(top2)), h2[top2], color="#8250df", alpha=0.85)
    ax.set_title(f"Actor h2 ReLU top48  ||h2||={np.linalg.norm(h2):.1f}")
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(2, 3, 6)
    labels = ["steer μ", "speed μ", "slip", "ay/10", "v/7"]
    vals = [
        float(mu[0]),
        float(mu[1]),
        float(info.get("slip_angle", env.beta)),
        float(info.get("ay", env.ay)) / 10.0,
        float(info.get("speed", env.v)) / 7.0,
    ]
    colors = ["#cf222e", "#1a7f37", "#8250df", "#bf3989", "#0550ae"]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("μ / slip / ay / v (norm)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default="runs/plain_Spielberg/best/best_model.zip",
    )
    p.add_argument("--map", type=str, default="Spielberg")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min-speed", type=float, default=2.0)
    p.add_argument("--max-speed", type=float, default=7.0)
    p.add_argument("--max-steer", type=float, default=0.3735)
    p.add_argument("--physics", type=str, default="st", choices=["st", "kinematic"])
    p.add_argument("--start-idx", type=int, default=10)
    p.add_argument("--duration-ms", type=int, default=100)
    p.add_argument("--max-frames", type=int, default=250)
    p.add_argument(
        "--out",
        type=str,
        default="docs/figures/policy_layers_activation_st_plain.gif",
    )
    args = p.parse_args()

    env = F1TenthMaplessEnv(
        args.map,
        seed=args.seed,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        max_steer=args.max_steer,
        physics=args.physics,
    )
    model = SAC.load(args.model, env=env, device="cpu")

    pose = _aligned_start(env, args.start_idx)
    obs, _ = env.reset(seed=args.seed, options={"pose": pose})
    xs, ys = [env.x], [env.y]
    frames: list[Image.Image] = []
    dt = float(getattr(env, "dt_ctrl", 0.1))
    info: dict = {}

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        if step % args.stride == 0:
            acts = actor_intermediates(model.policy, np.asarray(obs, dtype=np.float32))
            rgb = render_frame(
                env, xs, ys, acts, np.asarray(action), info, step, step * dt
            )
            frames.append(Image.fromarray(rgb))

        obs, r, term, trunc, info = env.step(action)
        xs.append(env.x)
        ys.append(env.y)
        if term or trunc:
            acts = actor_intermediates(model.policy, np.asarray(obs, dtype=np.float32))
            rgb = render_frame(
                env, xs, ys, acts, np.asarray(action), info, step, (step + 1) * dt
            )
            frames.append(Image.fromarray(rgb))
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) > args.max_frames:
        step_k = max(1, len(frames) // args.max_frames)
        frames = frames[::step_k]
    if frames:
        frames[0].save(
            out,
            save_all=True,
            append_images=frames[1:],
            duration=int(args.duration_ms),
            loop=0,
        )
    print(f"[viz-plain-gif] frames={len(frames)} -> {out.resolve()}")
    print(
        f"[viz-plain-gif] end lap={info.get('lap_completed')} "
        f"crash={info.get('collided')} rev={info.get('reversed')} "
        f"prog={info.get('progress')} vmax_seen_slip={info.get('slip_angle')}"
    )


if __name__ == "__main__":
    main()
