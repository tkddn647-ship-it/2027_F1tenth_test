"""
compare_st_ab.py — Physical AI ST plain vs connectome (same env/seeds).

  python compare_st_ab.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

OUT = Path("docs/figures")
WATCH = Path("watch_out_race")
OUT.mkdir(parents=True, exist_ok=True)
WATCH.mkdir(parents=True, exist_ok=True)

PLAIN = "runs/plain_Spielberg/best/best_model.zip"
CONN = "connectome_sac_f1tenth_Spielberg_st27_best.zip"
# fallback
if not Path(CONN).exists():
    CONN = "runs/connectome_Spielberg/best/best_model.zip"


def aligned_pose(env: F1TenthMaplessEnv, idx: int = 10) -> np.ndarray:
    cl = env.centerline
    i = int(idx) % len(cl)
    t = env._cl_tangent(i)
    th = float(np.arctan2(t[1], t[0]))
    return np.array([cl[i, 0], cl[i, 1], th], dtype=np.float64)


def make_env(seed: int = 0) -> F1TenthMaplessEnv:
    return F1TenthMaplessEnv(
        "Spielberg",
        seed=seed,
        min_speed=2.0,
        max_speed=7.0,
        max_steer=0.3735,
        physics="st",
    )


def load_model(path: str, env, plain: bool):
    custom = None
    if not plain:
        n, c = load_hemibrain_cache("hemibrain_cache")
        n, c = select_speed_relevant_subcircuit(n, c, max_neurons=256)
        n = n.reset_index(drop=True)
        identify_io_neurons(n)
        build_adjacency(n, c)
        custom = {"ConnectomeTemporalFeatures": ConnectomeTemporalFeatures}
    return SAC.load(path, env=env, custom_objects=custom, device="cpu")


def run_episode(model, env, seed: int, pose=None, max_steps: int = 2000):
    if pose is None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset(seed=seed, options={"pose": pose})
    xs, ys = [env.x], [env.y]
    speeds, slips, ays = [], [], []
    info = {}
    for step in range(max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        xs.append(env.x)
        ys.append(env.y)
        speeds.append(float(info.get("speed", 0)))
        slips.append(abs(float(info.get("slip_angle", 0))))
        ays.append(abs(float(info.get("ay", 0))))
        if term or trunc:
            break
    return {
        "steps": step + 1,
        "t": (step + 1) * 0.1,
        "lap": bool(info.get("lap_completed")),
        "crash": bool(info.get("collided")),
        "rev": bool(info.get("reversed")),
        "prog": float(info.get("progress", 0)),
        "v_mean": float(np.mean(speeds)) if speeds else 0.0,
        "v_max": float(np.max(speeds)) if speeds else 0.0,
        "slip_mean": float(np.mean(slips)) if slips else 0.0,
        "slip_max": float(np.max(slips)) if slips else 0.0,
        "ay_mean": float(np.mean(ays)) if ays else 0.0,
        "ay_max": float(np.max(ays)) if ays else 0.0,
        "xs": xs,
        "ys": ys,
        "info": info,
    }


def save_traj(env, xs, ys, title: str, path: Path):
    ox, oy = float(env.origin[0]), float(env.origin[1])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.imshow(
        env.occ,
        cmap="gray_r",
        origin="upper",
        extent=[ox, ox + env.w * env.resolution, oy, oy + env.h * env.resolution],
    )
    if env.centerline is not None:
        ax.plot(env.centerline[:, 0], env.centerline[:, 1], color="#888", lw=5, alpha=0.3)
    ax.plot(xs, ys, color="#e22", lw=2)
    ax.scatter(xs[0], ys[0], c="#0a5", s=60, zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], c="#f80", s=60, zorder=5, label="end")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_watch_frame(env, xs, ys, title: str) -> np.ndarray:
    ox, oy = float(env.origin[0]), float(env.origin[1])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=90)
    ax.imshow(
        env.occ,
        cmap="gray_r",
        origin="upper",
        extent=[ox, ox + env.w * env.resolution, oy, oy + env.h * env.resolution],
        interpolation="nearest",
    )
    if len(xs) > 1:
        ax.plot(xs, ys, color="#e22", lw=1.5)
    ax.scatter(env.x, env.y, c="#f80", s=40, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def make_gif(model, env, pose, tag: str, max_steps: int = 700):
    obs, _ = env.reset(seed=0, options={"pose": pose})
    xs, ys = [env.x], [env.y]
    frames = []
    info = {}
    for step in range(max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        xs.append(env.x)
        ys.append(env.y)
        if step % 2 == 0 or term or trunc:
            st = "LAP" if info.get("lap_completed") else (
                "CRASH" if info.get("collided") else "RUN"
            )
            title = (
                f"{tag} {st} t={(step+1)*0.1:.1f}s v={env.v:.1f} "
                f"slip={info.get('slip_angle', 0):+.3f} prog={info.get('progress', 0):.0%}"
            )
            frames.append(Image.fromarray(render_watch_frame(env, xs, ys, title)))
        if term or trunc:
            break
    if len(frames) > 280:
        frames = frames[:: max(1, len(frames) // 280)]
    gif = WATCH / f"sac_{tag}_Spielberg.gif"
    png = OUT / f"sac_{tag}_Spielberg_traj.png"
    if frames:
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50, loop=0)
    save_traj(
        env,
        xs,
        ys,
        f"{tag} | lap={info.get('lap_completed')} prog={info.get('progress', 0):.1%} "
        f"t={(step+1)*0.1:.1f}s",
        png,
    )
    # also copy traj to watch_out
    save_traj(
        env,
        xs,
        ys,
        f"{tag} | lap={info.get('lap_completed')} prog={info.get('progress', 0):.1%} "
        f"t={(step+1)*0.1:.1f}s",
        WATCH / f"sac_{tag}_Spielberg_traj.png",
    )
    print(
        f"[gif] {tag}: t={(step+1)*0.1:.1f}s lap={info.get('lap_completed')} "
        f"crash={info.get('collided')} prog={info.get('progress')} -> {gif}"
    )
    return info, xs, ys


def main():
    print(f"[compare] plain={PLAIN}")
    print(f"[compare] conn={CONN}")
    env = make_env(0)
    plain = load_model(PLAIN, env, plain=True)
    conn = load_model(CONN, env, plain=False)

    seeds = list(range(12))
    rows = {"plain": [], "conn": []}
    for seed in seeds:
        for name, model in (("plain", plain), ("conn", conn)):
            env2 = make_env(seed)
            # reload not needed; reuse policy with new env ok for predict
            model.set_env(env2)
            r = run_episode(model, env2, seed)
            rows[name].append(r)
            print(
                f"{name:5s} seed={seed:2d} t={r['t']:.1f}s lap={r['lap']} "
                f"crash={r['crash']} rev={r['rev']} prog={r['prog']:.3f} "
                f"vmean={r['v_mean']:.2f} slipmax={r['slip_max']:.3f}"
            )

    # aligned demo (fair spawn)
    pose = aligned_pose(env, 10)
    env_p = make_env(0)
    env_c = make_env(0)
    plain.set_env(env_p)
    conn.set_env(env_c)
    rp = run_episode(plain, env_p, 0, pose=pose)
    rc = run_episode(conn, env_c, 0, pose=pose)
    print(
        f"aligned plain t={rp['t']:.1f}s lap={rp['lap']} prog={rp['prog']:.3f} "
        f"vmean={rp['v_mean']:.2f} slipmax={rp['slip_max']:.3f}"
    )
    print(
        f"aligned conn  t={rc['t']:.1f}s lap={rc['lap']} prog={rc['prog']:.3f} "
        f"vmean={rc['v_mean']:.2f} slipmax={rc['slip_max']:.3f}"
    )

    # GIFs with aligned start
    env_pg = make_env(0)
    env_cg = make_env(0)
    plain.set_env(env_pg)
    conn.set_env(env_cg)
    make_gif(plain, env_pg, pose, "st27_plain_best")
    make_gif(conn, env_cg, pose, "st27_conn_best")

    def summarize(name: str, rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "laps": sum(1 for r in rs if r["lap"]),
            "crashes": sum(1 for r in rs if r["crash"]),
            "revs": sum(1 for r in rs if r["rev"]),
            "prog_mean": float(np.mean([r["prog"] for r in rs])),
            "prog_max": float(np.max([r["prog"] for r in rs])),
            "t_mean": float(np.mean([r["t"] for r in rs])),
            "v_mean": float(np.mean([r["v_mean"] for r in rs])),
            "v_max_mean": float(np.mean([r["v_max"] for r in rs])),
            "slip_max_mean": float(np.mean([r["slip_max"] for r in rs])),
            "ay_max_mean": float(np.mean([r["ay_max"] for r in rs])),
        }

    summary = {
        "plain_seeds": summarize("plain", rows["plain"]),
        "conn_seeds": summarize("conn", rows["conn"]),
        "aligned_plain": {
            k: rp[k]
            for k in (
                "t",
                "lap",
                "crash",
                "rev",
                "prog",
                "v_mean",
                "v_max",
                "slip_mean",
                "slip_max",
                "ay_mean",
                "ay_max",
            )
        },
        "aligned_conn": {
            k: rc[k]
            for k in (
                "t",
                "lap",
                "crash",
                "rev",
                "prog",
                "v_mean",
                "v_max",
                "slip_mean",
                "slip_max",
                "ay_mean",
                "ay_max",
            )
        },
    }
    out_json = OUT / "st27_plain_vs_connectome.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[compare] summary ->", out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
