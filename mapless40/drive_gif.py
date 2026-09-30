"""
mapless40.drive_gif
===================
학습된 모델(zip, torch 불필요)로 시뮬을 달리며 GIF 로 저장.

  python -m mapless40.drive_gif --model best_model.zip --map ifac --laps 2 --obstacles 2 --out drive.gif

왼쪽: 트랙 전체 + 지나온 궤적(속도 색) + 차 + 현재 LiDAR 점
오른쪽: 차 주변 확대 (정책이 보는 LiDAR 빔)
프레임 = 0.1 s (40 Hz 틱 4개) → 실시간 속도로 재생.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--map", default="ifac")
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--obstacles", type=int, default=0)
    p.add_argument("--s0", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-s", type=float, default=40.0)
    p.add_argument("--every", type=int, default=4, help="틱 몇 개마다 한 프레임 (4 = 0.1 s)")
    p.add_argument("--out", default="drive.gif")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from .config import EnvConfig
    from .env import MaplessRaceEnv40
    from .evaluate import model_controller
    from .obs_builder import beam_angles

    cfg = EnvConfig(max_episode_s=a.max_s)
    from .np_actor import apply_model_cfg
    apply_model_cfg(cfg, a.model)
    env = MaplessRaceEnv40(maps=(a.map,), cfg=cfg)
    ctrl = model_controller(a.model)
    obs, _ = env.reset(seed=a.seed, options={"map": a.map, "s0": a.s0, "lat": 0.0, "dyaw": 0.0,
                                              "v0": 1.5, "n_obstacles": a.obstacles})
    g, line = env.track.grid, env.track.line
    ext = [g.ox, g.ox + g.w * g.res, g.oy, g.oy + g.h * g.res]
    ang = beam_angles(cfg.lidar)
    rmax = cfg.lidar.range_max

    fig, (ax, az) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03, wspace=0.05)
    free = ~g.occ
    frames, xs, ys, vs = [], [], [], []
    k, info = 0, {}
    cmap = matplotlib.colormaps["viridis"]
    L, W = 0.60, 0.30

    def car_poly(x, y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.array([[cfg.front, W / 2], [cfg.front, -W / 2], [-cfg.rear, -W / 2], [-cfg.rear, W / 2]])
        return np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]]

    while True:
        act = ctrl(obs, env)
        obs, _, term, trunc, info = env.step(act)
        xs.append(info["x"]); ys.append(info["y"]); vs.append(info["speed"])
        done = term or trunc or info["laps"] >= a.laps
        if k % a.every == 0 or done:
            x, y, yaw = info["x"], info["y"], info["yaw"]
            r = obs["scan"][-1].astype(np.float32) * rmax
            hit = r < rmax * 0.999
            lx = x + cfg.lidar.mount_x * np.cos(yaw)
            ly = y + cfg.lidar.mount_x * np.sin(yaw)
            px = lx + r * np.cos(yaw + ang)
            py = ly + r * np.sin(yaw + ang)
            for axx, zoom in ((ax, False), (az, True)):
                axx.clear()
                axx.imshow(free, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.55)
                axx.plot(line.x, line.y, lw=0.6, color="#f28e2b", alpha=0.7)
                for cx, cy, rr in env.obstacles:
                    axx.add_patch(plt.Circle((cx, cy), rr, color="#d62728", zorder=5))
                axx.scatter(xs, ys, c=np.array(vs), cmap=cmap, vmin=0, vmax=6, s=2, zorder=3)
                if zoom:
                    for i in range(0, len(ang), 15):
                        axx.plot([lx, px[i]], [ly, py[i]], color="#4e79a7", lw=0.3, alpha=0.5)
                axx.scatter(px[hit], py[hit], s=1.5 if zoom else 0.6, color="#e15759", zorder=4)
                axx.add_patch(plt.Polygon(car_poly(x, y, yaw), color="#111", zorder=6))
                axx.set_xticks([]); axx.set_yticks([])
                if zoom:
                    axx.set_xlim(x - 5, x + 5); axx.set_ylim(y - 5, y + 5)
                else:
                    axx.set_xlim(ext[0], ext[1]); axx.set_ylim(ext[2], ext[3])
                axx.set_aspect("equal")
            status = "CRASH" if info.get("collided") else ""
            fig.suptitle(f"{a.map}  t={info['t']:5.1f}s  v={info['speed']:4.2f} m/s  "
                         f"steer={np.degrees(info['steer']):+5.1f}°  lap {info['laps']}"
                         f"{'  best ' + format(min(info['lap_times']), '.2f') + 's' if info['lap_times'] else ''}"
                         f"  {status}", fontsize=11, family="DejaVu Sans")
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
        k += 1
        if done:
            break
    frames += [frames[-1]] * 10                      # 마지막 장면 1초 유지
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * 0.025 * a.every), loop=0,
                   optimize=True)
    print(f"[gif] {out}  frames {len(frames)}  laps {info['laps']}  "
          f"best {min(info['lap_times']) if info['lap_times'] else None}  collided {info.get('collided')}")


if __name__ == "__main__":
    main()
