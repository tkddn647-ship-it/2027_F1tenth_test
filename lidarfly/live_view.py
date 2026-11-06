"""
lidarfly.live_view
==================
학습 중인 모델로 시뮬을 실시간 창에 띄운다. 에피소드가 끝날 때마다 모델 파일을 다시 읽어
학습이 진행되는 모습을 그대로 볼 수 있다.

  python -m lidarfly.live_view --model runs/lidarfly/best_model.zip --map ifac
  python -m lidarfly.live_view --model runs/lidarfly/best_model.zip --map roboracer_0817 --obstacles 2

왼쪽: 트랙 전체 + 궤적(속도 색) + 차 + LiDAR 점.  오른쪽: 차 주변 10 m 확대 + 빔.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .config import V_MAX, lidarfly_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/lidarfly/best_model.zip")
    p.add_argument("--map", default="ifac")
    p.add_argument("--obstacles", type=int, default=0)
    p.add_argument("--laps", type=int, default=3)
    p.add_argument("--max-s", type=float, default=60.0)
    p.add_argument("--v-max", type=float, default=V_MAX)
    p.add_argument("--every", type=int, default=2, help="틱 몇 개마다 화면 갱신 (40 Hz 틱)")
    p.add_argument("--speed", type=float, default=1.0, help="재생 배속 (1 = 실시간)")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import SAC

    from mapless40.obs_builder import beam_angles

    from .env import LidarFlyEnv

    cfg = lidarfly_config()
    cfg.action.v_max = cfg.raceline.v_max = float(a.v_max)
    cfg.max_episode_s = float(a.max_s)
    env = LidarFlyEnv(maps=a.map, cfg=cfg, seed=0, randomize=False)
    ang = beam_angles(cfg.lidar)
    rmax = cfg.lidar.range_max
    W = 0.30

    def car_poly(x, y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.array([[cfg.front, W / 2], [cfg.front, -W / 2], [-cfg.rear, -W / 2], [-cfg.rear, W / 2]])
        return np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]]

    model, mtime = None, None

    def load_model():
        nonlocal model, mtime
        try:
            t = os.path.getmtime(a.model)
            if t != mtime:
                model, mtime = SAC.load(a.model, device="cpu"), t
                print(f"[view] 모델 로드: step {int(model.num_timesteps):,}", flush=True)
        except Exception as e:              # 학습 쪽이 쓰는 중이면 다음 에피소드에 다시
            print(f"[view] 모델 로드 실패 ({e}), 이전 모델 사용", flush=True)
        return model

    plt.ion()
    fig, (ax, az) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03, wspace=0.05)
    ep = 0
    while plt.fignum_exists(fig.number):
        m = load_model()
        if m is None:
            print("[view] 모델 파일 대기 중…", flush=True)
            plt.pause(5.0)
            continue
        L = env.tracks[0].line.length
        obs, _ = env.reset(seed=100 + ep, options={"map": a.map, "s0": (L * 0.37 * ep) % L, "lat": 0.0,
                                                   "dyaw": 0.0, "v0": 1.5, "n_obstacles": a.obstacles})
        g, line = env.track.grid, env.track.line
        ext = [g.ox, g.ox + g.w * g.res, g.oy, g.oy + g.h * g.res]
        for axx in (ax, az):
            axx.clear()
            axx.imshow(~g.occ, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.55)
            axx.plot(line.x, line.y, lw=0.6, color="#f28e2b", alpha=0.7)
            for cx, cy, rr in env.obstacles:
                axx.add_patch(plt.Circle((cx, cy), rr, color="#d62728", zorder=5))
            axx.set_xticks([]); axx.set_yticks([]); axx.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        trail = ax.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=a.v_max, s=3, zorder=3)
        trail_z = az.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=a.v_max, s=3, zorder=3)
        pts = ax.scatter([], [], s=0.8, color="#e15759", zorder=4)
        pts_z = az.scatter([], [], s=2, color="#e15759", zorder=4)
        car = ax.add_patch(plt.Polygon(np.zeros((4, 2)), color="#111", zorder=6))
        car_z = az.add_patch(plt.Polygon(np.zeros((4, 2)), color="#111", zorder=6))
        beams = [az.plot([], [], color="#4e79a7", lw=0.3, alpha=0.5)[0] for _ in range(0, len(ang), 20)]
        xs, ys, vs = [], [], []
        k, t_wall0, info = 0, time.perf_counter(), {}
        while plt.fignum_exists(fig.number):
            act = m.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(act)
            xs.append(info["x"]); ys.append(info["y"]); vs.append(info["speed"])
            done = term or trunc or info["laps"] >= a.laps
            if k % a.every == 0 or done:
                x, y, yaw = info["x"], info["y"], info["yaw"]
                r = obs["scan"][-1].astype(np.float32) * rmax
                hit = r < rmax * 0.999
                lx, ly = x + cfg.lidar.mount_x * np.cos(yaw), y + cfg.lidar.mount_x * np.sin(yaw)
                px, py = lx + r * np.cos(yaw + ang), ly + r * np.sin(yaw + ang)
                xy, c = np.c_[xs, ys], np.array(vs)
                for tr in (trail, trail_z):
                    tr.set_offsets(xy); tr.set_array(c)
                pts.set_offsets(np.c_[px[hit], py[hit]]); pts_z.set_offsets(np.c_[px[hit], py[hit]])
                poly = car_poly(x, y, yaw)
                car.set_xy(poly); car_z.set_xy(poly)
                for j, b in enumerate(beams):
                    i = 20 * j
                    b.set_data([lx, px[i]], [ly, py[i]])
                az.set_xlim(x - 5, x + 5); az.set_ylim(y - 5, y + 5)
                best = f"  best {min(info['lap_times']):.2f}s" if info["lap_times"] else ""
                crash = "  CRASH" if info.get("collided") else ""
                fig.suptitle(f"lidarfly  {a.map}  model step {int(m.num_timesteps):,}  ep {ep}  "
                             f"t={info['t']:5.1f}s  v={info['speed']:4.2f} m/s  "
                             f"steer={np.degrees(info['steer']):+5.1f}°  lap {info['laps']}{best}{crash}",
                             fontsize=11)
                lag = info["t"] / a.speed - (time.perf_counter() - t_wall0)
                plt.pause(max(lag, 0.001))
            k += 1
            if done:
                break
        print(f"[view] ep {ep}: laps {info.get('laps')} best "
              f"{min(info['lap_times']) if info.get('lap_times') else None} crash {info.get('collided')}", flush=True)
        if plt.fignum_exists(fig.number):
            plt.pause(1.5)
        ep += 1


if __name__ == "__main__":
    main()
