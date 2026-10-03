"""
camfly.viz
==========
시뮬 카메라가 무엇을 보는지 GIF 로.  (torch 불필요)

  python -m camfly.viz --map ifac --driver pp --seconds 8 --out eye.gif
  python -m camfly.viz --map ifac --model best_model.zip --out eye.gif     # 학습 모델 (torch 필요)

왼쪽: 트랙 + 시야 부채꼴 + depth 가짜 스캔 점
오른쪽 위: 파리 눈 원본 밝기 (16×64, 정면 ±15° 가 가운데 절반)
오른쪽 가운데: 광수용체 적응 후 (정책 입력)
오른쪽 아래: 연속 프레임 차이 (움직임 — T4/T5 가 보는 것)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="ifac")
    p.add_argument("--driver", choices=["pp", "model"], default="pp")
    p.add_argument("--model", default=None)
    p.add_argument("--obstacles", type=int, default=1)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--out", default="camfly_eye.gif")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from mapless40.evaluate import pp_controller

    from .config import CamFlyConfig
    from .env import CamFlyEnv
    from .evaluate import model_controller

    cf = CamFlyConfig()
    env = CamFlyEnv(maps=(a.map,), cfg=cf, seed=a.seed)
    if a.model:
        ctrl = model_controller(a.model)
    else:
        ctrl = pp_controller(cf.env)
    obs, _ = env.reset(seed=a.seed, options={"map": a.map, "s0": 0.0, "lat": 0.0, "dyaw": 0.0, "v0": 2.0,
                                              "n_obstacles": a.obstacles})
    g, cam = env.track.grid, cf.cam
    ext = [g.ox, g.ox + g.w * g.res, g.oy, g.oy + g.h * g.res]
    az = cam.col_azimuths()
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.35, wspace=0.08)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_raw, ax_ad, ax_mo = (fig.add_subplot(gs[i, 1]) for i in range(3))
    frames, xs, ys = [], [], []
    prev = None
    n_steps = int(a.seconds * cf.fps)
    for k in range(n_steps):
        obs, _, te, tr, info = env.step(ctrl(obs, env))
        xs.append(info["x"]); ys.append(info["y"])
        if k % a.every == 0 or te or tr:
            x, y, yaw = info["x"], info["y"], info["yaw"]
            lx, ly = x + cam.mount_x * np.cos(yaw), y + cam.mount_x * np.sin(yaw)
            d = obs["dscan"][-1].astype(np.float32) * cam.depth_max
            ax_map.clear()
            ax_map.imshow(~g.occ, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.6)
            for cx, cy, r in env.obstacles:
                ax_map.add_patch(plt.Circle((cx, cy), r, color="#d62728"))
            ax_map.plot(xs, ys, color="#4e79a7", lw=1)
            for t in (az[0], az[-1]):
                ax_map.plot([lx, lx + cam.depth_max * np.cos(yaw + t)], [ly, ly + cam.depth_max * np.sin(yaw + t)],
                            color="#f28e2b", lw=0.8)
            ok = d < cam.depth_max - 1e-3
            ax_map.scatter(lx + d[ok] * np.cos(yaw + az[ok]), ly + d[ok] * np.sin(yaw + az[ok]), s=6,
                           color="#e15759", zorder=5)
            ax_map.set_xlim(x - 6, x + 6); ax_map.set_ylim(y - 6, y + 6); ax_map.set_aspect("equal")
            ax_map.set_xticks([]); ax_map.set_yticks([])
            ax_map.set_title(f"{a.map} t={info['t']:.1f}s v={info['speed']:.1f} m/s  (orange = 91° FOV, red = depth scan)",
                             fontsize=9)
            raw = env.last_raw_eye
            ext_e = [np.degrees(az[0]), np.degrees(az[-1]), -1, 1]
            ax_raw.clear(); ax_raw.imshow(raw[:, ::-1], cmap="gray", vmin=0, vmax=1, aspect="auto",
                                          interpolation="nearest")
            ax_raw.set_title("fly eye raw (16 x 64, left | right)", fontsize=8); ax_raw.set_xticks([]); ax_raw.set_yticks([])
            cur = obs["eye"][-1].astype(np.float32)
            ax_ad.clear(); ax_ad.imshow(cur[:, ::-1], cmap="gray", vmin=-1, vmax=1, aspect="auto",
                                        interpolation="nearest")
            ax_ad.set_title("after photoreceptor adaptation (policy input)", fontsize=8)
            ax_ad.set_xticks([]); ax_ad.set_yticks([])
            mo = cur - obs["eye"][-2].astype(np.float32)
            ax_mo.clear(); ax_mo.imshow(mo[:, ::-1], cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto",
                                        interpolation="nearest")
            ax_mo.set_title("frame difference n - (n-1)  (motion: what T4/T5 see)", fontsize=8)
            ax_mo.set_xticks([]); ax_mo.set_yticks([])
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
                          .convert("P", palette=Image.ADAPTIVE, colors=128))
        if te or tr:
            break
    out = Path(a.out)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * a.every / cf.fps),
                   loop=0, optimize=True)
    print(f"[viz] {out} frames {len(frames)}  crash {info.get('collided')}")


if __name__ == "__main__":
    main()
