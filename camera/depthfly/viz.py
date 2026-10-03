"""
depthfly.viz — depth 파리 눈이 보는 것 + 회로 활동 GIF (torch 불필요)

  python -m camera.depthfly.viz --map ifac --out depthfly.gif                    # 학습 전 (pure pursuit 주행, 초기 회로)
  python -m camera.depthfly.viz --map ifac --model best_model.zip --out run.gif   # 학습된 정책이 운전
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="ifac")
    p.add_argument("--model", default=None)
    p.add_argument("--obstacles", type=int, default=1)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--out", default="depthfly.gif")
    from .config import add_cfg_args
    add_cfg_args(p)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from mapless40.evaluate import pp_controller

    from .brain import feature_layout, features_numpy, forward_numpy, init_params, make_dn_wiring, n_inputs
    from .config import DepthFlyConfig
    from .env import DepthFlyEnv
    from .evaluate import model_controller

    cf = DepthFlyConfig(v_max=a.v_max, brake=a.brake)
    env = DepthFlyEnv(maps=(a.map,), cfg=cf, seed=a.seed)
    if a.model:
        ctrl = model_controller(a.model)
        P, M, S = getattr(ctrl, "params", None), getattr(ctrl, "mask", None), getattr(ctrl, "sign", None)
    else:
        ctrl = pp_controller(cf.env)
        P = None
    if P is None:
        P = init_params(n_inputs())
        M, S = make_dn_wiring(n_inputs())
    lay = feature_layout()
    obs, _ = env.reset(seed=a.seed, options={"map": a.map, "s0": 0.0, "lat": 0.0, "dyaw": 0.0, "v0": 2.0,
                                              "n_obstacles": a.obstacles})
    g, cam = env.track.grid, cf.cam
    ext = [g.ox, g.ox + g.w * g.res, g.oy, g.oy + g.h * g.res]
    az = cam.col_azimuths()
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1], hspace=0.6, wspace=0.08)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_n, ax_d, ax_f, ax_dn = (fig.add_subplot(gs[i, 1]) for i in range(4))
    frames, xs, ys, info = [], [], [], {}
    for k in range(int(a.seconds * cf.fps)):
        obs, _, te, tr, info = env.step(ctrl(obs, env))
        xs.append(info["x"]); ys.append(info["y"])
        if k % a.every == 0 or te or tr:
            x, y, yaw = info["x"], info["y"], info["yaw"]
            lx, ly = x + cam.mount_x * np.cos(yaw), y + cam.mount_x * np.sin(yaw)
            near = obs["near"].astype(np.float32)
            ax_map.clear()
            ax_map.imshow(~g.occ, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.6)
            for cx, cy, r in env.obstacles:
                ax_map.add_patch(plt.Circle((cx, cy), r, color="#d62728"))
            ax_map.plot(xs, ys, color="#4e79a7", lw=1)
            for t in (az[0], az[-1]):
                ax_map.plot([lx, lx + cam.depth_max * np.cos(yaw + t)], [ly, ly + cam.depth_max * np.sin(yaw + t)],
                            color="#f28e2b", lw=0.8)
            ax_map.set_xlim(x - 6, x + 6); ax_map.set_ylim(y - 6, y + 6); ax_map.set_aspect("equal")
            ax_map.set_xticks([]); ax_map.set_yticks([])
            ax_map.set_title(f"{a.map} t={info['t']:.1f}s v={info['speed']:.1f} m/s  (orange: 91° FOV, 7 m)", fontsize=9)
            ax_n.clear(); ax_n.imshow(near[-1][:, ::-1], cmap="magma", vmin=0, vmax=0.6, aspect="auto",
                                      interpolation="nearest")
            ax_n.set_title("depth fly eye: nearness 0.25/r (16 x 64, left | right)", fontsize=8)
            ax_d.clear(); ax_d.imshow((near[-1] - near[-2])[:, ::-1], cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                                      aspect="auto", interpolation="nearest")
            ax_d.set_title("change n - (n-1)  (input to T4/T5)", fontsize=8)
            feat = features_numpy(near[None], P)[0]
            o, names, vals = 0, [], []
            for nm, n in lay.items():
                vals.append(feat[o:o + n]); names.append(nm); o += n
            ax_f.clear()
            ax_f.bar(np.arange(feat.size), feat, color="#4e79a7", width=1.0)
            o = 0
            for nm, n in lay.items():
                ax_f.axvline(o - 0.5, color="#bbb", lw=0.5); ax_f.text(o + n / 2, ax_f.get_ylim()[1] * 0.85, nm,
                                                                      ha="center", fontsize=6); o += n
            ax_f.set_xticks([]); ax_f.set_title("optic lobe outputs (HS, VS, LPLC2, LC, L3)", fontsize=8)
            dn = forward_numpy(near[None], P, M, S)[0]
            ax_dn.clear(); ax_dn.bar(np.arange(dn.size), dn, color="#e15759", width=1.0); ax_dn.set_ylim(-1, 1)
            ax_dn.set_xticks([]); ax_dn.set_title("descending neurons (48) -> linear readout -> steer, speed", fontsize=8)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
                          .convert("P", palette=Image.ADAPTIVE, colors=128))
        if te or tr:
            break
    out = Path(a.out)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * a.every / cf.fps), loop=0,
                   optimize=True)
    print(f"[viz] {out} frames {len(frames)} crash {info.get('collided')}")


if __name__ == "__main__":
    main()
