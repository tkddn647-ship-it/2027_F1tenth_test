"""
depthfly.report_figs — REPORT.md 그림 (torch 불필요).   python -m camera.depthfly.report_figs --out docs/report

  sensor.png   카메라 옆모습(16행 시선과 덕트) · 거리별 덕트가 걸리는 행 수 · 같은 장면의 깨끗한/잡음 가까움 영상
  signal.png   주행 중 시각엽 경로별 평균 크기: 수정 전 / 수정 후 / 수정 후 잡음만 (로그 축)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#2b2b2b", "#6b6b66", "#e4e4e0"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)


def group_magnitudes(gain_fix: bool, noise: bool, static: bool, seconds: float = 10.0):
    from mapless40.evaluate import pp_controller

    from . import brain
    from .config import DepthFlyConfig
    from .env import DepthFlyEnv
    old = brain.GAIN0, brain.SIGMA_LAM
    if not gain_fix:                      # 수정 전 회로 재현: 고정 이득 없음(=ln2), 대비 적응 없음
        brain.GAIN0, brain.SIGMA_LAM = (np.log(2.0),) * 5, None
    try:
        cf = DepthFlyConfig()
        env = DepthFlyEnv(maps=("ifac",), cfg=cf, seed=0, sensor_noise=noise)
        obs, _ = env.reset(seed=0, options={"map": "ifac", "s0": 0, "lat": 0, "dyaw": 0, "v0": 2.0, "n_obstacles": 2})
        ctrl, p, F = pp_controller(cf.env), brain.init_params(brain.n_inputs()), []
        for _ in range(int(seconds * cf.fps)):
            obs, _, te, tr, _ = env.step(ctrl(obs, env))
            n = obs["near"].astype(np.float32)
            if static:
                x, y, yaw = env.state[0], env.state[1], env.state[4]
                n = np.stack([env.eye.frame(env.track.grid, x, y, yaw, env.obstacles, env.rng, noise=True)
                              for _ in range(3)])
            F.append(brain.features_numpy(n[None], p)[0])
            if te or tr:
                break
    finally:
        brain.GAIN0, brain.SIGMA_LAM = old
    F, o, out = np.abs(np.array(F)), 0, {}
    for k, v in brain.feature_layout().items():
        out[k] = float(F[:, o:o + v].mean()); o += v
    return out


def fig_signal(out: Path):
    import matplotlib.pyplot as plt
    rows = [("before fix (driving)", group_magnitudes(False, True, False), ORANGE),
            ("after fix (driving)", group_magnitudes(True, True, False), BLUE),
            ("after fix, sensor noise only (car still)", group_magnitudes(True, True, True), AQUA)]
    names = list(rows[0][1])
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=130)
    for i, (lab, d, c) in enumerate(rows):
        ax.bar(x + (i - 1) * (w + 0.02), [d[k] for k in names], w, color=c, label=lab, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(3e-5, 0.5)
    ax.set_xticks(x, ["HS\n(sideways flow)", "VS\n(vertical flow)", "LPLC2\n(looming)", "LC\n(edges)",
                      "LUM\n(mean near)", "NEAR\n(max near)"], fontsize=8)
    ax.set_ylabel("mean |output| (log)", color=MUTED, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    _style(ax)
    ax.set_title("Optic-lobe pathway sizes while driving ifac (pure pursuit, 2 obstacles)", fontsize=9, loc="left", pad=22)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    fig.tight_layout(); fig.savefig(out / "signal.png"); plt.close(fig)
    return rows


def fig_sensor(out: Path):
    import matplotlib.pyplot as plt

    from camera.camfly.tests import _wall_grid

    from .config import CameraSpec, SceneSpec
    from .sensor import DepthEyeSim
    cam, sc = CameraSpec(), SceneSpec()
    el = cam.row_elevations()
    fig = plt.figure(figsize=(10, 6.2), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.55, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    for e in el:
        hit = (cam.height + np.arange(0.05, 7, 0.01) * np.tan(e))
        d_end = 7.0 if e >= 0 else min(7.0, cam.height / -np.tan(e))
        ax.plot([0, d_end], [cam.height, cam.height + d_end * np.tan(e)], color=BLUE, lw=0.5, alpha=0.6)
    for d in (1.0, 3.0, 6.0):
        ax.add_patch(plt.Rectangle((d, 0), 0.08, sc.duct_height, color=ORANGE, zorder=3))
        ax.text(d + 0.04, sc.duct_height + 0.04, f"duct {d:.0f} m", ha="center", fontsize=7, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot([0], [cam.height], "o", color=INK, ms=4)
    ax.set_xlim(-0.2, 7); ax.set_ylim(-0.05, 0.9)
    ax.set_xlabel("forward [m]", fontsize=8, color=MUTED); ax.set_ylabel("height [m]", fontsize=8, color=MUTED)
    ax.set_title("Side view: 16 eye rows (camera 0.18 m, pitch -10°)", fontsize=9, loc="left")
    _style(ax)
    ax2 = fig.add_subplot(gs[0, 1])
    ds = np.linspace(0.3, 7, 200)
    n = [int(((cam.height + d * np.tan(el) >= cam.min_obj_height) & (cam.height + d * np.tan(el) <= sc.duct_height)).sum())
         for d in ds]
    ax2.step(ds, n, where="mid", color=BLUE, lw=2)
    ax2.set_ylim(0, 8); ax2.set_xlabel("distance to duct [m]", fontsize=8, color=MUTED)
    ax2.set_ylabel("rows that see the duct (of 16)", fontsize=8, color=MUTED)
    ax2.grid(color=GRID, lw=0.6); _style(ax2)
    ax2.set_title("33 cm duct covers few rows beyond ~2 m", fontsize=9, loc="left")
    g = _wall_grid(2.5)
    eye = DepthEyeSim(cam, sc)
    obst = np.array([[1.4, 0.35, 0.18]])
    clean = eye.frame(g, 0.0, 0.0, 0.25, obst, np.random.default_rng(0), noise=False)
    eye.hold.reset()
    noisy = eye.frame(g, 0.0, 0.0, 0.25, obst, np.random.default_rng(1), noise=True)
    for j, (img, t) in enumerate(((clean, "nearness, no sensor error"), (noisy, "nearness with assumed Gemini 2L error + holes"))):
        a = fig.add_subplot(gs[1, j])
        im = a.imshow(img[:, ::-1], cmap="magma", vmin=0, vmax=0.4, aspect="auto", interpolation="nearest")
        a.set_title(t + "\n(wall 2.5 m ahead, obstacle 1.4 m ahead; floor removed)", fontsize=8, loc="left")
        a.set_xlabel("column (left | right, centre ±15° dense)", fontsize=7, color=MUTED)
        a.set_ylabel("row", fontsize=7, color=MUTED); _style(a)
    fig.colorbar(im, ax=fig.axes[-2:], fraction=0.025, pad=0.02).ax.tick_params(labelsize=7)
    fig.savefig(out / "sensor.png", bbox_inches="tight"); plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/report")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_sensor(out)
    rows = fig_signal(out)
    for lab, d, _ in rows:
        print(f"{lab:42s} " + "  ".join(f"{k} {v:.2e}" for k, v in d.items()))


if __name__ == "__main__":
    main()
