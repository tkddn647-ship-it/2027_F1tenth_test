"""
mapless40.viz_encoder
=====================
LiDAR CNN 인코더에 "들어가는 것"과 "나오는 것"을 그림으로 본다. torch 없이 numpy 만으로 돈다.

  python -m mapless40.viz_encoder --map ifac --out viz_out
  python -m mapless40.viz_encoder --map ifac --model runs/.../best_model.zip   # 학습된 가중치 (torch 필요)

가중치를 안 주면 PyTorch 기본 초기화와 같은 분포의 **학습 전 랜덤 가중치**를 쓴다.
그래서 출력 특징맵은 "모양·크기·흐름"을 보여줄 뿐, 아직 의미 있는 패턴은 아니다.
학습 후 --model 로 다시 돌리면 같은 그림에서 무엇을 배웠는지 비교할 수 있다.

산출물:
  enc1d_<map>.png   1D Conv: 입력 4×1125 → Conv 32×563 → 64×282 → 64×141 → z 4×48
  bev_<map>.png     BEV: 입력 5채널(프레임 4 + 빈공간) → Conv2d 16·32·64·64 채널 → 192
  bev_<map>.gif     주행하면서 BEV 입력과 Conv 출력이 바뀌는 모습
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import EnvConfig, LidarSpec, NormSpec
from .env import MaplessRaceEnv40
from .evaluate import ftg_controller
from .obs_builder import beam_angles


# ------------------------------------------------------------------ numpy ops
def conv1d(x, w, b, stride=2, pad=2):
    """x (C, L), w (O, C, K) → (O, L')."""
    C, L = x.shape
    O, _, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    Lo = (L + 2 * pad - K) // stride + 1
    idx = np.arange(Lo)[:, None] * stride + np.arange(K)[None, :]     # (Lo, K)
    cols = xp[:, idx]                                                 # (C, Lo, K)
    return np.einsum("clk,ock->ol", cols, w) + b[:, None]


def conv2d(x, w, b, stride=2, pad=1):
    """x (C, H, W), w (O, C, 3, 3) → (O, H', W')."""
    C, H, W = x.shape
    O, _, K, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - K) // stride + 1
    Wo = (W + 2 * pad - K) // stride + 1
    out = np.zeros((O, Ho, Wo), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            patch = xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride]   # (C, Ho, Wo)
            out += np.einsum("chw,oc->ohw", patch, w[:, :, i, j])
    return out + b[:, None, None]


relu = lambda a: np.maximum(a, 0.0)  # noqa: E731


def _init(rng, shape):
    """PyTorch 기본 초기화 (kaiming_uniform a=√5 → U(±1/√fan_in))."""
    fan_in = int(np.prod(shape[1:]))
    bound = 1.0 / np.sqrt(fan_in)
    return rng.uniform(-bound, bound, shape).astype(np.float32), \
        rng.uniform(-bound, bound, shape[0]).astype(np.float32)


class Weights:
    def __init__(self, seed: int = 0, model_zip: str | None = None, n_beams: int = 1125):
        rng = np.random.default_rng(seed)
        self.c1 = _init(rng, (32, 1, 5)); self.c2 = _init(rng, (64, 32, 5)); self.c3 = _init(rng, (64, 64, 5))
        L = n_beams
        for _ in range(3):
            L = (L + 4 - 5) // 2 + 1
        self.pool_k = L // 8
        self.fc1 = _init(rng, (48, 64 * 8))
        self.b1 = _init(rng, (16, 5, 3, 3)); self.b2 = _init(rng, (32, 16, 3, 3))
        self.b3 = _init(rng, (64, 32, 3, 3)); self.b4 = _init(rng, (64, 64, 3, 3))
        self.fcb = _init(rng, (192, 64 * 5 * 5))
        self.trained = {"conv1d": False, "bev": False}
        if model_zip:
            self._load(model_zip)

    def _load(self, model_zip):
        import torch
        from stable_baselines3 import SAC
        sd = SAC.load(model_zip, device="cpu").policy.actor.features_extractor.state_dict()
        g = lambda k: sd[k].numpy()  # noqa: E731
        if "scan_enc.conv.0.weight" in sd:
            self.c1 = (g("scan_enc.conv.0.weight"), g("scan_enc.conv.0.bias"))
            self.c2 = (g("scan_enc.conv.2.weight"), g("scan_enc.conv.2.bias"))
            self.c3 = (g("scan_enc.conv.4.weight"), g("scan_enc.conv.4.bias"))
            self.fc1 = (g("scan_enc.fc.1.weight"), g("scan_enc.fc.1.bias"))
            self.trained["conv1d"] = True
        if "scan_enc.cnn.0.weight" in sd:
            self.b1 = (g("scan_enc.cnn.0.weight"), g("scan_enc.cnn.0.bias"))
            self.b2 = (g("scan_enc.cnn.2.weight"), g("scan_enc.cnn.2.bias"))
            self.b3 = (g("scan_enc.cnn.4.weight"), g("scan_enc.cnn.4.bias"))
            self.b4 = (g("scan_enc.cnn.6.weight"), g("scan_enc.cnn.6.bias"))
            self.fcb = (g("scan_enc.fc.0.weight"), g("scan_enc.fc.0.bias"))
            self.trained["bev"] = True
        del torch


def enc1d(scan, W: Weights):
    """scan (4, N) → 층별 활성 (프레임별 리스트)."""
    acts = {"in": scan, "c1": [], "c2": [], "c3": [], "pool": [], "z": []}
    for k in range(scan.shape[0]):
        a1 = relu(conv1d(scan[k][None], *W.c1))
        a2 = relu(conv1d(a1, *W.c2))
        a3 = relu(conv1d(a2, *W.c3))
        kk = W.pool_k
        p = a3[:, :kk * 8].reshape(64, 8, kk).mean(-1)
        z = relu(W.fc1[0] @ p.reshape(-1) + W.fc1[1])
        for n, v in (("c1", a1), ("c2", a2), ("c3", a3), ("pool", p), ("z", z)):
            acts[n].append(v)
    return acts


# ------------------------------------------------------------------ BEV (policy.BEVRasterizer 의 numpy 판)
def bev_raster(scan, state, lidar: LidarSpec, norm: NormSpec, x_min=-1.5, x_max=13.5,
               y_half=7.5, res=0.1, free_samples=16, free_stride=3):
    T, N = scan.shape
    H, Wd = int(round((x_max - x_min) / res)), int(round(2 * y_half / res))
    img = np.zeros((T + 1, H, Wd), np.float32)
    ang = beam_angles(lidar)
    ca, sa = np.cos(ang), np.sin(ang)
    r = scan.astype(np.float32) * lidar.range_max
    hit = scan < 0.999
    v = state[0:T] * norm.v_scale
    w = state[4:4 + T] * norm.w_scale
    dt = state[13:13 + T] * norm.dt_nominal + norm.dt_nominal
    px, py, pth = np.zeros(T), np.zeros(T), np.zeros(T)
    for k in range(T - 2, -1, -1):
        j = k + 1
        dth, d = w[j] * dt[j], v[j] * dt[j]
        lx, ly = -d * np.cos(dth / 2), d * np.sin(dth / 2)
        c, s = np.cos(pth[j]), np.sin(pth[j])
        px[k], py[k], pth[k] = px[j] + c * lx - s * ly, py[j] + s * lx + c * ly, pth[j] - dth

    def put(ch, x, y):
        row = np.floor((x - x_min) / res).astype(int)
        col = np.floor((y_half - y) / res).astype(int)
        ok = (row >= 0) & (row < H) & (col >= 0) & (col < Wd)
        img[ch, row[ok], col[ok]] = 1.0

    for k in range(T):
        lx = r[k] * ca + lidar.mount_x
        ly = r[k] * sa
        c, s = np.cos(pth[k]), np.sin(pth[k])
        gx, gy = px[k] + c * lx - s * ly, py[k] + s * lx + c * ly
        put(k, gx[hit[k]], gy[hit[k]])
    rs = r[T - 1, ::free_stride]
    fr = (np.arange(free_samples) + 0.5) / free_samples
    tt = rs[:, None] * fr[None]
    put(T, (tt * ca[::free_stride, None] + lidar.mount_x).ravel(), (tt * sa[::free_stride, None]).ravel())
    return img, dict(x_min=x_min, x_max=x_max, y_half=y_half)


def encbev(img, W: Weights):
    a1 = relu(conv2d(img, *W.b1)); a2 = relu(conv2d(a1, *W.b2))
    a3 = relu(conv2d(a2, *W.b3)); a4 = relu(conv2d(a3, *W.b4))
    p = a4[:, :10, :10].reshape(64, 5, 2, 5, 2).mean((2, 4))
    out = relu(W.fcb[0] @ p.reshape(-1) + W.fcb[1])
    return {"in": img, "c1": a1, "c2": a2, "c3": a3, "c4": a4, "out": out}


# ------------------------------------------------------------------ drawing
INK, MUTED, GRID = "#1d2733", "#6b7683", "#d9dee4"
FRAME_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]   # 오래된 → 최신 (한 색상 명도 단계)
CMAP = "Blues"


def _style(ax, title=None, sub=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=4)
    if sub:
        ax.text(0, -0.02, sub, transform=ax.transAxes, va="top", fontsize=7.5, color=MUTED)


def _tile(maps, ncol):
    """(C, h, w) → 한 장 격자 (채널마다 자기 최대값으로 정규화)."""
    C, h, w = maps.shape
    nrow = int(np.ceil(C / ncol))
    g = np.full((nrow * (h + 1) - 1, ncol * (w + 1) - 1), np.nan, np.float32)
    for c in range(C):
        m = maps[c]
        mx = m.max()
        r0, c0 = (c // ncol) * (h + 1), (c % ncol) * (w + 1)
        g[r0:r0 + h, c0:c0 + w] = m / mx if mx > 0 else 0.0
    return g


def _bev_rgb(img):
    T = img.shape[0] - 1
    rgb = np.ones(img.shape[1:] + (3,), np.float32)
    free = img[T] > 0
    rgb[free] = np.array([0.93, 0.95, 0.97])
    for k in range(T):
        col = np.array([int(FRAME_COLORS[k][i:i + 2], 16) / 255 for i in (1, 3, 5)])
        rgb[img[k] > 0] = col
    return rgb


def draw_world(ax, env, span=6.0):
    g = env.track.grid
    x, y, yaw = env.state[0], env.state[1], env.state[4]
    ext = [g.ox, g.ox + g.w * g.res, g.oy, g.oy + g.h * g.res]
    ax.imshow(np.where(g.occ, 0.35, 1.0), cmap="gray", extent=ext, vmin=0, vmax=1, origin="upper")
    raw = env.lidar.scan(x, y, yaw, noise=False)
    ang = beam_angles(env.cfg.lidar)
    lx = x + env.cfg.lidar.mount_x * np.cos(yaw)
    ly = y + env.cfg.lidar.mount_x * np.sin(yaw)
    for i in range(0, ang.size, 15):
        ax.plot([lx, lx + raw[i] * np.cos(yaw + ang[i])], [ly, ly + raw[i] * np.sin(yaw + ang[i])],
                color="#9ecae1", lw=0.5, zorder=2)
    ax.scatter(lx + raw * np.cos(yaw + ang), ly + raw * np.sin(yaw + ang), s=1.2, color="#08306b", zorder=3)
    L = 0.6
    ax.arrow(x, y, L * np.cos(yaw), L * np.sin(yaw), width=0.08, color="#e6550d", zorder=4)
    ax.set_xlim(x - span, x + span); ax.set_ylim(y - span, y + span); ax.set_aspect("equal")


def fig_1d(env, obs, W, out, trained):
    import matplotlib.pyplot as plt
    a = enc1d(obs["scan"].astype(np.float32), W)
    fig = plt.figure(figsize=(15, 8.6), facecolor="white")
    gs = fig.add_gridspec(4, 4, height_ratios=[1.25, 1, 1, 1], hspace=0.55, wspace=0.18)
    tag = "학습된 가중치" if trained else "학습 전 랜덤 가중치 (PyTorch 기본 초기화)"
    fig.suptitle(f"1D Conv LiDAR 인코더 — {env.track.name} · v={env.state[3]:.1f} m/s · {tag}",
                 x=0.01, ha="left", fontsize=12, color=INK)

    ax = fig.add_subplot(gs[0, 0]); draw_world(ax, env)
    _style(ax, "① 실제 상황 (표시용 맵)", "주황 화살표 = 차, 파란 점 = LiDAR 점")
    ax = fig.add_subplot(gs[0, 1:])
    ang = np.degrees(beam_angles(env.cfg.lidar))
    for k in range(4):
        ax.plot(ang, obs["scan"][k].astype(np.float32) * 15, color=FRAME_COLORS[k], lw=1.0,
                label=f"d_(n-{4 - k})")
    ax.set_xlim(ang[0], ang[-1]); ax.set_ylim(0, 15.5)
    ax.set_xlabel("빔 각도 [°]  (−135 = 오른쪽, +135 = 왼쪽)", fontsize=8, color=MUTED)
    ax.set_ylabel("거리 [m]", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5); ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper right")
    ax.set_title("② CNN 입력 1장 = 거리 배열 1125개 (4프레임 겹쳐 그림, 15 m에서 자름)", loc="left", fontsize=9.5, color=INK)

    ax = fig.add_subplot(gs[1, :])
    ax.imshow(obs["scan"].astype(np.float32), aspect="auto", cmap=CMAP + "_r", vmin=0, vmax=1, interpolation="nearest")
    _style(ax, "③ 입력 행렬 4 × 1125  (행 = 시간 n-4…n-1, 열 = 각도, 진할수록 가까움)")

    for i, (key, shp, ttl) in enumerate([("c1", "32 × 563", "Conv1 출력"), ("c2", "64 × 282", "Conv2 출력"),
                                         ("c3", "64 × 141", "Conv3 출력")]):
        ax = fig.add_subplot(gs[2, i])
        m = a[key][-1]
        ax.imshow(m / (m.max(1, keepdims=True) + 1e-9), aspect="auto", cmap=CMAP, interpolation="nearest")
        _style(ax, f"④ {ttl} {shp}", "행 = 필터, 열 = 각도 (최신 프레임)")
    ax = fig.add_subplot(gs[2, 3])
    m = a["pool"][-1]
    ax.imshow(m / (m.max() + 1e-9), aspect="auto", cmap=CMAP, interpolation="nearest")
    _style(ax, "⑤ 평균 풀링 64 × 8", "각도를 8구간으로 요약")

    ax = fig.add_subplot(gs[3, :])
    z = np.stack(a["z"])
    ax.imshow(z / (z.max() + 1e-9), aspect="auto", cmap=CMAP, interpolation="nearest")
    for k in range(1, 4):
        ax.axhline(k - 0.5, color="white", lw=2)
    _style(ax, "⑥ CNN 출력 z: 프레임마다 48개 → 4 × 48 = 192개를 이어 붙여 다음 MLP로",
           "행 = 프레임 n-4…n-1, 열 = 특징 48개")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_bev(env, obs, W, out, trained):
    import matplotlib.pyplot as plt
    img, ex = bev_raster(obs["scan"], obs["state"], env.cfg.lidar, env.cfg.norm)
    a = encbev(img, W)
    fig = plt.figure(figsize=(15, 8.4), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.22, wspace=0.10,
                          left=0.01, right=0.99, top=0.92, bottom=0.04)
    tag = "학습된 가중치" if trained else "학습 전 랜덤 가중치 (PyTorch 기본 초기화)"
    fig.suptitle(f"BEV 2D CNN LiDAR 인코더 — {env.track.name} · v={env.state[3]:.1f} m/s · "
                 f"ω={env.state[5]:+.2f} rad/s · {tag}", x=0.01, ha="left", fontsize=12, color=INK)

    ax = fig.add_subplot(gs[0, 0]); draw_world(ax, env, span=7.5)
    _style(ax, "① 실제 상황 (표시용 맵)")

    ax = fig.add_subplot(gs[0, 1])
    # 행 = 앞(x), 열 = 옆(y). 보기 좋게 앞이 위로 오도록 상하 반전
    ax.imshow(_bev_rgb(img)[::-1], interpolation="nearest")
    H = img.shape[1]
    car_r = H - 1 - int((0 - ex["x_min"]) / 0.1)
    ax.scatter([img.shape[2] / 2], [car_r], marker="^", s=40, color="#e6550d")
    _style(ax, "② CNN 입력 5채널 150 × 150 (10 cm/칸)",
           "진한 파랑 = 최신 프레임 벽, 옅을수록 과거 (IMU·속도로 정렬), 회색 = 빔이 지나간 빈 공간")

    ax = fig.add_subplot(gs[0, 2:])
    chans = np.concatenate([img[k:k + 1] for k in range(5)])
    t = _tile(chans[:, ::-1], 3)
    ax.imshow(t, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    _style(ax, "③ 같은 입력을 채널별로", "윗줄: d_(n-4) · d_(n-3) · d_(n-2)   아랫줄: d_(n-1) · 빈공간")

    for i, (key, ncol, ttl) in enumerate([("c1", 4, "④ Conv1 출력 16 × 75 × 75"),
                                           ("c2", 8, "⑤ Conv2 출력 32 × 38 × 38"),
                                           ("c3", 8, "⑥ Conv3 출력 64 × 19 × 19")]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(_tile(a[key][:, ::-1], ncol), cmap=CMAP, interpolation="nearest")
        _style(ax, ttl, "채널마다 자기 최대값으로 정규화")
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(_tile(a["c4"][:, ::-1], 8), cmap=CMAP, interpolation="nearest")
    _style(ax, "⑦ Conv4 출력 64 × 10 × 10 → 풀링 → FC → 192",
           f"최종 출력 192개 중 0 아닌 값 {int((a['out'] > 0).sum())}개")
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    return img, a


def gif_bev(frames, out, fps=10):
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)


def frame_bev_small(env, obs, W):
    import matplotlib.pyplot as plt
    img, _ = bev_raster(obs["scan"], obs["state"], env.cfg.lidar, env.cfg.norm)
    a = encbev(img, W)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), facecolor="white",
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1], wspace=0.08))
    draw_world(axs[0], env, 7.5); _style(axs[0], f"실제  t={env.t:.1f}s  v={env.state[3]:.1f} m/s")
    axs[1].imshow(_bev_rgb(img)[::-1], interpolation="nearest"); _style(axs[1], "CNN 입력 (BEV 5ch)")
    axs[2].imshow(_tile(a["c1"][:, ::-1], 4), cmap=CMAP, interpolation="nearest"); _style(axs[2], "Conv1 16ch")
    axs[3].imshow(_tile(a["c4"][:, ::-1], 8), cmap=CMAP, interpolation="nearest"); _style(axs[3], "Conv4 64ch → 192")
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def _setup_font():
    import matplotlib
    from matplotlib import font_manager
    matplotlib.use("Agg")
    for name in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic",
                 "AppleGothic", "Noto Sans CJK JP"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="ifac")
    p.add_argument("--model", default=None, help="학습된 SAC zip (torch 필요)")
    p.add_argument("--steps", type=int, default=260, help="스냅샷까지 FTG 로 주행할 틱 수")
    p.add_argument("--gif-steps", type=int, default=240)
    p.add_argument("--out", default="viz_out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    _setup_font()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(max_episode_s=300)
    env = MaplessRaceEnv40(maps=a.map, cfg=cfg, seed=a.seed, randomize=False)
    W = Weights(a.seed, a.model, cfg.lidar.n_beams)
    ctrl = ftg_controller(cfg)
    obs, _ = env.reset(seed=a.seed, options={"s0": 0.0, "lat": 0.0, "dyaw": 0.0, "v0": 1.5})

    frames = []
    for k in range(max(a.steps, a.gif_steps)):
        obs, _, term, trunc, info = env.step(ctrl(obs))
        if term or trunc:
            obs, _ = env.reset(seed=a.seed + k)
        if k < a.gif_steps and k % 3 == 0:
            frames.append(frame_bev_small(env, obs, W))
        if k == a.steps - 1:
            fig_1d(env, obs, W, out / f"enc1d_{env.track.name}.png", W.trained["conv1d"])
            fig_bev(env, obs, W, out / f"bev_{env.track.name}.png", W.trained["bev"])
    gif_bev(frames, out / f"bev_{env.track.name}.gif")
    print(f"saved → {out}/enc1d_{env.track.name}.png, bev_{env.track.name}.png, bev_{env.track.name}.gif")


if __name__ == "__main__":
    main()
