"""
mapless40.evaluate
==================
학습된 정책(또는 기준 컨트롤러)을 여러 맵에서 돌려 완주율·랩타임·충돌을 표로 출력하고
궤적 PNG 를 저장한다.

  python -m mapless40.evaluate --model runs/.../best_model.zip --maps Spielberg,Budapest,ifac
  python -m mapless40.evaluate --model runs/.../actor.ts.pt   --maps ifac        # 배포 파일로 확인
  python -m mapless40.evaluate --baseline ftg  --maps ifac                        # mapless 기준선
  python -m mapless40.evaluate --baseline pp   --maps ifac                        # privileged 상한 참고용
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import MaplessRaceEnv40
from .obs_builder import beam_angles


def ftg_controller(cfg: EnvConfig, look: float = 4.0, bubble_r: float = 0.45, v_max: float = 5.0):
    """Follow-the-Gap (mapless 고전 기준선, 최신 스캔만 사용).

    ±100° 안에서 거리를 look 으로 자르고, 가장 가까운 점 주변을 버블로 지운 뒤
    연속된 빈 구간(gap) 중 가장 긴 것의 가장 먼 점 쪽으로 조향.
    """
    ang = beam_angles(cfg.lidar)
    fov = np.abs(ang) < np.radians(100)
    front = np.abs(ang) < np.radians(8)

    def f(obs, env=None):
        d_full = obs["scan"][-1].astype(np.float32) * cfg.lidar.range_max
        d = np.where(fov, np.minimum(d_full, look), 0.0)
        d = np.convolve(d, np.ones(5) / 5, mode="same")
        i_min = int(np.argmin(np.where(fov, d, 99.0)))
        half = np.arctan2(bubble_r, max(d[i_min], 0.05))
        d[np.abs(ang - ang[i_min]) < half] = 0.0
        free = d > 1.0
        best, cur, best_rng, start = 0, 0, (0, 0), 0
        for i, fr in enumerate(free):
            if fr:
                if cur == 0:
                    start = i
                cur += 1
                if cur > best:
                    best, best_rng = cur, (start, i + 1)
            else:
                cur = 0
        a, b = best_rng
        if b > a:
            seg = d[a:b]
            j = a + int(np.argmax(seg))
            target = 0.5 * ang[j] + 0.5 * ang[(a + b) // 2]
        else:
            target = 0.0
        steer = float(np.clip(target, -cfg.act.steer_max, cfg.act.steer_max))
        v = float(np.clip(min(1.5 + 0.45 * d_full[front].min(), v_max) * (1 - 0.8 * abs(steer) / cfg.act.steer_max),
                          cfg.action.v_min, v_max))
        return np.array([steer / cfg.act.steer_max,
                         (v - cfg.action.v_min) / (cfg.action.v_max - cfg.action.v_min) * 2 - 1],
                        dtype=np.float32)
    return f


def pp_controller(cfg: EnvConfig, scale: float = 0.9):
    """레이싱라인 pure pursuit — privileged(맵 사용), 비교용 상한."""
    def f(obs, env):
        line = env.track.line
        x, y, _, v, yaw, _, _ = env.state
        Ld = max(0.8, 0.3 * v + 0.5)
        j = (env.line_idx + int(Ld / line.ds)) % line.n
        a = np.arctan2(line.y[j] - y, line.x[j] - x) - yaw
        a = (a + np.pi) % (2 * np.pi) - np.pi
        steer = np.arctan2(2 * 0.33 * np.sin(a), Ld)
        vr = float(np.min(line.at_s(np.linspace(env.s, env.s + 0.3 * v, 5), line.v_ref))) * scale
        vr = np.clip(vr, cfg.action.v_min, cfg.action.v_max)
        return np.array([np.clip(steer / cfg.act.steer_max, -1, 1),
                         (vr - cfg.action.v_min) / (cfg.action.v_max - cfg.action.v_min) * 2 - 1],
                        dtype=np.float32)
    return f


def model_controller(path: str):
    p = Path(path)
    if p.suffix == ".pt":
        import torch
        m = torch.jit.load(str(p), map_location="cpu").eval()

        def f(obs, env=None):
            with torch.no_grad():
                s = torch.from_numpy(obs["scan"].astype(np.float32))[None]
                st = torch.from_numpy(obs["state"])[None]
                return m(s, st)[0].numpy()
        return f
    try:
        from stable_baselines3 import SAC
    except ImportError:                     # torch 없는 PC: numpy 판 actor (conv1d)
        from .np_actor import NumpyActor
        act = NumpyActor(str(p))
        print(f"[eval] torch 없음 → numpy actor 로 평가 (step {act.num_timesteps:,})")
        return act
    model = SAC.load(str(p), device="cpu")
    return lambda obs, env=None: model.predict(obs, deterministic=True)[0]


def run(env: MaplessRaceEnv40, ctrl, m: str, s0: float, seed: int, laps_target: int, n_obs: int = 0):
    obs, _ = env.reset(seed=seed, options={"map": m, "s0": s0, "lat": 0.0, "dyaw": 0.0, "v0": 1.5,
                                           "n_obstacles": n_obs})
    xs, ys, vs = [], [], []
    while True:
        obs, _, term, trunc, info = env.step(ctrl(obs, env))
        xs.append(info["x"]); ys.append(info["y"]); vs.append(info["speed"])
        if term or trunc or info["laps"] >= laps_target:
            break
    return info, np.array(xs), np.array(ys), np.array(vs)


def plot(env, m, xs, ys, vs, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = next(t for t in env.tracks if t.name == m)
    g = tr.grid
    ext = [g.ox, g.ox + g.w * g.res, g.oy, g.oy + g.h * g.res]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(~g.occ, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.6)
    ax.plot(tr.line.x, tr.line.y, lw=0.6, color="tab:orange", label="raceline (privileged)")
    for cx, cy, r in env.obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color="#d62728", zorder=5))
    sc = ax.scatter(xs, ys, c=vs, s=3, cmap="viridis", vmin=0, vmax=7)
    fig.colorbar(sc, ax=ax, label="speed [m/s]", shrink=0.7)
    pad = 3
    ax.set_xlim(min(xs.min(), tr.line.x.min()) - pad, max(xs.max(), tr.line.x.max()) + pad)
    ax.set_ylim(min(ys.min(), tr.line.y.min()) - pad, max(ys.max(), tr.line.y.max()) + pad)
    ax.set_aspect("equal"); ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--baseline", choices=["ftg", "pp"], default=None)
    p.add_argument("--maps", default="ifac")
    p.add_argument("--spawns", type=int, default=3)
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--max-s", type=float, default=120.0)
    p.add_argument("--out", default="eval_out")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--obstacles", type=int, default=0, help="트랙 위 장애물 개수 (0 = 없음)")
    a = p.parse_args()

    cfg = EnvConfig(max_episode_s=a.max_s)
    if a.model:
        from .np_actor import apply_model_cfg
        apply_model_cfg(cfg, a.model)          # 학습 때 range_max 등 그대로
    env = MaplessRaceEnv40(maps=a.maps, cfg=cfg, seed=0, sensor_noise=not a.no_noise, randomize=False)
    if a.model:
        ctrl, name = model_controller(a.model), Path(a.model).stem
    elif a.baseline == "ftg":
        ctrl, name = ftg_controller(cfg), "ftg"
    else:
        ctrl, name = pp_controller(cfg), "pp_privileged"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in env.tracks:
        for k in range(a.spawns):
            info, xs, ys, vs = run(env, ctrl, t.name, t.line.length * k / a.spawns, 100 + k, a.laps, a.obstacles)
            best = min(info["lap_times"]) if info["lap_times"] else None
            rows.append(dict(map=t.name, spawn=k, obstacles=a.obstacles, laps=info["laps"], best_lap=best,
                             progress=round(info["progress"], 3), collided=info["collided"],
                             v_mean=round(float(vs.mean()), 2), sim_t=round(info["t"], 1)))
            print(f"{name:14s} {t.name:12s} spawn{k} obs={a.obstacles} laps={info['laps']} best_lap={best} "
                  f"prog={info['progress']:.2f} crash={info['collided']} v̄={vs.mean():.2f}")
            if k == 0:
                plot(env, t.name, xs, ys, vs, out / f"{name}_{t.name}.png",
                     f"{name} · {t.name} · laps {info['laps']} · best {best if best else '-'} s")
    (out / f"{name}_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    done = np.mean([r["laps"] >= 1 for r in rows])
    print(f"[{name}] lap completion {done:.0%}  →  {out}")


if __name__ == "__main__":
    main()
