"""depthfly.bc_eval — 모방학습 결과(npz) 평가: 출발 위치 여러 곳 × 여러 바퀴.   python -m camera.depthfly.bc_eval --npz bc_init.npz"""

from __future__ import annotations

import argparse

import numpy as np

from .bc import Student
from .config import DepthFlyConfig
from .env import DepthFlyEnv


def run(npz, maps=("ifac", "roboracer_0817"), spawns=4, laps=3, obstacles=0, max_s=60.0, seed=7):
    st = Student.load(npz) if isinstance(npz, (str, bytes)) or hasattr(npz, "__fspath__") else npz
    meta = getattr(st, "meta", {})
    cf = DepthFlyConfig(v_max=meta.get("v_max", 8.0), brake=meta.get("brake", False))
    rows = []
    for m in maps:
        env = DepthFlyEnv(maps=(m,), cfg=cf, seed=seed, randomize=False)
        L = next(t for t in env.tracks if t.name == m).line.length
        for k in range(spawns):
            obs, _ = env.reset(seed=seed + k, options={"map": m, "s0": L * k / spawns, "lat": 0.0, "dyaw": 0.0,
                                                       "v0": 2.0, "n_obstacles": obstacles})
            info = {}
            for _ in range(int(max_s * cf.fps)):
                obs, _, te, tr, info = env.step(st(obs))
                if te or tr or info["laps"] >= laps:
                    break
            lt = info.get("lap_times", [])
            rows.append((m, k, int(info["laps"]), bool(info["collided"]), min(lt) if lt else None,
                         round(float(info["progress"]), 2)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--maps", default="ifac,roboracer_0817")
    p.add_argument("--spawns", type=int, default=4)
    p.add_argument("--laps", type=int, default=3)
    p.add_argument("--obstacles", type=int, default=0)
    a = p.parse_args()
    rows = run(a.npz, tuple(a.maps.split(",")), a.spawns, a.laps, a.obstacles)
    for m, k, n, c, best, prog in rows:
        print(f"{m:16s} 출발 {k}  바퀴 {n}  {'충돌' if c else '무사고'}  최고랩 {best if best else '-'}  진행 {prog}")
    ok = sum(1 for r in rows if r[2] >= a.laps)
    print(f"[bc_eval] {ok}/{len(rows)} 가 {a.laps}바퀴 완주")


if __name__ == "__main__":
    main()
