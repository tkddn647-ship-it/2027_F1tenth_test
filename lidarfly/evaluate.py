"""lidarfly.evaluate — 맵별 랩타임·완주율.

  python -m lidarfly.evaluate --model runs/lidarfly/best_model.zip --maps ifac,roboracer_0817 --v-max 7
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import V_MAX, lidarfly_config


def model_controller(path: str):
    from stable_baselines3 import SAC
    m = SAC.load(path, device="cpu")
    print(f"[eval] SB3 step {int(m.num_timesteps):,}")
    return lambda obs, env=None: m.predict(obs, deterministic=True)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--maps", default="ifac,roboracer_0817")
    p.add_argument("--spawns", type=int, default=4)
    p.add_argument("--laps", type=int, default=3)
    p.add_argument("--obstacles", type=int, default=0)
    p.add_argument("--max-s", type=float, default=60.0)
    p.add_argument("--v-max", type=float, default=V_MAX, help="최고 속도 [m/s] (학습과 같은 값)")
    a = p.parse_args()

    from .env import LidarFlyEnv
    cfg = lidarfly_config()
    cfg.action.v_max = float(a.v_max)
    cfg.raceline.v_max = float(a.v_max)
    cfg.max_episode_s = float(a.max_s)
    if cfg.action.v_max < cfg.action.v_min:
        raise SystemExit(f"--v-max {cfg.action.v_max} < v_min {cfg.action.v_min}")
    maps = [m for m in a.maps.split(",") if m]
    env = LidarFlyEnv(maps=maps, cfg=cfg, seed=0, randomize=False)
    ctrl = model_controller(a.model)
    for m in maps:
        L = next(t for t in env.tracks if t.name == m).line.length
        finished = 0
        bests = []
        for k in range(a.spawns):
            obs, _ = env.reset(seed=100 + k, options={
                "map": m, "s0": L * k / a.spawns, "lat": 0.0, "dyaw": 0.0,
                "v0": 1.5, "n_obstacles": a.obstacles})
            vs = []
            while True:
                obs, _, te, tr, info = env.step(ctrl(obs, env))
                vs.append(info["speed"])
                if te or tr or info["laps"] >= a.laps:
                    break
            ok = info["laps"] >= a.laps
            finished += int(ok)
            best = min(info["lap_times"]) if info["lap_times"] else None
            if best is not None:
                bests.append(best)
            print(f"{m:16s} spawn{k} obs={a.obstacles} laps={info['laps']} best={best} "
                  f"prog={info['progress']:.2f} crash={info['collided']} v̄={float(np.mean(vs)):.2f}")
        lap = min(bests) if bests else None
        print(f"[lidarfly eval] {m} 완주 {finished}/{a.spawns} 최고랩={lap}")


if __name__ == "__main__":
    main()
