"""
camfly.evaluate  —  학습 모델(zip) 평가.  encoder=fly 는 torch 없이 numpy 로 돈다.

  python -m camera.camfly.evaluate --model best_model.zip --maps ifac,roboracer_0817 --spawns 4 --laps 3 --obstacles 0
"""

from __future__ import annotations

import argparse

import numpy as np


def model_controller(path: str):
    try:
        from .np_actor import CamNumpyActor
        act = CamNumpyActor(path)
        print(f"[eval] numpy actor (step {act.num_timesteps:,})")
        return act
    except Exception as e:
        print(f"[eval] numpy actor 불가 ({e}) → SB3 로 로드")
        from stable_baselines3 import SAC
        m = SAC.load(path, device="cpu")
        return lambda obs, env=None: m.predict(obs, deterministic=True)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--maps", default="ifac,roboracer_0817")
    p.add_argument("--spawns", type=int, default=4)
    p.add_argument("--laps", type=int, default=3)
    p.add_argument("--obstacles", type=int, default=0)
    p.add_argument("--max-s", type=float, default=60.0)
    a = p.parse_args()
    from .config import CamFlyConfig
    from .env import CamFlyEnv
    cf = CamFlyConfig()
    cf.env.max_episode_s = a.max_s
    maps = [m for m in a.maps.split(",") if m]
    env = CamFlyEnv(maps=maps, cfg=cf, seed=0, randomize=False)
    ctrl = model_controller(a.model)
    done = 0
    for m in maps:
        L = next(t for t in env.tracks if t.name == m).line.length
        for k in range(a.spawns):
            obs, _ = env.reset(seed=100 + k, options={"map": m, "s0": L * k / a.spawns, "lat": 0.0, "dyaw": 0.0,
                                                       "v0": 1.5, "n_obstacles": a.obstacles})
            vs = []
            while True:
                obs, _, te, tr, info = env.step(ctrl(obs, env))
                vs.append(info["speed"])
                if te or tr or info["laps"] >= a.laps:
                    break
            done += info["laps"] >= a.laps
            best = min(info["lap_times"]) if info["lap_times"] else None
            print(f"{m:15s} spawn{k} obs={a.obstacles} laps={info['laps']} best={best} "
                  f"prog={info['progress']:.2f} crash={info['collided']} v̄={np.mean(vs):.2f}")
    print(f"[camfly eval] completion {done}/{len(maps) * a.spawns}")


if __name__ == "__main__":
    main()
