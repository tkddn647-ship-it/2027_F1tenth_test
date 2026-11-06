"""
lidarfly.bc — 레이싱라인 모방학습 (DAgger), torch 없이 numpy.  SAC 시작점 만들기.

  python -m lidarfly.bc --out runs/lidarfly_bc/bc_init.npz --v-max 7
  python -m lidarfly.train ... --bc-init runs/lidarfly_bc/bc_init.npz --v-max 7      # 이어서 SAC

camera.depthfly.bc 와 같은 방법 (depthfly v3 에서 처음부터 SAC 는 10만 스텝에 0.2~0.3바퀴,
모방학습 → SAC 로 풀림).  lidarfly 회로 = depthfly 회로(DepthFlyBrain) 앞에 라이다 → 가까움 변환만 붙인 것이라
학생(Student)·학습(fit)은 depthfly 것을 그대로 쓰고, 관측에서 특징을 뽑는 부분만 바꾼다.

  선생님 = 레이싱라인 pure pursuit (맵·위치를 아는 privileged 운전자, mapless40.evaluate.pp_controller)
  반복 0   : 선생님 운전 + 조향 OU 잡음
  반복 1~K : 학생 운전 (선생님 비율 β 감소), 정답은 '선생님이라면' (DAgger)
  매 반복   : 모은 데이터 전부로 DN 세기·바이어스 + 읽기층 학습 (시각엽 이득 고정, 직전 명령 입력 0 고정)

기본은 장애물 없음 (pure pursuit 는 장애물을 모른다).
--obstacles: 장애물 있는 에피소드 (env 기본 70 %) + 장애물을 피하는 선생님 (camera.depthfly.expert.avoid_controller,
  privileged: 맵·장애물 위치를 안다) + 라인 밖 출발 50 % (횡 ±0.4 m, 방향 σ 0.25 rad, 2~4.5 m/s).
  평가에 장애물 2개 조건을 더하고, best 는 두 조건 평균 진행으로 고른다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from camera.depthfly.bc import Student
from camera.depthfly.brain import features_numpy as _df_features
from mapless40.evaluate import pp_controller

from .brain import scans_to_near_np
from .config import DEFAULT_MAPS, V_MAX, lidarfly_config
from .env import LidarFlyEnv


class LidarStudent(Student):
    """depthfly Student 와 같은 회로·읽기층.  입력만 scan (3, 1080) → 가까움 (3, 2, 72)."""

    range_max = 10.0

    def feats(self, scan):
        near = scans_to_near_np(np.asarray(scan, np.float32)[None], self.range_max)
        return _df_features(near, self.p)[0]

    def __call__(self, obs, env=None):
        return self.act_from(self.feats(obs["scan"]), np.asarray(obs["state"], np.float32)).astype(np.float32)

    @classmethod
    def load(cls, path):
        st = super().load(path)
        st.__class__ = cls
        st.range_max = float(st.meta.get("range_max", 10.0))
        return st


def make_cfg(v_max: float, max_s: float = 60.0):
    cfg = lidarfly_config()
    cfg.action.v_max = cfg.raceline.v_max = float(v_max)
    cfg.max_episode_s = float(max_s)
    return cfg


def _reset(env, rng, obstacles: bool, hard_p: float):
    if not obstacles:
        return env.reset(options={"n_obstacles": 0})
    o = {}
    if rng.random() < hard_p:
        o = dict(lat=float(rng.uniform(-0.4, 0.4)), dyaw=float(rng.normal(0.0, 0.25)), v0=float(rng.uniform(2.0, 4.5)))
    return env.reset(options=o)                    # 장애물 개수는 env 기본 (ObstacleSpec.prob 0.7, 1~3개)


def collect(env, expert, student, steps, beta, steer_noise, rng, ep_steps, obstacles=False, hard_p=0.5):
    X, ST, A = [], [], []
    obs, _ = _reset(env, rng, obstacles, hard_p)
    k = noise = 0
    crashes = 0
    for _ in range(steps):
        a_star = expert(obs, env)
        X.append(student.feats(obs["scan"])); ST.append(np.asarray(obs["state"], np.float32)); A.append(a_star)
        if rng.random() < beta:
            noise = 0.85 * noise + rng.normal(0, steer_noise)
            a = a_star.copy(); a[0] = np.clip(a[0] + noise, -1, 1)
        else:
            a = student(obs)
        obs, _, te, tr, info = env.step(a)
        k += 1
        if te or tr or k >= ep_steps:
            crashes += bool(info.get("collided"))
            obs, _ = _reset(env, rng, obstacles, hard_p)
            k = noise = 0
    return np.array(X, np.float32), np.array(ST, np.float32), np.array(A, np.float32), crashes


def eval_run(student, v_max, maps=("ifac", "roboracer_0817"), spawns=3, laps=2, max_s=40.0, seed=7, obstacles=0):
    """출발 위치 spawns 곳 × 맵.  행 = (맵, 출발, 바퀴, 충돌, 최고랩, 진행)."""
    cfg = make_cfg(v_max, max_s)
    rows = []
    env = LidarFlyEnv(maps=",".join(maps), cfg=cfg, seed=seed, randomize=False)
    for m in maps:
        L = next(t for t in env.tracks if t.name == m).line.length
        for k in range(spawns):
            obs, _ = env.reset(seed=seed + k, options={"map": m, "s0": L * k / spawns, "lat": 0.0, "dyaw": 0.0,
                                                       "v0": 2.0, "n_obstacles": obstacles})
            info = {}
            while True:
                obs, _, te, tr, info = env.step(student(obs))
                if te or tr or info["laps"] >= laps:
                    break
            lt = info.get("lap_times", [])
            rows.append((m, k, int(info["laps"]), bool(info["collided"]), round(min(lt), 2) if lt else None,
                         round(float(info["progress"]), 2)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=DEFAULT_MAPS)
    p.add_argument("--iters", type=int, default=14)
    p.add_argument("--steps", type=int, default=15_000, help="반복마다 모을 스텝 (40 Hz)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--steer-noise", type=float, default=0.03, help="OU 잡음 (정상상태 σ ≈ 1.9×)")
    p.add_argument("--ep-s", type=float, default=20.0, help="에피소드 길이 (짧게 끊어 출발 위치를 다양하게)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v-max", type=float, default=V_MAX)
    p.add_argument("--out", default="runs/lidarfly_bc/bc_init.npz")
    p.add_argument("--obstacles", action="store_true",
                   help="장애물 에피소드 + 장애물을 피하는 선생님 + 라인 밖 출발 50 %%")
    a = p.parse_args()

    cfg = make_cfg(a.v_max)
    env = LidarFlyEnv(maps=a.maps, cfg=cfg, seed=a.seed)
    if a.obstacles:
        from camera.depthfly.expert import avoid_controller
        expert = avoid_controller(cfg)
    else:
        expert = pp_controller(cfg)
    st = LidarStudent(seed=a.seed)
    st.range_max = float(cfg.lidar.range_max)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    X = ST = A = None
    hist, best = [], -1e9
    dt = cfg.timing.dt_scan
    for it in range(a.iters):
        beta = 1.0 if it == 0 else max(0.0, 0.5 * (1 - it / max(a.iters - 1, 1)))
        t0 = time.time()
        x, s, y, crashes = collect(env, expert, st, a.steps, beta, a.steer_noise, rng, int(a.ep_s / dt),
                                   obstacles=a.obstacles)
        X = x if X is None else np.concatenate([X, x]); ST = s if ST is None else np.concatenate([ST, s])
        A = y if A is None else np.concatenate([A, y])
        print(f"[bc] 반복 {it}  β={beta:.2f}  +{len(x):,} (총 {len(X):,})  충돌 {crashes}회  "
              f"수집 {time.time() - t0:.0f}s", flush=True)
        st.fit(X, ST, A, epochs=a.epochs, seed=a.seed + it, log=lambda m: print(m, flush=True))
        rows = eval_run(st, a.v_max)
        done = sum(r[2] >= 2 for r in rows)
        prog = float(np.mean([min(r[5], 2.0) for r in rows]))
        best_lap = {m: min([r[4] for r in rows if r[0] == m and r[4]] or [None]) for m in ("ifac", "roboracer_0817")}
        res = {"완주(2바퀴)": f"{done}/{len(rows)}", "평균 진행(바퀴)": round(prog, 2), "최고랩": best_lap}
        if a.obstacles:
            rows_o = eval_run(st, a.v_max, obstacles=2)
            prog_o = float(np.mean([min(r[5], 2.0) for r in rows_o]))
            res["장애물2 완주"] = f"{sum(r[2] >= 2 for r in rows_o)}/{len(rows_o)}"
            res["장애물2 평균 진행"] = round(prog_o, 2)
            prog = 0.5 * (prog + prog_o)
        hist.append({"iter": it, "beta": beta, "n": len(X), "crashes": crashes, "eval": res})
        print(f"[bc]   시험 (출발 3곳×2맵, 2바퀴): {res}", flush=True)
        meta = {"v_max": a.v_max, "range_max": st.range_max, "wiring": "bilateral", "n_dn": 48, "fan_in": 12,
                "seed": a.seed, "iter": it, "eval": res, "history": hist, "sensor": "lidar1080",
                "obstacles": bool(a.obstacles)}
        st.meta = meta
        st.save(out.with_name(out.stem + "_last.npz"), meta)
        if prog >= best:
            best = prog
            st.save(out, meta)
            print(f"[bc]   best 갱신 → {out}", flush=True)
    print(f"[bc] 끝. best = {out} (반복 {json.loads(str(np.load(out)['meta']))['iter']})")


if __name__ == "__main__":
    main()
