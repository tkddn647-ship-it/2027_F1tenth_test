"""
depthfly.bc — 레이싱라인 모방학습 (DAgger), torch 없이 numpy.   SAC 시작점 만들기.

  python -m camera.depthfly.bc --out runs/depthfly_bc/bc_init.npz
  python -m camera.depthfly.train ... --bc-init runs/depthfly_bc/bc_init.npz      # 이어서 SAC

선생님(expert) = 레이싱라인 pure pursuit (맵·위치를 아는 privileged 운전자, mapless40.evaluate.pp_controller).
학생 = 실차에 올라갈 회로 그대로 (depth 가까움 3프레임 → 시각엽 → DN 48 → 선형 읽기), 선생님 행동을 따라 하도록.

  반복 0   : 선생님이 운전 + 조향에 잡음 (라인에서 벗어났다 돌아오는 장면도 담기)
  반복 1~K : 학생이 운전 (선생님 섞는 비율 β 를 줄여감), 매 순간 '선생님이라면 어떻게 했을지' 를 정답으로 기록 (DAgger)
  매 반복   : 모은 데이터 전부로 DN 세기·바이어스 + 읽기층 학습 (시각엽 이득은 고정)

장애물은 넣지 않는다 (선생님이 장애물을 피하지 않음). 장애물 회피와 더 빠른 주행은 이어지는 SAC 가 배운다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mapless40.evaluate import pp_controller

from .brain import (PARAM_KEYS, _softplus, features_numpy, init_params, make_dn_wiring_bilateral, n_inputs,
                    symmetric_init)
from .config import DepthFlyConfig, add_cfg_args
from .env import DepthFlyEnv

W_ACT = np.array([2.0, 1.0], np.float32)          # 손실 가중치: 조향, 속도
CMD_IDX = slice(7, 11)   # state 안 '직전 명령 2개' (hist=3: v̄×3, ω̄×3, ω, 명령 4값, Δt×3).
# 모방학습에서 직전 명령을 보게 하면 학생이 '방금 한 걸 그대로 반복' 하는 법만 배운다 (copycat 문제: 검증 손실은 낮은데 혼자 몰면 충돌).
# → 모방학습 동안 읽기층의 이 입력 가중치를 0 으로 고정. SAC 에서는 다시 학습된다.


class Student:
    def __init__(self, n_dn: int = 48, fan_in: int = 12, seed: int = 0, sdim: int = 14):
        self.p = symmetric_init(init_params(n_inputs(), n_dn, seed), n_dn)
        self.M, self.S = make_dn_wiring_bilateral(n_dn, fan_in, seed)
        rng = np.random.default_rng(seed + 7)
        self.mu_w = (rng.normal(0, 0.01, (2, n_dn + sdim))).astype(np.float32)
        self.mu_w[:, n_dn + CMD_IDX.start:n_dn + CMD_IDX.stop] = 0.0
        self.mu_b = np.zeros(2, np.float32)

    def feats(self, near):
        return features_numpy(np.asarray(near, np.float32)[None], self.p)[0]

    def act_from(self, x, st):
        W = self.S * self.M * _softplus(self.p["w_dn"])
        dn = np.tanh(x @ W.T + self.p["b_dn"])
        h = np.concatenate([dn, st], axis=-1)
        return np.tanh(h @ self.mu_w.T + self.mu_b)

    def __call__(self, obs, env=None):
        return self.act_from(self.feats(obs["near"]), np.asarray(obs["state"], np.float32)).astype(np.float32)

    # ---------------------------------------------------------------- 학습 (Adam, 손으로 쓴 역전파)
    def fit(self, X, ST, A, epochs=15, lr=3e-3, batch=512, seed=0, log=print):
        P = {"w_dn": self.p["w_dn"], "b_dn": self.p["b_dn"], "mu_w": self.mu_w, "mu_b": self.mu_b}
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v) for k, v in P.items()}
        b1, b2, t = 0.9, 0.999, 0
        rng = np.random.default_rng(seed)
        N = len(X)
        for ep in range(epochs):
            perm = rng.permutation(N)
            tot = 0.0
            for i in range(0, N, batch):
                j = perm[i:i + batch]
                x, st, a_t = X[j], ST[j], A[j]
                sp = _softplus(P["w_dn"])
                W = self.S * self.M * sp
                z = x @ W.T + P["b_dn"]
                dn = np.tanh(z)
                h = np.concatenate([dn, st], axis=1)
                a = np.tanh(h @ P["mu_w"].T + P["mu_b"])
                err = a - a_t
                tot += float((W_ACT * err ** 2).sum())
                ga = 2 * W_ACT * err / len(j)
                gu = ga * (1 - a ** 2)
                g = {"mu_w": gu.T @ h, "mu_b": gu.sum(0)}
                gdn = (gu @ P["mu_w"])[:, :dn.shape[1]] * (1 - dn ** 2)
                g["b_dn"] = gdn.sum(0)
                g["w_dn"] = (gdn.T @ x) * self.S * self.M / (1 + np.exp(-P["w_dn"]))
                g["mu_w"][:, self.mu_w.shape[1] - 14 + CMD_IDX.start:self.mu_w.shape[1] - 14 + CMD_IDX.stop] = 0.0
                t += 1
                for k in P:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                    P[k] -= (lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + 1e-8)).astype(np.float32)
            if ep == 0 or ep == epochs - 1:
                log(f"    epoch {ep + 1}/{epochs}  loss {tot / N:.4f}")
        self.p["w_dn"], self.p["b_dn"], self.mu_w, self.mu_b = P["w_dn"], P["b_dn"], P["mu_w"], P["mu_b"]

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        st = cls.__new__(cls)
        st.p = {k: d[f"brain.{k}"] for k in PARAM_KEYS}
        st.M, st.S, st.mu_w, st.mu_b = d["mask"], d["sign"], d["mu_w"], d["mu_b"]
        st.meta = json.loads(str(d["meta"]))
        return st

    def save(self, path, meta):
        np.savez(path, **{f"brain.{k}": self.p[k] for k in PARAM_KEYS}, mask=self.M, sign=self.S,
                 mu_w=self.mu_w, mu_b=self.mu_b, meta=json.dumps(meta))


def collect(env, expert, student, steps, beta, steer_noise, rng, ep_steps, opts=None):
    opts = {"n_obstacles": 0} if opts is None else opts
    X, ST, A = [], [], []
    obs, _ = env.reset(options=dict(opts))
    k = noise = 0
    crashes = 0
    for _ in range(steps):
        a_star = expert(obs, env)
        X.append(student.feats(obs["near"])); ST.append(np.asarray(obs["state"], np.float32)); A.append(a_star)
        if rng.random() < beta:
            noise = 0.85 * noise + rng.normal(0, steer_noise)
            a = a_star.copy(); a[0] = np.clip(a[0] + noise, -1, 1)
        else:
            a = student(obs)
        obs, _, te, tr, info = env.step(a)
        k += 1
        if te or tr or k >= ep_steps:
            crashes += bool(info.get("collided"))
            obs, _ = env.reset(options=dict(opts))
            k = noise = 0
    return np.array(X, np.float32), np.array(ST, np.float32), np.array(A, np.float32), crashes


def lap_test(student, cf, maps=("ifac", "roboracer_0817"), max_s=40.0):
    out = {}
    for m in maps:
        env = DepthFlyEnv(maps=(m,), cfg=cf, seed=123, randomize=False)
        obs, _ = env.reset(seed=123, options={"map": m, "s0": 0.0, "lat": 0.0, "dyaw": 0.0, "v0": 2.0, "n_obstacles": 0})
        info = {}
        for _ in range(int(max_s * cf.fps)):
            obs, _, te, tr, info = env.step(student(obs))
            if te or tr or info["laps"] >= 1:
                break
        out[m] = dict(laps=int(info["laps"]), t=round(float(info["t"]), 2), crash=bool(info["collided"]),
                      progress=round(float(info.get("progress", info["laps"])), 2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default="ifac:3,roboracer_0817:3,Spielberg:1,Silverstone:1,Monza:1")
    p.add_argument("--iters", type=int, default=14)
    p.add_argument("--steps", type=int, default=15_000, help="반복마다 모을 스텝")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--steer-noise", type=float, default=0.03, help="OU 잡음 (정상상태 σ ≈ 1.9×)")
    p.add_argument("--ep-s", type=float, default=20.0, help="에피소드 길이 (짧게 끊어 출발 위치를 다양하게)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/depthfly_bc/bc_init.npz")
    p.add_argument("--obstacles", action="store_true",
                   help="장애물 있는 에피소드 (환경 기본 70 %%) + 장애물을 피하는 선생님 (expert.avoid_controller) + 라인 밖 출발 50 %%")
    add_cfg_args(p)
    a = p.parse_args()

    cf = DepthFlyConfig(v_max=a.v_max, brake=a.brake)
    maps = [m for m in a.maps.split(",") if m]
    env = DepthFlyEnv(maps=maps, cfg=cf, seed=a.seed, hard_spawn_p=0.5 if a.obstacles else 0.0)
    if a.obstacles:
        from .expert import avoid_controller
        expert, opts = avoid_controller(cf.env), {}
    else:
        expert, opts = pp_controller(cf.env), None
    st = Student(seed=a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    X = ST = A = None
    hist, best = [], -1e9
    for it in range(a.iters):
        beta = 1.0 if it == 0 else max(0.0, 0.5 * (1 - it / max(a.iters - 1, 1)))
        t0 = time.time()
        x, s, y, crashes = collect(env, expert, st, a.steps, beta, a.steer_noise, rng, int(a.ep_s * cf.fps), opts)
        X = x if X is None else np.concatenate([X, x]); ST = s if ST is None else np.concatenate([ST, s])
        A = y if A is None else np.concatenate([A, y])
        print(f"[bc] 반복 {it}  β={beta:.2f}  +{len(x):,} (총 {len(X):,})  충돌 {crashes}회  수집 {time.time() - t0:.0f}s")
        st.fit(X, ST, A, epochs=a.epochs, seed=a.seed + it)
        from .bc_eval import run as eval_run
        st.meta = {"v_max": a.v_max, "brake": a.brake}
        rows = eval_run(st, spawns=3, laps=2, max_s=40.0)
        done = sum(r[2] >= 2 for r in rows)
        prog = float(np.mean([min(r[5], 2.0) for r in rows]))
        best_lap = {m: min([r[4] for r in rows if r[0] == m and r[4]] or [None]) for m in ("ifac", "roboracer_0817")}
        res = {"완주(2바퀴)": f"{done}/{len(rows)}", "평균 진행(바퀴)": round(prog, 2), "최고랩": best_lap}
        hist.append({"iter": it, "beta": beta, "n": len(X), "crashes": crashes, "eval": res})
        print(f"[bc]   시험 (출발 3곳×2맵, 2바퀴): {res}")
        score = prog
        meta = {"v_max": a.v_max, "brake": a.brake, "wiring": "bilateral", "n_dn": 48, "fan_in": 12,
                "seed": a.seed, "iter": it, "eval": res, "history": hist}
        st.save(out.with_name(out.stem + "_last.npz"), meta)
        if score >= best:
            best = score
            st.save(out, meta)
            print(f"[bc]   best 갱신 → {out}")
    print(f"[bc] 끝. best = {out} (반복 {json.loads(str(np.load(out)['meta']))['iter']})")


if __name__ == "__main__":
    main()
