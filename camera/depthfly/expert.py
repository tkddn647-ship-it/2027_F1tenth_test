"""
depthfly.expert
===============
장애물을 피하는 선생님 (모방학습용, privileged: 맵·레이싱라인·장애물 위치를 안다).

레이싱라인 pure pursuit 는 장애물을 모른다 → 그걸 모방한 정책도 장애물을 못 피했다 (v3c: 장애물 2개 주행 32/60 충돌).
여기서는 pure pursuit 의 목표점을 레이싱라인에서 옆으로 옮긴다: 목표점 후보(횡 ±1.6 m) 중 벽·장애물에서 0.42 m 이상 떨어지고
레이싱라인·직전 선택에 가장 가까운 것. 차 뒤 0.6 m 부터 목표점 앞 1.2 m 까지 필요한 오프셋 중 가장 큰 것을 쓴다.
"""

from __future__ import annotations

import numpy as np

LATS = np.arange(-1.6, 1.61, 0.1)
NEED = 0.42           # 목표점이 벽·장애물 표면에서 이만큼은 떨어져야 한다 (차 반폭 0.15 + 여유)
V_NEAR = 2.2          # 옆으로 비켜 가는 동안 속도 상한 [m/s]


def avoid_controller(cfg, scale: float = 0.9):
    st = {"prev": 0.0, "t": -1.0}

    def choose(env, j, prev):
        """레이싱라인의 j 번째 점에서 옆으로 lat 만큼 옮긴 후보들 중, 여유가 충분하고 라인·직전 선택에 가까운 것."""
        line = env.track.line
        px, py = line.x[j] - LATS * np.sin(line.psi[j]), line.y[j] + LATS * np.cos(line.psi[j])
        cw = env.track.grid.distance(px, py)
        c = cw.copy()
        for ox, oy, r in env.obstacles:
            c = np.minimum(c, np.hypot(px - ox, py - oy) - r)
        # 벽 너머의 후보는 못 쓴다: 라인(0)에서 바깥으로 가며 벽을 만나면 그 뒤는 버림 (장애물은 넘어가도 됨)
        ok = c >= NEED
        mid = len(LATS) // 2
        for rng_ in (range(mid, len(LATS)), range(mid, -1, -1)):
            blocked = False
            for i in rng_:
                blocked |= cw[i] < 0.05
                ok[i] &= not blocked
        cost = np.abs(LATS) + 0.5 * np.abs(LATS - prev)
        if ok.any():
            return float(LATS[np.where(ok, cost, np.inf).argmin()])
        return float(LATS[np.argmax(c)])

    def f(obs, env):
        line = env.track.line
        x, y, _, v, yaw, _, _ = env.state
        if env.t < st["t"]:
            st["prev"] = 0.0                              # 새 에피소드
        st["t"] = env.t
        Ld = max(1.0, 0.3 * v + 0.6)
        j1 = (env.line_idx + int(Ld / line.ds)) % line.n
        off = 0.0
        if len(env.obstacles):
            # 차 뒤 0.6 m ~ 목표점 앞 1.2 m 사이 모든 지점에서 필요한 횡 오프셋 중 가장 큰 것을 쓴다
            # (목표점만 보면 장애물을 지나기도 전에 라인으로 돌아가 장애물 옆구리를 친다)
            k0, k1 = env.line_idx - int(0.6 / line.ds), env.line_idx + int((Ld + 1.2 + 0.4 * v) / line.ds)
            prev = st["prev"]
            for k in range(k0, k1 + 1, max(1, int(0.3 / line.ds))):
                l = choose(env, k % line.n, prev)
                if abs(l) > abs(off):
                    off = l
            st["prev"] = off
            if off:
                # 목표점 자리에서 그 오프셋이 벽에 붙으면 (트랙이 좁아지는 곳) 여유가 있는 가장 가까운 값으로
                Ld = 0.9                                 # 라인 밖에서는 가까이 보고 따라간다 (코너 안쪽 벽을 자르지 않게)
                j1 = (env.line_idx + int(Ld / line.ds)) % line.n
                qx, qy = line.x[j1] - LATS * np.sin(line.psi[j1]), line.y[j1] + LATS * np.cos(line.psi[j1])
                c = env.track.grid.distance(qx, qy)
                for ox, oy, r in env.obstacles:
                    c = np.minimum(c, np.hypot(qx - ox, qy - oy) - r)
                good = c >= 0.35
                if good.any():
                    off = float(LATS[np.where(good, np.abs(LATS - off), np.inf).argmin()])
        px, py = line.x[j1] - off * np.sin(line.psi[j1]), line.y[j1] + off * np.cos(line.psi[j1])
        a = np.arctan2(py - y, px - x) - yaw
        a = (a + np.pi) % (2 * np.pi) - np.pi
        steer = np.arctan2(2 * 0.33 * np.sin(a), np.hypot(px - x, py - y))
        vr = float(np.min(line.at_s(np.linspace(env.s, env.s + 0.3 * v, 5), line.v_ref))) * scale
        slow = abs(off) > 0.05
        if not slow and len(env.obstacles) and v > V_NEAR:
            # 타력 감속뿐이라 미리 줄여야 한다: 감속에 필요한 거리 안에 비켜 가야 할 장애물이 있으면 속도를 낮춘다
            reach = (v * v - V_NEAR * V_NEAR) / (2 * 1.0) + 2.0
            for k in range(env.line_idx, env.line_idx + int(reach / line.ds), max(1, int(0.6 / line.ds))):
                if choose(env, k % line.n, 0.0) != 0.0:
                    slow = True
                    break
        if slow:
            vr = min(vr, V_NEAR)
        vr = np.clip(vr, cfg.action.v_min, cfg.action.v_max)
        return np.array([np.clip(steer / cfg.act.steer_max, -1, 1),
                         (vr - cfg.action.v_min) / (cfg.action.v_max - cfg.action.v_min) * 2 - 1], dtype=np.float32)
    return f
