"""
depthfly.polarity — 회로 단계별 극성(부호) 검사.   python -m camera.depthfly.polarity

시뮬 depth 눈으로 실제 장면을 만들어 "이 상황에서 이 뉴런이 +/− 여야 한다"를 하나씩 확인한다 (잡음 끔).
좌표 약속 (모든 단계 공통):
  열 0 = 오른쪽 끝(−45.5°) … 열 63 = 왼쪽 끝(+45.5°).  섹터 0~3 = 오른쪽, 4~7 = 왼쪽.  밴드 0 = 위, 1 = 아래.
  T4/T5 'left'  = 영상이 왼쪽(열 증가)으로 흐름.  HS = left − right  → 영상이 왼쪽으로 흐르면 +.
  VS = down − up → 영상이 아래로 흐르면 +.  LPLC2 = 섹터별 가까움 증가율(다가옴) ≥ 0.
"""

from __future__ import annotations

import numpy as np

from mapless40.raycast import GridMap

from .brain import feature_layout, features_numpy, init_params, n_inputs
from .config import DepthCameraSpec, SceneSpec
from .sensor import DepthEyeSim


def corridor(y_left: float = 0.7, y_right: float = -1.4, x_end: float = 9.0) -> GridMap:
    """x 방향 직선 통로: 왼쪽 벽 y = y_left, 오른쪽 벽 y = y_right, 앞 막힘 x = x_end."""
    res, n = 0.02, 700
    ox, oy = -3.0, -7.0
    occ = np.zeros((n, n), bool)
    ys = oy + (np.arange(n) + 0.5) * res
    xs = ox + (np.arange(n) + 0.5) * res
    occ[(np.abs(ys - y_left) < 0.05), :] = True
    occ[(np.abs(ys - y_right) < 0.05), :] = True
    occ[:, (np.abs(xs - x_end) < 0.05)] = True
    # GridMap 은 행 0 = 위(y 큼) 규약일 수 있으므로 flip 해서 맞춘다
    return GridMap(occ[::-1].copy(), res, (ox, oy))


def _frames(grid, poses, obstacles=None):
    eye = DepthEyeSim(DepthCameraSpec(), SceneSpec())
    obs = np.zeros((0, 3)) if obstacles is None else np.asarray(obstacles, float)
    out = []
    for i, (x, y, yaw) in enumerate(poses):
        o = obs if obstacles is None or np.ndim(obstacles) == 2 else np.asarray(obstacles[i], float)
        out.append(eye.frame(grid, x, y, yaw, o, np.random.default_rng(0), noise=False))
    return np.stack(out)[None].astype(np.float32)


def _groups(feat):
    o, g = 0, {}
    for k, v in feature_layout().items():
        g[k] = feat[o:o + v]; o += v
    for k in ("HS", "VS", "LC", "LUM"):
        g[k] = g[k].reshape(-1, 8)          # (밴드, 섹터) 로
    return g


def scenarios():
    p = init_params(n_inputs())
    f = lambda fr: _groups(features_numpy(fr, p)[0])  # noqa: E731
    g = corridor()
    R = []
    # 1) 제자리 왼쪽 회전 (반시계): 세상이 오른쪽으로 흐름 → HS 합 −
    d = f(_frames(g, [(0, 0, 0.0), (0, 0, 0.03), (0, 0, 0.06)]))
    R.append(("왼쪽으로 회전", "HS 합 < 0 (영상이 오른쪽으로 흐름)", d["HS"].sum(), d["HS"].sum() < 0))
    d = f(_frames(g, [(0, 0, 0.0), (0, 0, -0.03), (0, 0, -0.06)]))
    R.append(("오른쪽으로 회전", "HS 합 > 0", d["HS"].sum(), d["HS"].sum() > 0))
    # 2) 직진. 매끈한 벽은 depth 영상이 안 변함 (벽을 따라 움직여도 같은 방향의 거리가 같음) → HS ≈ 0 이 정상.
    d = f(_frames(g, [(0, 0, 0), (0.12, 0, 0), (0.24, 0, 0)]))
    R.append(("직진: 매끈한 벽", "HS ≈ 0 (depth 엔 무늬가 없음)", np.abs(d["HS"]).sum(), np.abs(d["HS"]).sum() < 1e-3))
    #    벽에 기둥(덕트 틈·이음매 역할)이 있으면 양옆은 바깥쪽(앞→뒤) 흐름 → 왼쪽 섹터 HS +, 오른쪽 섹터 HS −
    posts = [[x, 0.62, 0.06] for x in np.arange(0.6, 6.0, 0.45)] + [[x, -1.32, 0.06] for x in np.arange(0.8, 6.0, 0.45)]
    d = f(_frames(g, [(0, 0, 0), (0.12, 0, 0), (0.24, 0, 0)], posts))
    hl, hr = d["HS"][:, 4:].sum(), d["HS"][:, :4].sum()
    R.append(("직진 + 벽 기둥: 왼쪽 시야", "HS > 0 (왼쪽 바깥으로 흐름)", hl, hl > 0))
    R.append(("직진 + 벽 기둥: 오른쪽 시야", "HS < 0 (오른쪽 바깥으로 흐름)", hr, hr < 0))
    nl, nr = d["NEAR"][4:].max(), d["NEAR"][:4].max()
    R.append(("왼쪽 벽이 더 가까움", "NEAR 왼쪽 > 오른쪽", nl - nr, nl > nr))
    # 3) 정면 장애물로 다가감 / 멀어짐 → LPLC2 가운데 섹터
    ob = [[2.2, 0.0, 0.2]]
    d = f(_frames(g, [(0, 0, 0), (0.15, 0, 0), (0.30, 0, 0)], ob))
    a = d["LPLC2"][3:5].sum()
    d2 = f(_frames(g, [(0.30, 0, 0), (0.15, 0, 0), (0, 0, 0)], ob))
    b = d2["LPLC2"][3:5].sum()
    R.append(("정면 장애물로 다가감", "LPLC2 가운데 > 0", a, a > 0))
    R.append(("정면 장애물에서 멀어짐", "다가감보다 작음", b - a, b < a))
    # 4) 장애물이 왼쪽 앞에 있음 → NEAR·LC 왼쪽 섹터가 큼
    d = f(_frames(g, [(0, 0, 0)] * 3, [[1.5, 0.35, 0.2]]))
    R.append(("장애물 왼쪽 앞", "NEAR 왼쪽(섹터 4~7) 최대", float(np.argmax(d["NEAR"])), np.argmax(d["NEAR"]) >= 4))
    # 5) 하행 뉴런 (좌우 대칭 배선): 왼쪽이 가까우면 왼쪽 DN 흥분·오른쪽 DN 억제, 좌우 뒤집은 영상 = 짝 DN 이 똑같이 반응
    from .brain import forward_numpy, make_dn_wiring_bilateral, mirror_dn_index, symmetric_init
    M, S = make_dn_wiring_bilateral()
    q = symmetric_init(p)
    x = _frames(g, [(0, 0, 0)] * 3, [[1.5, 0.35, 0.2]])
    dn = forward_numpy(x, q, M, S)[0]
    R.append(("DN: 왼쪽에 장애물·벽", "왼쪽 DN 평균 > 오른쪽 DN 평균", dn[:20].mean() - dn[20:40].mean(),
              dn[:20].mean() > dn[20:40].mean()))
    x = _frames(g, [(0, 0, 0), (0.12, 0.05, 0.02), (0.24, 0.1, 0.04)], [[2.0, 0.3, 0.2]])
    e = np.abs(forward_numpy(x[..., ::-1].copy(), q, M, S)[0][mirror_dn_index()] - forward_numpy(x, q, M, S)[0]).max()
    R.append(("DN: 좌우 뒤집은 영상", "짝 DN 출력 동일 (오차 0)", e, e < 1e-5))
    return R


def main():
    rows = scenarios()
    ok = 0
    print(f"{'상황':22s} {'기대':34s} {'값':>10s}  결과")
    for name, exp, val, good in rows:
        ok += bool(good)
        print(f"{name:22s} {exp:34s} {val:10.4f}  {'OK' if good else '틀림'}")
    print(f"{ok}/{len(rows)} OK")
    return ok == len(rows)


if __name__ == "__main__":
    main()
