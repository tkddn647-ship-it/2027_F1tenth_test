"""
mapless40.raceline
==================
레이싱라인 로드 · 제원 기반 속도 프로파일 재계산 · Frenet 투영.

학습 전용 privileged 정보다 (보상 + critic 입력). 정책(actor)과 실차에는 들어가지 않는다.

우선순위: <Track>_raceline.csv (f1tenth_racetracks, ';' 구분)
          → *raceline*.csv → *centerline*.csv (x,y 만 사용)
속도는 CSV 의 vx 를 쓰지 않고 RacelineSpec(a_lat, a_accel, a_brake)로 다시 계산한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import RacelineSpec


def _read_xy(path: Path) -> np.ndarray:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    delim = ";" if ";" in lines[0] else ","
    rows = []
    for ln in lines:
        parts = [p for p in ln.replace(" ", "").split(delim) if p != ""]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=np.float64)
    if "raceline" in path.name and delim == ";" and arr.shape[1] >= 3:
        return arr[:, 1:3]          # s; x; y; ...
    return arr[:, 0:2]


def find_line_file(map_yaml: Path) -> Path | None:
    folder = map_yaml.parent
    stem = map_yaml.stem.replace("_map", "")
    for pat in (f"{stem}_raceline.csv", "*raceline*.csv", f"{stem}_centerline.csv",
                f"{map_yaml.stem}_centerline.csv", "*centerline*.csv"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def _smooth_periodic(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    pad = win
    xp = np.concatenate([x[-pad:], x, x[:pad]])
    return np.convolve(xp, k, mode="same")[pad:-pad]


def push_from_walls(x: np.ndarray, y: np.ndarray, grid, margin: float,
                    iters: int = 60, step: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """벽까지 거리 < margin 인 점을 거리장 기울기 방향으로 밀어낸다.

    f1tenth_racetracks 레이싱라인은 점 차량 기준이라 벽에서 0.1~0.3 m 까지 붙는다.
    Roboracer 차체(반폭 0.15, 앞 0.33)로는 그대로 따라가면 부딪히므로 여유를 준다.
    """
    x, y = x.copy(), y.copy()
    for _ in range(iters):
        d = grid.distance(x, y)
        bad = d < margin
        if not bad.any():
            break
        gx, gy = grid.grad(x[bad], y[bad])
        nrm = np.hypot(gx, gy) + 1e-9
        x[bad] += step * gx / nrm
        y[bad] += step * gy / nrm
    return x, y


def min_curvature_band(x: np.ndarray, y: np.ndarray, grid, margin: float,
                       iters: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """센터라인 → 최소곡률 근사 라인 (elastic band).

    매 반복마다 점을 이웃 두 점의 중점 쪽으로 당겨(줄을 팽팽하게 = 곡률·길이 감소)
    벽까지 margin 보다 가까워진 점은 거리장 기울기로 밀어낸다.
    레이싱라인 CSV 가 없는 맵(ifac 등 직접 매핑한 트랙)에서 센터라인 대신 쓴다.
    ifac: 센터라인 46.1 m → 40.6 m, 같은 한계에서 이론 랩타임 14.6 s → 10.6 s.
    """
    x, y = x.copy(), y.copy()
    for _ in range(iters):
        xm = 0.5 * (np.roll(x, 1) + np.roll(x, -1))
        ym = 0.5 * (np.roll(y, 1) + np.roll(y, -1))
        x += 0.5 * (xm - x)
        y += 0.5 * (ym - y)
        x, y = push_from_walls(x, y, grid, margin, iters=3, step=0.02)
    return x, y


class Raceline:
    def __init__(self, xy: np.ndarray, spec: RacelineSpec, ds: float = 0.2, grid=None,
                 decel_fn=None, optimize: bool = False):
        self.decel_fn = decel_fn
        xy = np.asarray(xy, dtype=np.float64)
        if np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
            xy = xy[:-1]
        seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(s_raw[-1])
        n = max(20, int(length / ds))
        s_u = np.arange(n) * (length / n)
        xy_c = np.vstack([xy, xy[:1]])
        x = np.interp(s_u, s_raw, xy_c[:, 0])
        y = np.interp(s_u, s_raw, xy_c[:, 1])
        win = max(1, int(spec.smooth_m / (length / n)))
        x, y = _smooth_periodic(x, win), _smooth_periodic(y, win)
        if grid is not None and spec.wall_margin > 0:
            x, y = push_from_walls(x, y, grid, spec.wall_margin)
            if optimize:
                x, y = min_curvature_band(x, y, grid, spec.wall_margin)
            x, y = _smooth_periodic(x, 3), _smooth_periodic(y, 3)
            x, y = push_from_walls(x, y, grid, spec.wall_margin, iters=20)
        # 재매개변수화 (밀어낸 뒤 호장 길이 갱신)
        seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(s_raw[-1])
        n = max(20, int(self.length / ds))
        self.ds = self.length / n
        self.s = np.arange(n) * self.ds
        self.x = np.interp(self.s, s_raw, np.append(x, x[0]))
        self.y = np.interp(self.s, s_raw, np.append(y, y[0]))
        dx = np.gradient(np.concatenate([self.x[-2:], self.x, self.x[:2]]))[2:-2]
        dy = np.gradient(np.concatenate([self.y[-2:], self.y, self.y[:2]]))[2:-2]
        self.psi = np.arctan2(dy, dx)
        dpsi = np.diff(np.unwrap(np.concatenate([self.psi, self.psi[:1]])))
        kappa = dpsi / self.ds
        self.kappa = _smooth_periodic(kappa, max(1, int(spec.kappa_smooth_m / self.ds)))
        self.spec = spec
        self.v_ref = self.speed_profile(spec)
        self.n = n

    def speed_profile(self, sp: RacelineSpec) -> np.ndarray:
        k = np.abs(self.kappa) + 1e-6
        v = np.minimum(sp.v_max, np.sqrt(sp.a_lat / k))
        n = v.size
        for _ in range(2):                     # 폐곡선: 두 바퀴 돌려 수렴
            for i in range(n):                 # 가속 제한 (앞으로)
                vp = v[i - 1]
                v[i] = min(v[i], np.sqrt(vp * vp + 2 * sp.a_accel * self.ds))
            for i in range(n - 1, -1, -1):     # 감속 제한 (뒤로)
                vn = v[(i + 1) % n]
                a_b = sp.a_brake if self.decel_fn is None else min(sp.a_brake, self.decel_fn(vn))
                v[i] = min(v[i], np.sqrt(vn * vn + 2 * a_b * self.ds))
        return np.maximum(v, sp.v_min)

    # --- Frenet -------------------------------------------------------------
    def project(self, px: float, py: float, hint: int | None = None) -> tuple[int, float, float]:
        """(idx, s, e_y) — e_y 는 라인 기준 왼쪽 +."""
        if hint is None:
            cand = np.arange(self.n)
        else:
            cand = (hint + np.arange(-int(10 / self.ds), int(30 / self.ds))) % self.n
        d2 = (self.x[cand] - px) ** 2 + (self.y[cand] - py) ** 2
        j = int(np.argmin(d2))
        if hint is not None and d2[j] > 9.0:          # 3 m 이상 벗어나면 전역 재탐색
            return self.project(px, py, None)
        i = int(cand[j])
        tx, ty = np.cos(self.psi[i]), np.sin(self.psi[i])
        rx, ry = px - self.x[i], py - self.y[i]
        along = rx * tx + ry * ty
        e_y = -rx * ty + ry * tx
        s = (self.s[i] + along) % self.length
        return i, float(s), float(e_y)

    def heading_error(self, idx: int, yaw: float) -> float:
        return float((yaw - self.psi[idx] + np.pi) % (2 * np.pi) - np.pi)

    def at_s(self, s_query: np.ndarray, arr: np.ndarray) -> np.ndarray:
        sq = np.asarray(s_query) % self.length
        return np.interp(sq, np.append(self.s, self.length), np.append(arr, arr[0]))

    def lookahead(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        sp = self.spec
        q = s + sp.lookahead_ds * np.arange(1, sp.lookahead_n + 1)
        return self.at_s(q, self.kappa), self.at_s(q, self.v_ref)

    def ds_wrap(self, s0: float, s1: float) -> float:
        d = s1 - s0
        if d < -0.5 * self.length:
            d += self.length
        elif d > 0.5 * self.length:
            d -= self.length
        return d


def load_raceline(map_yaml: Path, spec: RacelineSpec, grid=None, decel_fn=None) -> tuple[Raceline, Path]:
    f = find_line_file(Path(map_yaml))
    if f is None:
        raise FileNotFoundError(f"레이싱라인/센터라인 CSV 없음: {map_yaml.parent}")
    optimize = spec.optimize_centerline and "raceline" not in f.name
    return Raceline(_read_xy(f), spec, grid=grid, decel_fn=decel_fn, optimize=optimize), f
