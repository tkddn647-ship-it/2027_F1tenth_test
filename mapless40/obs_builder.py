"""
mapless40.obs_builder
=====================
시뮬 env 와 실차 ROS 노드가 **같이 import** 하는 관측 생성 코드 (numpy 전용).

여기를 고치면 sim·실차가 동시에 바뀐다. 한쪽만 따로 전처리하지 말 것.

  preprocess_scan()   원본 LaserScan ranges → 고정 격자 n_beams, 0~1
  IntervalAverager    100 Hz IMU / 50 Hz 속도 → 스캔 구간 평균
  ObsHistory          스캔 4장 + 상태 17 → Dict 관측
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import STATE_DIM, LidarSpec, NormSpec, ActuatorSpec, ActionSpec


def beam_angles(spec: LidarSpec) -> np.ndarray:
    """차량 기준 빔 각도 [rad], 오른쪽(−135°) → 왼쪽(+135°), 반시계 +."""
    half = spec.fov / 2.0
    return np.linspace(-half, half, spec.n_beams).astype(np.float64)


def preprocess_scan(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    spec: LidarSpec,
    range_min: float | None = None,
) -> np.ndarray:
    """원본 스캔 → (n_beams,) float32 ∈ [0, 1].

    - 원본 빔 수와 무관하게 고정 각도 격자로 재배치 (20 Hz 2250점이든 40 Hz 1125점이든)
    - 같은 격자칸에 여러 빔이 오면 **최소값** (구멍 난 벽이 벽으로 남도록)
    - 무반사(0, inf, NaN, < range_min) → range_max
    - 빈 칸은 이웃 칸 값으로 채움
    """
    r = np.asarray(ranges, dtype=np.float64).copy()
    rmin = spec.range_min if range_min is None else max(range_min, spec.range_min)
    bad = ~np.isfinite(r) | (r < rmin)
    r[bad] = spec.range_max
    r = np.minimum(r, spec.range_max)

    ang = angle_min + angle_increment * np.arange(r.size) + spec.mount_yaw
    ang = (ang + np.pi) % (2.0 * np.pi) - np.pi

    half = spec.fov / 2.0
    inc = spec.angle_inc
    idx = np.rint((ang + half) / inc).astype(np.int64)
    ok = (idx >= 0) & (idx < spec.n_beams)

    out = np.full(spec.n_beams, np.inf)
    np.minimum.at(out, idx[ok], r[ok])
    empty = ~np.isfinite(out)
    if empty.all():
        out[:] = spec.range_max
    elif empty.any():
        xs = np.arange(spec.n_beams)
        out[empty] = np.interp(xs[empty], xs[~empty], out[~empty])
    return (out / spec.range_max).astype(np.float32)


def invalid_bins(ranges: np.ndarray, angle_min: float, angle_increment: float,
                 mount_yaw: float, n_bins: int = 36, near: float = 0.35) -> np.ndarray:
    """차량 기준 각도 n_bins 칸별 '무효 빔' 비율 (0~1). 칸 0 = −180°.

    무효 = 무반사(inf/0/NaN) 또는 near 보다 가까움(차체).  LPX-T1 은 뒤쪽 ~90° 가
    차체·하우징에 가려 늘 무효라서, 정지 상태 스캔 몇십 장을 평균하면 가장 긴 무효
    구간의 중심이 차 뒤(±180°)에 와야 한다.  정면(0°) 근처면 라이다 방향 설정이 틀린 것.
    """
    r = np.asarray(ranges, dtype=np.float64)
    bad = ~np.isfinite(r) | (r <= 0.0) | (r < near)
    ang = angle_min + angle_increment * np.arange(r.size) + mount_yaw
    ang = (ang + np.pi) % (2.0 * np.pi) - np.pi
    b = np.clip(((ang + np.pi) / (2 * np.pi) * n_bins).astype(np.int64), 0, n_bins - 1)
    tot = np.bincount(b, minlength=n_bins).astype(np.float64)
    nb = np.bincount(b, weights=bad.astype(np.float64), minlength=n_bins)
    frac = np.ones(n_bins)                     # 빔이 아예 없는 칸 = 무효 (FOV 밖)
    has = tot > 0
    frac[has] = nb[has] / tot[has]
    return frac


def blocked_center_deg(frac: np.ndarray, thresh: float = 0.8) -> tuple[float, float]:
    """invalid_bins 평균 → (가장 긴 연속 무효 구간 중심각 [deg], 폭 [deg]). 없으면 (nan, 0)."""
    n = frac.size
    m = frac >= thresh
    if not m.any():
        return float("nan"), 0.0
    if m.all():
        return float("nan"), 360.0
    start = int(np.argmin(m))                  # 유효 칸에서 시작해 원형으로 훑음
    best_len, best_mid, run = 0, 0.0, 0
    for k in range(1, n + 1):
        i = (start + k) % n
        if m[i]:
            run += 1
            if run > best_len:
                best_len, best_mid = run, (start + k) - (run - 1) / 2.0
        else:
            run = 0
    w = 360.0 / n
    center = ((best_mid % n) + 0.5) * w - 180.0
    return float((center + 180.0) % 360.0 - 180.0), best_len * w


class IntervalAverager:
    """타임스탬프가 붙은 샘플을 쌓고 [t0, t1] 구간 평균(사다리꼴)을 낸다."""

    def __init__(self, maxlen: int = 64, default: float = 0.0):
        self.buf: deque[tuple[float, float]] = deque(maxlen=maxlen)
        self.last = float(default)

    def push(self, t: float, value: float) -> None:
        self.buf.append((float(t), float(value)))
        self.last = float(value)

    def latest(self) -> float:
        return self.last

    def mean(self, t0: float, t1: float) -> float:
        pts = [(t, v) for t, v in self.buf if t0 < t <= t1 + 1e-9]
        # 오래된 샘플 정리 (t0 이전 1개는 경계 보간용으로 남김)
        while len(self.buf) > 2 and self.buf[1][0] <= t0:
            self.buf.popleft()
        if not pts:
            return self.last
        if len(pts) == 1:
            return pts[0][1]
        t = np.array([p[0] for p in pts])
        v = np.array([p[1] for p in pts])
        span = t[-1] - t[0]
        if span <= 1e-9:
            return float(v.mean())
        return float(np.trapezoid(v, t) / span) if hasattr(np, "trapezoid") else float(np.trapz(v, t) / span)


class ObsHistory:
    """최근 hist 개 스캔 구간의 관측을 보관하고 Dict 관측을 만든다.

    state 벡터 (17) 순서 — 정책·노드 공용, 바꾸면 재학습 필요:
      [0:4]   v̄_{n-4..n-1}      / v_scale
      [4:8]   ω̄_{n-4..n-1}      / w_scale (clip ±1)
      [8]     ω_latest           / w_scale (clip ±1)
      [9:11]  (δ, v_cmd)_{n-2}   δ/δ_max, v_cmd/v_scale
      [11:13] (δ, v_cmd)_{n-1}
      [13:17] (Δt − 0.025)/0.025 for n-4..n-1
    """

    def __init__(self, lidar: LidarSpec, norm: NormSpec, act: ActuatorSpec,
                 action: ActionSpec, hist: int = 4):
        self.lidar, self.norm, self.act, self.action = lidar, norm, act, action
        self.hist = hist
        self.scans: deque[np.ndarray] = deque(maxlen=hist)
        self.v: deque[float] = deque(maxlen=hist)
        self.w: deque[float] = deque(maxlen=hist)
        self.dt: deque[float] = deque(maxlen=hist)
        self.cmds: deque[tuple[float, float]] = deque(maxlen=2)
        self.w_latest = 0.0

    def reset(self, scan: np.ndarray, v: float, w: float,
              cmd: tuple[float, float] | None = None) -> None:
        self.scans.clear(); self.v.clear(); self.w.clear(); self.dt.clear(); self.cmds.clear()
        for _ in range(self.hist):
            self.push(scan, v, w, w, self.norm.dt_nominal)
        c = cmd if cmd is not None else (0.0, v)
        self.cmds.extend([c, c])

    def push(self, scan: np.ndarray, v_mean: float, w_mean: float,
             w_latest: float, dt: float) -> None:
        self.scans.append(np.asarray(scan, dtype=np.float32))
        self.v.append(float(v_mean))
        self.w.append(float(w_mean))
        self.dt.append(float(dt))
        self.w_latest = float(w_latest)

    def push_cmd(self, steer: float, v_cmd: float) -> None:
        self.cmds.append((float(steer), float(v_cmd)))

    @property
    def ready(self) -> bool:
        return len(self.scans) == self.hist

    def state(self) -> np.ndarray:
        n = self.norm
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[0:4] = np.asarray(self.v) / n.v_scale
        s[4:8] = np.clip(np.asarray(self.w) / n.w_scale, -1.0, 1.0)
        s[8] = np.clip(self.w_latest / n.w_scale, -1.0, 1.0)
        c = list(self.cmds)
        s[9] = c[0][0] / self.act.steer_max
        s[10] = c[0][1] / n.v_scale
        s[11] = c[1][0] / self.act.steer_max
        s[12] = c[1][1] / n.v_scale
        s[13:17] = np.clip((np.asarray(self.dt) - n.dt_nominal) / n.dt_nominal, -2.0, 4.0)
        return s

    def scan_stack(self) -> np.ndarray:
        return np.stack(list(self.scans), axis=0).astype(np.float16)

    def observation(self, priv: np.ndarray | None = None) -> dict[str, np.ndarray]:
        obs = {"scan": self.scan_stack(), "state": self.state()}
        if priv is not None:
            obs["priv"] = np.asarray(priv, dtype=np.float32)
        return obs


def action_to_command(a: np.ndarray, act: ActuatorSpec, action: ActionSpec) -> tuple[float, float]:
    """정책 출력 a ∈ [−1, 1]² → (δ_cmd [rad], v_cmd [m/s])."""
    a0 = float(np.clip(a[0], -1.0, 1.0))
    a1 = float(np.clip(a[1], -1.0, 1.0))
    steer = a0 * act.steer_max
    v_cmd = action.v_min + 0.5 * (a1 + 1.0) * (action.v_max - action.v_min)
    return steer, v_cmd
