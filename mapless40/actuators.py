"""
mapless40.actuators
===================
명령(δ_cmd, v_cmd) → Single-Track 입력(조향각속도 sv, 종가속 accl).

실차 구성 반영:
  - 서보: dead time + 1차 지연 + 각속도 제한
  - 구동: dead time + 속도 P 제어 + 가속·저크 제한
          AUTO 모드 duty 하한 0 → **능동 제동 없음**, 감속은 타력(c0 + c1·v)뿐
파라미터는 전부 config.ActuatorSpec 에서 오며 [측정 필요].
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import ActuatorSpec


class _DeadTime:
    def __init__(self, delay_s: float, dt: float, init: float):
        n = max(0, int(round(delay_s / dt)))
        self.q: deque[float] = deque([init] * n, maxlen=n) if n > 0 else None
        self.n = n

    def __call__(self, x: float) -> float:
        if self.n == 0:
            return x
        out = self.q[0]
        self.q.append(x)
        return out


class ServoModel:
    def __init__(self, spec: ActuatorSpec, dt: float, delta0: float = 0.0):
        self.s, self.dt = spec, dt
        self.delay = _DeadTime(spec.servo_dead_time, dt, delta0)

    def steer_velocity(self, delta_cmd: float, delta: float) -> float:
        s = self.s
        target = float(np.clip(self.delay(delta_cmd), -s.steer_max, s.steer_max))
        sv = (target - delta) / max(s.servo_tau, 1e-3)
        return float(np.clip(sv, -s.servo_rate_max, s.servo_rate_max))


class DriveModel:
    def __init__(self, spec: ActuatorSpec, dt: float, v0: float = 0.0):
        self.s, self.dt = spec, dt
        self.delay = _DeadTime(spec.drive_dead_time, dt, v0)
        self.a_prev = 0.0

    def accel(self, v_cmd: float, v: float) -> float:
        s = self.s
        vc = self.delay(v_cmd)
        a_req = s.speed_kp * (vc - v)
        coast = -(s.coast_decel_c0 + s.coast_decel_c1 * max(v, 0.0))
        if a_req >= 0.0:
            a = min(a_req, s.accel_max)
        else:
            if s.brake_enabled:
                a = max(a_req, -s.brake_decel_max)
            else:
                # 모터 토크 0 → 타력 감속만. 요구 감속이 더 작으면 그만큼만.
                a = max(a_req, coast)
        # 저크 제한 (duty rate limit 근사)
        da = s.jerk_max * self.dt
        a = float(np.clip(a, self.a_prev - da, self.a_prev + da))
        if v <= 0.05 and a < 0.0:
            a = 0.0   # 후진 방지
        self.a_prev = a
        return a
