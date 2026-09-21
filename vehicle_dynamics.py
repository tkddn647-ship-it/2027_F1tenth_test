# SPDX: BSD-3-Clause (adapted from f1tenth_gym / CommonRoad single-track)
# Original: https://github.com/f1tenth/f1tenth_gym  (Hongrui Zheng / TUM CPS)
"""F1TENTH single-track dynamic bicycle (tire slip, mu, steer-rate limits)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*_a, **_k):  # type: ignore
        def wrap(fn):
            return fn

        return wrap


# Default params: f1tenth_gym concrete-floor identification
F110_ST_PARAMS = dict(
    mu=1.0489,
    C_Sf=4.718,
    C_Sr=5.4562,
    lf=0.15875,
    lr=0.17145,
    h=0.074,
    m=3.74,
    I=0.04712,
    s_min=-0.4189,
    s_max=0.4189,
    sv_min=-3.2,
    sv_max=3.2,
    v_switch=7.319,
    a_max=9.51,
    v_min=-5.0,
    v_max=20.0,
)


@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    if vel > v_switch:
        pos_limit = a_max * v_switch / vel
    else:
        pos_limit = a_max
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.0
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit
    return accl


@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    if (steering_angle <= s_min and steering_velocity <= 0) or (
        steering_angle >= s_max and steering_velocity >= 0
    ):
        steering_velocity = 0.0
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max
    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(
    x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max
):
    lwb = lf + lr
    u0 = steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max)
    u1 = accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)
    return np.array(
        [x[3] * np.cos(x[4]), x[3] * np.sin(x[4]), u0, u1, x[3] / lwb * np.tan(x[2])]
    )


@njit(cache=True)
def vehicle_dynamics_st(
    x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max
):
    g = 9.81
    u0 = steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max)
    u1 = accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)
    u = np.array([u0, u1])

    if abs(x[3]) < 0.5:
        lwb = lf + lr
        f_ks = vehicle_dynamics_ks(
            x[0:5], u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max
        )
        return np.array(
            [
                f_ks[0],
                f_ks[1],
                f_ks[2],
                f_ks[3],
                f_ks[4],
                u[1] / lwb * np.tan(x[2]) + x[3] / (lwb * np.cos(x[2]) ** 2) * u[0],
                0.0,
            ]
        )

    return np.array(
        [
            x[3] * np.cos(x[6] + x[4]),
            x[3] * np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu * m / (x[3] * I * (lr + lf))
            * (lf**2 * C_Sf * (g * lr - u[1] * h) + lr**2 * C_Sr * (g * lf + u[1] * h))
            * x[5]
            + mu * m / (I * (lr + lf))
            * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h))
            * x[6]
            + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u[1] * h) * x[2],
            (mu / (x[3] ** 2 * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) * lr - C_Sf * (g * lr - u[1] * h) * lf) - 1)
            * x[5]
            - mu / (x[3] * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h)) * x[6]
            + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * x[2],
        ]
    )


@njit(cache=True)
def pid_speed_steer(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    steer_diff = steer - current_steer
    if abs(steer_diff) > 1e-4:
        sv = (steer_diff / abs(steer_diff)) * max_sv
    else:
        sv = 0.0

    vel_diff = speed - current_speed
    if current_speed > 0.0:
        if vel_diff > 0:
            kp = 10.0 * max_a / max_v
        else:
            kp = 10.0 * max_a / max(-min_v, 1e-3)
        accl = kp * vel_diff
    else:
        if vel_diff > 0:
            kp = 2.0 * max_a / max_v
        else:
            kp = 2.0 * max_a / max(-min_v, 1e-3)
        accl = kp * vel_diff
    return accl, sv


@dataclass
class STParams:
    mu: float = 1.0489
    C_Sf: float = 4.718
    C_Sr: float = 5.4562
    lf: float = 0.15875
    lr: float = 0.17145
    h: float = 0.074
    m: float = 3.74
    I: float = 0.04712
    s_min: float = -0.4189
    s_max: float = 0.4189
    sv_min: float = -3.2
    sv_max: float = 3.2
    v_switch: float = 7.319
    a_max: float = 9.51
    v_min: float = -5.0
    v_max: float = 20.0

    @classmethod
    def f110_default(cls) -> "STParams":
        return cls(**F110_ST_PARAMS)

    def as_tuple(self):
        return (
            self.mu,
            self.C_Sf,
            self.C_Sr,
            self.lf,
            self.lr,
            self.h,
            self.m,
            self.I,
            self.s_min,
            self.s_max,
            self.sv_min,
            self.sv_max,
            self.v_switch,
            self.a_max,
            self.v_min,
            self.v_max,
        )


def integrate_st_rk4(state: np.ndarray, u: np.ndarray, params: STParams, dt: float) -> np.ndarray:
    """One RK4 step. state = [x,y,delta,v,yaw,yaw_rate,beta]. u = [steer_vel, accel]."""
    p = params.as_tuple()

    def f(s):
        return vehicle_dynamics_st(s, u, *p)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    out = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # wrap yaw
    out[4] = (out[4] + np.pi) % (2 * np.pi) - np.pi
    return out


def lateral_accel(v: float, yaw_rate: float) -> float:
    return float(v * yaw_rate)
