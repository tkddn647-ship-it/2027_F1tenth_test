"""
lidarfly.config
===============
LiDAR 1080빔, 40 Hz, 속도 2~7 m/s. mapless40.EnvConfig 를 복사해 이 패키지에서만 바꾼다.
mapless40 기본값(1125빔, hist 4, v_max 5)은 그대로 둔다.

출처 구분:
  실측     — 차량에서 잰 값 (Roboracer-2026-main 주석이 실측이라고 적은 것)
  펌웨어   — ESP 루프 파라미터로 계산. 서보 스텝 응답을 잰 값은 아니다
  사용자 지정 — 이 작업에서 지정한 센서·속도
  추정     — 실차 코드에서 m/s²·데드타임을 못 찾아 mapless40 추정값을 유지
"""

from __future__ import annotations

import math

from mapless40.config import EnvConfig

# 사용자 지정: FOV 270°, 빔 1080 (균일 격자). 데이터시트 각도 분해능으로 역산한 값이 아니다.
N_BEAMS = 1080
FOV_DEG = 270.0
DT_SCAN = 0.025          # 40 Hz
HIST = 3                 # 현재 + 과거 2
V_MIN = 2.0
V_MAX = 7.0              # 사용자 지정. 실차 최고 속도 7 m/s 는 사용자 확인(2026-10-03)

# 실측 전륜각. control_node.py max_steering_angle_rad = 0.3735
# (20260816 요레이트 실측, S=1 → 약 21.4°). stanley max_steering_angle_real_rad 와 동일.
STEER_MAX_RAD = 0.3735

# 펌웨어 1차 지연. stanley_waypoint_follow_node.py:
#   esp_smooth_factor=0.20, esp_loop_period_sec=0.020
#   τ = -dt / ln(1 - a). 주석의 "시정수 90 ms" 와 같다. 스텝 응답 실측은 아님.
_ESP_SMOOTH = 0.20
_ESP_LOOP_S = 0.020
SERVO_TAU_S = -_ESP_LOOP_S / math.log(1.0 - _ESP_SMOOTH)

DEFAULT_MAPS = "ifac:3,roboracer_0817:3,random:2"


def state_dim(hist: int = HIST) -> int:
    """v̄×hist, ω̄×hist, ω_latest, (δ, v_cmd)×2, Δt×hist."""
    return 3 * hist + 5


def lidarfly_config() -> EnvConfig:
    """mapless40 기본 EnvConfig 를 복사한 뒤 lidarfly 값만 덮어쓴다."""
    c = EnvConfig()
    c.lidar.fov_deg = FOV_DEG          # 사용자 지정
    c.lidar.n_beams = N_BEAMS          # 사용자 지정
    c.timing.dt_scan = DT_SCAN         # 40 Hz
    c.timing.hist = HIST
    c.norm.dt_nominal = DT_SCAN
    c.action.v_min = V_MIN
    c.action.v_max = V_MAX
    c.raceline.v_min = V_MIN           # 레이싱라인 속도는 행동 범위와 같게 (학습 신호만)
    c.raceline.v_max = V_MAX
    c.act.steer_max = STEER_MAX_RAD    # 실측 ±21.4°

    # 서보 시정수: 펌웨어 SMOOTH_FACTOR 로 계산 (위 SERVO_TAU_S). 스텝 응답 실측 아님.
    c.act.servo_tau = float(SERVO_TAU_S)
    # 순수 데드타임[s] 은 control_node·yaml 에 없다 → mapless40 추정.
    c.act.servo_dead_time = 0.015      # 추정
    # Stanley steering_rate_limit_radps=6.5 는 서보각 단위 소프트웨어 상한이고,
    # 보정 시 전륜각으로 6.5*(0.3735/0.8727)≈2.8 rad/s 로 줄어든다. 기구 각속도 실측이
    # 아니라 mapless40 의 보수적 추정 4.0 rad/s 를 유지한다.
    c.act.servo_rate_max = 4.0         # 추정

    # 가감속 [m/s²]: control_node 의 speed_kp=0.15 는 duty PI 게인이라 단위가 다르다.
    # speed_profile.py 의 a_accel 7 / a_brake 4 는 파일 주석이 "TODO 실측" 인 임시값이라
    # 쓰지 않는다. mapless40 추정값을 유지.
    c.act.accel_max = 5.0              # 추정
    c.act.brake_decel_max = 4.0        # 추정. brake_enabled=False 이면 미사용
    c.act.coast_decel_c0 = 0.6         # 추정 (타력 감속 c0 + c1·v)
    c.act.coast_decel_c1 = 0.15        # 추정
    c.act.drive_dead_time = 0.030      # 추정. 구동 지연 초는 실차 코드에 없음
    c.act.speed_kp = 4.0               # 추정 [(v_cmd-v)→가속, 1/s]. duty 게인과 다름
    c.act.jerk_max = 40.0              # 추정. control_node duty_rate_limit 0.6/s 의 mapless40 근사
    # 능동 브레이크 없음: control_node AUTO 속도 PI duty 하한 0 (코드 사실, 감속 크기 실측 아님).
    c.act.brake_enabled = False
    return c
