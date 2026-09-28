"""
mapless40.config
================
40 Hz mapless asymmetric-SAC 설정값 한곳 모음.

값 옆 [측정 필요] 표시는 실차에서 식별해야 하는 파라미터다.
기본값은 Roboracer-2026-main 설정·로그와 LPX-T1 데이터시트에서 가져온 추정치.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass
class LidarSpec:
    """SLAMTEC LPX-T1 @ 40 Hz (60 kHz / 40 Hz = 1500 pts/rev → 0.24°)."""

    fov_deg: float = 270.0
    n_beams: int = 1125            # 270° / 0.24°  (실차 len(ranges) 확인 후 맞출 것)
    range_max: float = 15.0        # 반사율 10% 한계 15 m → 여기서 자르고 /15 정규화
    range_min: float = 0.05
    noise_std: float = 0.02        # 데이터시트 ±20–30 mm
    dropout_prob: float = 0.005    # 무반사 빔 비율 (→ range_max 로 채움)
    mount_x: float = 0.139         # CG 기준 전방 오프셋 [m] (base_link 0.31 − lr 0.171)
    mount_yaw: float = 0.0         # 실차 스캔 0° 가 차 정면이 아니면 보정 [측정 필요]

    @property
    def fov(self) -> float:
        return math.radians(self.fov_deg)

    @property
    def angle_inc(self) -> float:
        return self.fov / (self.n_beams - 1)


@dataclass
class TimingSpec:
    dt_scan: float = 0.025         # LiDAR 40 Hz = 제어 주기
    phys_dt: float = 0.005         # 동역학 서브스텝
    imu_every: int = 2             # 100 Hz  (서브스텝 2개마다)
    speed_every: int = 4           # 50 Hz VESC
    hist: int = 4                  # 스캔 n-4 … n-1
    compute_latency: float = 0.010  # 스캔 완료 → 명령 적용까지 [측정 필요] (보수적으로 10 ms)
    scan_jitter_prob: float = 0.05  # 틱 길이가 ±5 ms 흔들릴 확률 (Ethernet/UDP)
    scan_drop_prob: float = 0.01    # 스캔 누락 → 명령 2틱 유지

    @property
    def substeps(self) -> int:
        return int(round(self.dt_scan / self.phys_dt))


@dataclass
class ActuatorSpec:
    """서보·구동 응답. 전부 [측정 필요] — 지면 step 테스트로 식별할 것."""

    steer_max: float = 0.3735      # 실측 전륜각 ±21.4°
    # 모르는 값은 실차보다 불리하게(느리게) 잡는다 → 실차가 더 좋으면 여유가 남는다
    servo_dead_time: float = 0.015
    servo_tau: float = 0.050       # 1차 지연 시정수
    servo_rate_max: float = 4.0    # rad/s (Stanley 설정 6.5, f1tenth_gym 3.2 사이에서 느린 쪽)
    drive_dead_time: float = 0.030
    speed_kp: float = 4.0          # (v_cmd − v) → 가속 요구 [1/s]
    accel_max: float = 5.0         # speed_profile a_accel 7 보다 보수적
    jerk_max: float = 40.0         # duty rate limit 0.6/s 근사
    # 실차 AUTO 속도 PI 는 duty 하한 0 → 능동 제동 없음 (control_node.py:767)
    brake_enabled: bool = False
    brake_decel_max: float = 4.0   # brake_enabled 일 때만
    coast_decel_c0: float = 0.6    # 타력 감속 = c0 + c1·v  [측정 필요]
    coast_decel_c1: float = 0.15


@dataclass
class ImuSpec:
    gyro_noise_std: float = 0.02   # rad/s
    gyro_bias_std: float = 0.01    # 에피소드마다 샘플
    speed_noise_std: float = 0.03  # m/s


@dataclass
class ActionSpec:
    v_min: float = 2.0
    v_max: float = 5.0             # 간단 버전: 2~5 m/s


@dataclass
class NormSpec:
    """관측 정규화 상수 — sim·실차 노드가 반드시 같은 값을 써야 한다."""

    v_scale: float = 7.0
    w_scale: float = 3.0
    dt_nominal: float = 0.025


@dataclass
class RacelineSpec:
    a_lat: float = 5.0             # 실차 한계 6.0 에 여유 (μ 랜덤화 포함)
    a_accel: float = 4.0
    a_brake: float = 1.8           # 실차 타력 감속 수준에 맞춤 [측정 필요]
    v_max: float = 5.0
    v_min: float = 2.0
    lookahead_n: int = 10
    lookahead_ds: float = 1.0
    wall_margin: float = 0.45      # 라인을 벽에서 최소 이만큼 떨어뜨림 (반폭 0.15 + 여유)
    smooth_m: float = 1.0          # 라인 좌표 스무딩 창 [m]
    kappa_smooth_m: float = 2.0    # 곡률 스무딩 창 [m]
    # 레이싱라인 CSV 가 없고 센터라인만 있는 맵은 최소곡률 근사 라인으로 바꿔 쓴다.
    # (센터라인 기준이면 ifac 이론 랩타임이 실측보다 4~5 s 느리게 나옴)
    optimize_centerline: bool = True


@dataclass
class RewardSpec:
    w_progress: float = 1.0        # Δs [m] 당
    w_ey: float = 0.5              # e_y² · Δt
    w_v: float = 0.08              # (v − v_ref)² · Δt  (과속은 ×over_mult)
    over_mult: float = 3.0
    w_dsteer: float = 0.1          # |δ_n − δ_{n−1}| / δ_max (크면 초반 탐색 억제)
    w_slip: float = 2.0            # max(|β|−0.05, 0) · Δt
    collision: float = -20.0
    lap_bonus: float = 20.0
    reverse_s: float = 1.0         # 이만큼 연속 후진(Δs<0)하면 종료 [s]


@dataclass
class ObstacleSpec:
    """정적 장애물 (콘·박스 크기 원통). 에피소드마다 무작위, 한쪽은 반드시 지나갈 틈을 남긴다."""

    prob: float = 0.7              # 이 확률로 장애물 있는 에피소드
    max_n: int = 3                 # 1 ~ max_n 개
    r_range: tuple[float, float] = (0.12, 0.30)   # 반지름 [m]
    min_gap: float = 0.80          # 장애물 옆 통과 폭 최소 (차폭 0.30 + 넉넉한 여유)
    min_ahead: float = 6.0         # 스폰 지점에서 최소 이만큼 앞에
    min_sep: float = 2.0           # 장애물끼리 라인 방향 최소 간격 [m]
    lat_range: float = 0.6         # 레이싱라인 기준 좌우 배치 범위 [m]


@dataclass
class EnvConfig:
    lidar: LidarSpec = field(default_factory=LidarSpec)
    timing: TimingSpec = field(default_factory=TimingSpec)
    act: ActuatorSpec = field(default_factory=ActuatorSpec)
    imu: ImuSpec = field(default_factory=ImuSpec)
    action: ActionSpec = field(default_factory=ActionSpec)
    norm: NormSpec = field(default_factory=NormSpec)
    raceline: RacelineSpec = field(default_factory=RacelineSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    obstacles: ObstacleSpec = field(default_factory=ObstacleSpec)
    max_episode_s: float = 60.0
    # 선형 타이어 ST 모델에서 μ 는 코너링 강성 배율일 뿐 횡력 한계가 아니다.
    # 기존 STParams.roboracer() 의 μ = a_lat/g ≈ 0.61 은 과도한 언더스티어만 만들고
    # 횡가속은 제한하지 못한다 → f1tenth_gym 식별값 μ 를 쓰고, 한계는 a_lat_cap 으로 따로 건다.
    mu: float | None = 1.0489
    mu_rand: float = 0.10          # 에피소드마다 μ × U(1−r, 1+r)
    a_lat_cap: float = 6.0         # 마찰 한계 근사 |v·ω| ≤ cap [측정 필요] (보수적으로 실차 스펙 6)
    a_lat_cap_rand: float = 0.10
    spawn_lat_std: float = 0.15
    spawn_heading_std: float = 0.08
    spawn_v_range: tuple[float, float] = (0.5, 3.0)
    half_width: float = 0.15       # 차체
    front: float = 0.329           # CG → 앞끝 (base_link 0.50 − lr)
    rear: float = 0.271            # CG → 뒤끝

    def to_dict(self) -> dict:
        return asdict(self)


# 관측 차원 (정책·노드 공용)
STATE_DIM = 17   # v̄×4, ω̄×4, ω_latest, (δ,v)_{n-2}, (δ,v)_{n-1}, Δt×4
PRIV_DIM = 27    # e_y, e_ψ, v−v_ref, β, κ×10, v_ref×10, 앞 장애물(거리, 옆 오프셋, 반지름)
GAMMA_40HZ = 0.99 ** 0.25
