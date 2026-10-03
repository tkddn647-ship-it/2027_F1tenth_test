"""
depthfly.config
===============
Gemini 2L depth 한 대 + 초파리 커넥톰. 카메라·장면·차량 설정은 camfly 와 같다.

  - 30 Hz: Gemini 2L depth 는 Unbinned 모드 (최적 0.30~7.0 m) 에서 최대 30 fps.
           Binned Sparse 모드는 640×400 / 320×200 에서 60 fps 가능하지만 최적 범위 0.25~5.0 m (데이터시트 v1.0).
           depth 계산은 카메라 칩(MX6600)이 하고, Jetson 은 16×64 재샘플링 + 회로(≈0.5 ms)만 → 부하 거의 없음.
  - 과거 3프레임 (n-2, n-1, n ≈ 100 ms).
  - 속도 2~8 m/s (V_MAX). 타력 감속만으로는 7 m 시야 안에서 못 줄이므로 고속은 --brake 와 같이 봐야 한다.
  - 행 배치: 덕트(33 cm)는 3 m 밖에서 지평선 ±3° 안에만 보인다 → 16행 중 12행을 지평선 ±8° 에 몰아줌
           (균일 배치일 때 값이 있는 행이 사실상 1개였음).
"""

from __future__ import annotations

from camera.camfly.config import CamFlyConfig, CameraSpec, SceneSpec, state_dim  # noqa: F401


V_MAX = 8.0          # depthfly 속도 범위 2~8 m/s (mapless40 은 2~5). 학습·평가·실차 노드가 같은 값을 써야 한다


def DepthFlyConfig(fps: float = 30.0, hist: int = 3, v_max: float = V_MAX, brake: bool = False) -> CamFlyConfig:  # noqa: N802
    cf = CamFlyConfig(cam=DepthCameraSpec(), fps=fps, hist=hist)
    cf.env.action.v_max = cf.env.raceline.v_max = float(v_max)
    cf.env.act.brake_enabled = bool(brake)           # 능동 브레이크 (실차 control_node 는 지금 타력 감속만)
    return cf


def add_cfg_args(p) -> None:
    """학습·평가·시각화 공통 인자. 모델과 같은 값으로 평가해야 한다 (행동의 속도 축척이 v_max 에 달려 있음)."""
    p.add_argument("--v-max", type=float, default=V_MAX, help="최고 속도 [m/s] (최저 2)")
    p.add_argument("--brake", action="store_true", help="능동 브레이크 4 m/s² (기본: 타력 감속만)")


def DepthCameraSpec(**kw) -> CameraSpec:  # noqa: N802
    kw.setdefault("row_acute_half_deg", 8.0)
    return CameraSpec(**kw)
