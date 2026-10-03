"""
depthfly.config
===============
Gemini 2L depth 한 대 + 초파리 커넥톰. 카메라·장면·차량 설정은 camfly 와 같다.

  - 30 Hz: Gemini 2L depth 는 Unbinned 모드 (최적 0.30~7.0 m) 에서 최대 30 fps.
           Binned Sparse 모드는 640×400 / 320×200 에서 60 fps 가능하지만 최적 범위 0.25~5.0 m (데이터시트 v1.0).
           depth 계산은 카메라 칩(MX6600)이 하고, Jetson 은 16×64 재샘플링 + 회로(≈0.5 ms)만 → 부하 거의 없음.
  - 과거 3프레임 (n-2, n-1, n ≈ 100 ms).
  - 행 배치: 덕트(33 cm)는 3 m 밖에서 지평선 ±3° 안에만 보인다 → 16행 중 12행을 지평선 ±8° 에 몰아줌
           (균일 배치일 때 값이 있는 행이 사실상 1개였음).
"""

from __future__ import annotations

from camera.camfly.config import CamFlyConfig, CameraSpec, SceneSpec, state_dim  # noqa: F401


def DepthFlyConfig(fps: float = 30.0, hist: int = 3) -> CamFlyConfig:  # noqa: N802
    return CamFlyConfig(cam=DepthCameraSpec(), fps=fps, hist=hist)


def DepthCameraSpec(**kw) -> CameraSpec:  # noqa: N802
    kw.setdefault("row_acute_half_deg", 8.0)
    return CameraSpec(**kw)
