"""
depthfly.config
===============
Gemini 2L depth 한 대 + 초파리 커넥톰. 카메라·장면·차량 설정은 camfly 와 같다.

  - 30 Hz: Gemini 2L depth 는 1280×800 / 640×400 에서 최대 30 fps (40 fps 모드 없음).
           depth 계산은 카메라 칩(MX6600)이 하고, Jetson 은 16×64 재샘플링 + 회로(≈0.5 ms)만 → 부하 거의 없음.
  - 과거 3프레임 (n-2, n-1, n ≈ 100 ms).
"""

from __future__ import annotations

from camfly.config import CamFlyConfig, CameraSpec, SceneSpec, state_dim  # noqa: F401


def DepthFlyConfig(fps: float = 30.0, hist: int = 3) -> CamFlyConfig:  # noqa: N802
    return CamFlyConfig(fps=fps, hist=hist)
