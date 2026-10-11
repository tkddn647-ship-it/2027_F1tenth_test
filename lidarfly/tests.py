"""
lidarfly.tests
==============
  python -m lidarfly.tests
"""

from __future__ import annotations

import math
import sys
import time
import traceback

import numpy as np

from mapless40.config import PRIV_DIM, EnvConfig, LidarSpec

from .config import (
    FOV_DEG, HIST, N_BEAMS, SERVO_TAU_S, STEER_MAX_RAD, V_MAX, V_MIN,
    lidarfly_config, state_dim,
)
from .env import LidarFlyEnv, parse_maps
from .randmap import make_random_track


def test_mapless40_defaults_unchanged():
    """옛 모델 호환: mapless40 기본 빔·히스토리·속도는 그대로."""
    base = EnvConfig()
    assert LidarSpec().n_beams == 1125
    assert base.lidar.n_beams == 1125
    assert base.timing.hist == 4
    assert base.action.v_max == 5.0
    assert base.timing.dt_scan == 0.025


def test_lidarfly_config_sources():
    c = lidarfly_config()
    assert c.lidar.n_beams == N_BEAMS == 1080
    assert c.lidar.fov_deg == FOV_DEG == 270.0
    assert abs(c.timing.dt_scan - 0.025) < 1e-12
    assert c.timing.hist == HIST == 3
    assert c.action.v_min == V_MIN == 2.0 and c.action.v_max == V_MAX == 7.0
    assert c.raceline.v_min == 2.0 and c.raceline.v_max == 7.0
    assert abs(c.act.steer_max - STEER_MAX_RAD) < 1e-12
    assert abs(math.degrees(c.act.steer_max) - 21.4) < 0.05
    assert abs(c.act.servo_tau - SERVO_TAU_S) < 1e-12
    assert 0.085 < c.act.servo_tau < 0.095          # 펌웨어 계산 ≈ 90 ms
    assert c.act.brake_enabled is False
    assert c.norm.dt_nominal == 0.025
    # 호출마다 독립 복사 (mapless40 기본 객체를 공유하지 않음)
    other = lidarfly_config()
    other.lidar.n_beams = 1
    assert c.lidar.n_beams == 1080


def test_parse_weights():
    names, w = parse_maps("ifac:3,roboracer_0817:3,random:2")
    assert names == ["ifac", "roboracer_0817", "random"]
    assert w == [3.0, 3.0, 2.0]


def test_random_track_raceline():
    cfg = lidarfly_config()
    tr = make_random_track(cfg, np.random.default_rng(0))
    assert tr.name == "random"
    assert tr.line.length > 20.0
    assert tr.line.v_ref.shape == (tr.line.n,)
    assert np.isfinite(tr.line.v_ref).all()
    assert tr.line.v_ref.max() <= cfg.raceline.v_max + 1e-6
    assert tr.line.v_ref.min() >= cfg.raceline.v_min - 1e-6
    d = tr.grid.distance(tr.line.x, tr.line.y)
    assert float(d.min()) > 0.3, float(d.min())
    assert tr.grid.occ.any() and (~tr.grid.occ).any()


def test_env_three_maps_obs_shape():
    maps = "ifac:3,roboracer_0817:3,random:2"
    env = LidarFlyEnv(maps=maps, seed=0, sensor_noise=False, randomize=False)
    assert [t.name for t in env.tracks] == ["ifac", "roboracer_0817", "random"]
    assert np.allclose(env.track_p, np.array([3, 3, 2], np.float64) / 8.0)
    sd = state_dim(3)
    assert sd == 14
    for name in ("ifac", "roboracer_0817", "random"):
        obs, info = env.reset(seed=1, options={
            "map": name, "s0": 1.0, "lat": 0.0, "dyaw": 0.0, "v0": 2.0, "n_obstacles": 0,
        })
        assert info["map"] == name
        assert obs["scan"].shape == (3, 1080)
        assert obs["scan"].dtype == np.float16
        assert obs["state"].shape == (sd,)
        assert obs["priv"].shape == (PRIV_DIM,)
        assert obs["scan"].size + obs["state"].size == 3 * 1080 + sd
        for k, sp in env.observation_space.spaces.items():
            assert obs[k].shape == sp.shape and obs[k].dtype == sp.dtype, k
        assert np.isfinite(obs["state"]).all() and np.isfinite(obs["priv"]).all()
        obs2, rew, term, trunc, info2 = env.step(np.array([0.0, 0.0], np.float32))
        assert obs2["scan"].shape == (3, 1080)
        assert obs2["scan"].size + obs2["state"].size == 3 * 1080 + sd
        assert np.isfinite(rew)
        assert info2["map"] == name
        # 최신 스캔이 마지막 프레임. 첫 step 뒤에는 스택이 한 칸 밀린다.
        assert obs2["scan"].shape[0] == 3


def main():
    tests = [
        test_mapless40_defaults_unchanged,
        test_lidarfly_config_sources,
        test_parse_weights,
        test_random_track_raceline,
        test_env_three_maps_obs_shape,
    ]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
