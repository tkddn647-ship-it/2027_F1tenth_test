"""
mapless40.tests
===============
  python -m mapless40.tests            # numpy 테스트 + (torch/SB3 있으면) 정책 테스트
  python -m mapless40.tests --quick    # 느린 주행 테스트 제외

pytest 없이도 돈다. 실패하면 AssertionError 로 멈춘다.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from .config import STATE_DIM, PRIV_DIM, EnvConfig, LidarSpec
from .obs_builder import IntervalAverager, ObsHistory, action_to_command, beam_angles, preprocess_scan

CFG = EnvConfig()


# ------------------------------------------------------------------ numpy
def test_preprocess_scan_grid_and_min():
    sp = LidarSpec()
    # 20 Hz 급 원본 (0.12°, 2250 + 여분) → 1125 격자, 같은 칸은 최소값
    n = 3000
    amin, inc = -np.pi, 2 * np.pi / n
    r = np.full(n, 10.0)
    r[1500] = 2.0                       # 정면 한 빔만 가까움 (구멍 난 벽의 벽 빔)
    r[10] = np.inf; r[11] = 0.0; r[12] = np.nan
    s = preprocess_scan(r, amin, inc, sp)
    assert s.shape == (sp.n_beams,) and s.dtype == np.float32
    assert 0.0 <= s.min() and s.max() <= 1.0
    mid = sp.n_beams // 2
    assert abs(s[mid] - 2.0 / sp.range_max) < 1e-6, "같은 칸의 최소값이 남아야 함"
    assert abs(s[0] - 10.0 / sp.range_max) < 1e-6


def test_preprocess_scan_orientation():
    sp = LidarSpec()
    n = 1500
    amin, inc = -np.pi, 2 * np.pi / n
    ang = amin + inc * np.arange(n)
    r = np.where(np.abs(ang - np.pi / 2) < 0.05, 1.0, 12.0)   # 왼쪽(+90°)에만 벽
    s = preprocess_scan(r, amin, inc, sp)
    a = beam_angles(sp)
    assert s[np.argmin(np.abs(a - np.pi / 2))] < 0.1
    assert s[np.argmin(np.abs(a + np.pi / 2))] > 0.7


def test_interval_averager():
    ia = IntervalAverager()
    for k in range(11):
        ia.push(k * 0.01, 1.0 if k * 0.01 <= 0.025 else 3.0)
    m = ia.mean(0.0, 0.025)
    assert abs(m - 1.0) < 1e-9, m
    assert abs(ia.mean(0.03, 0.10) - 3.0) < 1e-9


def test_obs_history_layout():
    c = CFG
    h = ObsHistory(c.lidar, c.norm, c.act, c.action, 4)
    scan = np.linspace(0, 1, c.lidar.n_beams).astype(np.float32)
    h.reset(scan, 2.0, 0.0)
    for k in range(4):
        h.push(scan * 0 + k / 10, 1.0 + k, 0.3 * k, 0.9, 0.025 + 0.005 * k)
    h.push_cmd(0.1, 3.0)
    h.push_cmd(-0.2, 4.0)
    o = h.observation(np.zeros(PRIV_DIM, np.float32))
    st = o["state"]
    assert o["scan"].shape == (4, c.lidar.n_beams) and o["scan"].dtype == np.float16
    assert st.shape == (STATE_DIM,)
    assert np.allclose(st[0:4], np.array([1, 2, 3, 4]) / c.norm.v_scale)
    assert np.allclose(st[4:8], np.array([0, 0.3, 0.6, 0.9]) / c.norm.w_scale)
    assert np.isclose(st[8], 0.9 / c.norm.w_scale)
    assert np.isclose(st[9], 0.1 / c.act.steer_max) and np.isclose(st[12], 4.0 / c.norm.v_scale)
    assert np.allclose(st[13:17], [0, 0.2, 0.4, 0.6], atol=1e-5)
    assert np.isclose(float(o["scan"][-1, 0]), 0.3, atol=1e-3)   # 최신 프레임이 마지막


def test_action_mapping():
    s, v = action_to_command(np.array([1.0, -1.0]), CFG.act, CFG.action)
    assert np.isclose(s, CFG.act.steer_max) and np.isclose(v, CFG.action.v_min)
    s, v = action_to_command(np.array([-1.0, 1.0]), CFG.act, CFG.action)
    assert np.isclose(s, -CFG.act.steer_max) and np.isclose(v, CFG.action.v_max)


def test_drive_model_no_active_brake():
    from .actuators import DriveModel
    d = DriveModel(CFG.act, 0.005, 6.0)
    a = [d.accel(1.5, 6.0) for _ in range(100)]
    c = CFG.act
    assert min(a) >= -(c.coast_decel_c0 + c.coast_decel_c1 * 6.0) - 1e-6, "능동 제동이 없어야 함"


def _env():
    from .env import MaplessRaceEnv40
    return MaplessRaceEnv40(maps="ifac", cfg=EnvConfig(), seed=0)


def test_raycast_matches_bruteforce():
    from .raycast import cast_rays
    env = _env()
    env.reset(seed=1)
    x, y, _, _, yaw, _, _ = env.state
    ang = beam_angles(CFG.lidar)[::15]
    r = cast_rays(env.track.grid, x, y, yaw, ang, 15.0)
    bf = []
    for a in ang:
        t = 0.0
        while t < 15.0 and not env.track.grid.occupied(np.array([x + t * np.cos(yaw + a)]),
                                                       np.array([y + t * np.sin(yaw + a)]))[0]:
            t += 0.01
        bf.append(min(t, 15.0))
    err = np.abs(np.array(bf) - r)
    assert err.max() < 0.06, err.max()


def test_env_spaces_and_latency():
    env = _env()
    obs, _ = env.reset(seed=2, options={"s0": 5.0, "lat": 0.0, "dyaw": 0.0, "v0": 2.0})
    for k, sp in env.observation_space.spaces.items():
        assert obs[k].shape == sp.shape and obs[k].dtype == sp.dtype, k
    env.randomize = False
    d0 = env.state[2]
    # 명령은 compute_latency + servo dead time 뒤에야 조향이 움직이기 시작한다
    delay = CFG.timing.compute_latency + CFG.act.servo_dead_time
    n_wait = int(np.ceil((delay + 1e-9) / CFG.timing.dt_scan))
    for _ in range(n_wait):
        env.step(np.array([1.0, 0.0], np.float32))
    if delay >= CFG.timing.dt_scan:
        pass  # 지연이 한 틱 이상이면 첫 틱에는 안 움직이는 게 정상
    env.step(np.array([1.0, 0.0], np.float32))
    assert env.state[2] > d0 + 1e-3, "지연 뒤에는 조향이 움직여야 함"
    assert env.state[2] < CFG.act.steer_max * 0.9, "서보 지연/속도제한이 있어야 함"


def test_env_ftg_completes_lap(max_s: float = 40.0):
    from .evaluate import ftg_controller
    env = _env()
    env.cfg.max_episode_s = max_s
    env.randomize = False
    ctrl = ftg_controller(env.cfg)
    obs, _ = env.reset(seed=3, options={"s0": 0.0, "lat": 0.0, "dyaw": 0.0, "v0": 1.5})
    while True:
        obs, r, term, trunc, info = env.step(ctrl(obs))
        if term or trunc or info["laps"] >= 1:
            break
    assert info["laps"] >= 1 and not info["collided"], info


# ------------------------------------------------------------------ torch
def _torch_ok():
    try:
        import torch  # noqa: F401
        import stable_baselines3  # noqa: F401
        return True
    except ImportError:
        return False


def test_policy_actor_ignores_priv():
    import torch
    from stable_baselines3 import SAC
    from .policy import AsymFeatures, AsymSACPolicy
    for enc in ("conv1d", "bev"):
        env = _env()
        model = SAC(AsymSACPolicy, env, buffer_size=100, learning_starts=10, batch_size=8,
                    policy_kwargs=dict(features_extractor_class=AsymFeatures,
                                       features_extractor_kwargs=dict(encoder=enc,
                                                                      lidar_cfg=CFG.to_dict()["lidar"],
                                                                      norm_cfg=CFG.to_dict()["norm"])),
                    device="cpu")
        pol = model.policy
        assert not pol.actor.features_extractor.use_priv and pol.critic.features_extractor.use_priv
        assert pol.critic.features_extractor is not pol.actor.features_extractor
        obs, _ = env.reset(seed=0)
        o1 = {k: torch.as_tensor(v)[None] for k, v in obs.items()}
        o2 = dict(o1); o2["priv"] = o1["priv"] + 5.0
        with torch.no_grad():
            a1 = pol.actor(o1, deterministic=True); a2 = pol.actor(o2, deterministic=True)
            act = torch.zeros(1, 2)
            q1 = pol.critic(o1, act)[0]; q2 = pol.critic(o2, act)[0]
        assert torch.allclose(a1, a2), "actor 가 priv 를 보면 안 됨"
        assert not torch.allclose(q1, q2), "critic 은 priv 를 봐야 함"
        model.learn(64)   # 학습 루프 스모크
        print(f"    encoder={enc}: actor params "
              f"{sum(p.numel() for p in pol.actor.parameters()):,}")


def test_bev_raster_geometry():
    import torch
    from .config import NormSpec
    from .policy import BEVRasterizer
    sp = LidarSpec()
    ras = BEVRasterizer(sp, NormSpec(), 4)
    a = beam_angles(sp)
    r = np.full(sp.n_beams, 1.0, np.float32)
    r[np.abs(a) < 0.02] = 5.0 / sp.range_max        # 정면 5 m 에 점
    scan = torch.tensor(np.stack([r] * 4))[None]
    state = torch.zeros(1, STATE_DIM)                # 정지 → 4프레임 같은 위치
    img = ras(scan, state)[0]
    row = int((5.0 + sp.mount_x - ras.x_min) / ras.res)
    col = int(ras.y_half / ras.res)
    assert img[3, row - 1:row + 2, col - 1:col + 2].sum() > 0, "정면 5 m 점이 격자에 찍혀야 함"
    assert img[4, row - 10, col] > 0, "빔 경로는 빈공간 채널에 찍혀야 함"


def test_connectome_dense_equals_sparse():
    """connectome_rnn 수정 확인: sparse(CPU) 경로 = h@W dense 식 = numpy 참조 구현."""
    import torch
    from connectome_rnn import ConnectomeRNN, reference_forward_numpy
    rng = np.random.default_rng(0)
    n = 200
    A = ((rng.random((n, n)) < 0.05) * rng.normal(size=(n, n))).astype(np.float32)
    inp, out = np.arange(10), np.arange(190, 200)
    m = ConnectomeRNN(A, inp, out, n_obs=6, n_act=3, dt=1.0, tau=3.0)
    with torch.no_grad():
        m.W_in.bias.zero_(); m.W_out.bias.zero_()
        m.scale.uniform_(0.05, 0.3)
    xs = torch.randn(3, 6)
    h, ys = None, []
    with torch.no_grad():
        for t in range(3):
            y, h = m(xs[t:t + 1], h=h, n_steps=1)      # n=200 > 128, CPU → sparse 경로
            ys.append(y[0].numpy())
        W = m.effective_weight()
        hd = torch.zeros(1, n)
        for t in range(3):
            drive = torch.zeros(1, n); drive[:, m.input_idx] = m.W_in(xs[t:t + 1])
            hd = hd + (m.dt / m.tau) * (-hd + torch.tanh(hd @ W + drive))
    assert torch.allclose(hd, h, atol=1e-5), "dense(h@W) 와 sparse 경로가 같아야 함"
    ref = reference_forward_numpy(A, m.scale.detach().numpy(), m.W_in.weight.detach().numpy(),
                                  m.W_out.weight.detach().numpy(), inp, out, xs.numpy(),
                                  dt=1.0, tau=3.0)
    assert np.allclose(ref, np.stack(ys), atol=1e-4), "numpy 참조 구현과 같아야 함"


def main():
    quick = "--quick" in sys.argv
    tests = [test_preprocess_scan_grid_and_min, test_preprocess_scan_orientation,
             test_interval_averager, test_obs_history_layout, test_action_mapping,
             test_drive_model_no_active_brake, test_raycast_matches_bruteforce,
             test_env_spaces_and_latency]
    if not quick:
        tests.append(test_env_ftg_completes_lap)
    if _torch_ok():
        tests += [test_bev_raster_geometry, test_connectome_dense_equals_sparse,
                  test_policy_actor_ignores_priv]
    else:
        print("(torch / stable-baselines3 없음 → 정책 테스트 건너뜀)")
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
