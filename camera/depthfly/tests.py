"""
depthfly.tests  —  python -m camera.depthfly.tests
numpy 테스트 (어디서나) + torch 테스트 (코랩: 회로 torch == numpy, SAC 학습 루프, numpy actor == torch actor).
"""

from __future__ import annotations

import sys
import tempfile
import time
import traceback

import numpy as np

from camera.camfly.tests import _wall_grid

from .brain import feature_layout, features_numpy, forward_numpy, init_params, make_dn_wiring, n_inputs
from .config import DepthCameraSpec as CameraSpec, DepthFlyConfig, SceneSpec, state_dim
from .sensor import NEAR_REF, DepthEyeReal, DepthEyeSim, HoleHold, keep_height


def _offsets():
    o, off = 0, {}
    for k, v in feature_layout().items():
        off[k] = slice(o, o + v); o += v
    return off


def test_sim_sensor_geometry():
    """3 m 앞 벽: 벽이 보이는 행만 값, 바닥·높이 기준 아래 = 없음, 값 = 0.25 / 수평거리."""
    cam, sc = CameraSpec(), SceneSpec()
    eye = DepthEyeSim(cam, sc)
    r = eye.ranges(_wall_grid(3.0), 0.0, 0.0, 0.0, np.zeros((0, 3)))
    mid = cam.n_cols // 2
    d = 3.0 - cam.mount_x
    z = cam.height + d * np.tan(cam.row_elevations())
    exp = keep_height(z, d, cam) & (z <= sc.duct_height)
    got = np.isfinite(r[:, mid])
    assert exp.any() and (exp == got).all(), (exp, r[:, mid])
    assert np.allclose(r[got, mid], d, atol=0.05)
    near = eye.frame(_wall_grid(3.0), 0.0, 0.0, 0.0, np.zeros((0, 3)), np.random.default_rng(0), noise=False)
    assert np.allclose(near[got, mid], NEAR_REF / d, atol=2e-3) and (near[~got, mid] == 0).all()


def test_floor_leak_pitch_error():
    """벽 없는 평평한 바닥: pitch 오차 ±1.5° 까지 가짜 칸 0 (시뮬·실차 둘 다)."""
    cam, sc = CameraSpec(), SceneSpec()
    eye = DepthEyeSim(cam, sc)
    W, H = 640, 400
    f = (W / 2) / np.tan(np.radians(91.0) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    yn = ((np.mgrid[0:H, 0:W][0]) - H / 2) / f
    for err in (-1.5, -1.0, 0.0, 1.0, 1.5):
        r = eye.ranges(_wall_grid(14.0), 0.0, 0.0, 0.0, np.zeros((0, 3)), np.radians(err))
        assert not (r < cam.depth_max).any(), (err, int((r < cam.depth_max).sum()))
        pt = np.radians(cam.pitch_deg + err)
        den = yn * np.cos(pt) - np.sin(pt)
        z = np.where(den > 1e-6, cam.height / np.maximum(den, 1e-6), 0.0)
        z[z > cam.depth_max] = 0.0
        near = DepthEyeReal(cam, K, W, H).frame(z.astype(np.float32))
        assert not (near > 0).any(), (err, int((near > 0).sum()))


def test_hole_hold():
    h = HoleHold(hold=2)
    a = np.array([0.5, 0.2], np.float32)
    assert np.allclose(h(a), a)
    z = np.array([0.0, 0.3], np.float32)
    assert np.allclose(h(z), [0.5, 0.3])          # 1 프레임 구멍 → 직전 값
    assert np.allclose(h(z), [0.5, 0.3])          # 2 프레임
    assert np.allclose(h(z), [0.0, 0.3])          # 3 프레임 → 정말 없음
    h.reset()
    assert np.allclose(h(z), z)


def test_real_sensor_synthetic_wall():
    """합성 Gemini depth 영상 (4 m 앞 벽 + 바닥, pitch −10°) → 가운데 벽 칸 ≈ 0.25/4, 바닥 칸 = 0."""
    cam = CameraSpec()
    W, H = 640, 400
    f = (W / 2) / np.tan(np.radians(94.0) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    pitch = np.radians(cam.pitch_deg)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    yc = (vv - H / 2) / f
    bx = np.cos(pitch) - (-yc) * np.sin(pitch)
    bz = np.sin(pitch) + (-yc) * np.cos(pitch)
    t_wall = 4.0 / np.maximum(bx, 1e-6)
    t_floor = np.where(bz < 0, cam.height / np.maximum(-bz, 1e-6), np.inf)
    hit_wall = (cam.height + t_wall * bz >= 0) & (t_wall < t_floor)
    depth = np.where(hit_wall, t_wall, t_floor).astype(np.float32)          # z_c (광축 방향 depth)
    depth[~np.isfinite(depth)] = 0.0
    eye = DepthEyeReal(cam, K, W, H, box=1)
    near = eye.frame(depth)
    mid = slice(cam.n_cols // 2 - 4, cam.n_cols // 2 + 4)
    col = near[:, mid]
    wall_rows = (col > 0).all(axis=1)
    assert wall_rows.any(), near[:, cam.n_cols // 2]
    assert np.allclose(col[wall_rows], NEAR_REF / 4.0, rtol=0.08), col[wall_rows]
    # 벽 아래 (바닥만 보이는 행) 은 0
    el = cam.row_elevations()
    floor_only = el < np.arctan2(-cam.height, 4.0) - np.radians(5)
    assert floor_only.any() and (col[floor_only] == 0).all(), col[floor_only]


def test_brain_selectivity():
    R, C = 16, 64
    p = init_params(n_inputs())
    off = _offsets()

    def bar(cols, val=0.5):
        e = np.zeros((1, 3, R, C), np.float32)
        for t, c in enumerate(cols):
            e[0, t, 4:8, c] = val
        return e
    hs_l = features_numpy(bar([20, 21, 22]), p)[0, off["HS"]].sum()
    hs_r = features_numpy(bar([22, 21, 20]), p)[0, off["HS"]].sum()
    assert hs_l > 0 > hs_r, (hs_l, hs_r)
    e = np.zeros((1, 3, R, C), np.float32)                                # 다가오는 물체: 커지고 가까워짐
    for t, w in enumerate([2, 4, 6]):
        e[0, t, 8 - w:8 + w, 32 - w:32 + w] = 0.2 + 0.1 * t
    lp_exp = features_numpy(e, p)[0, off["LPLC2"]].sum()
    lp_con = features_numpy(e[:, ::-1].copy(), p)[0, off["LPLC2"]].sum()
    assert lp_exp > 0 and lp_exp > 3 * lp_con, (lp_exp, lp_con)
    still = np.repeat(e[:, 2:3], 3, axis=1)                               # 정지 → 움직임 0, 가까움만
    f = features_numpy(still, p)[0]
    assert np.abs(f[off["HS"]]).sum() == 0 and f[off["LPLC2"]].sum() == 0 and f[off["NEAR"]].max() > 0
    mask, sign = make_dn_wiring(n_inputs())
    out = forward_numpy(e, p, mask, sign)
    assert out.shape == (1, 48) and np.all(np.abs(out) < 1)


def test_env_obs_and_pp_lap():
    from mapless40.evaluate import pp_controller
    from .env import DepthFlyEnv
    cf = DepthFlyConfig()
    assert cf.fps == 30 and cf.hist == 3
    env = DepthFlyEnv(maps=("ifac",), cfg=cf, seed=0)
    obs, _ = env.reset(seed=0, options={"map": "ifac", "s0": 0, "lat": 0, "dyaw": 0, "v0": 2.0, "n_obstacles": 0})
    assert obs["near"].shape == (3, 16, 64) and obs["state"].shape == (state_dim(3),)
    ctrl = pp_controller(cf.env)
    t0, n = time.perf_counter(), 0
    while True:
        obs, _, te, tr, info = env.step(ctrl(obs, env)); n += 1
        if te or tr or info["laps"] >= 1:
            break
    assert info["laps"] >= 1 and not info["collided"], info
    assert 0 <= float(obs["near"].min()) and float(obs["near"].max()) <= 1
    print(f"   ifac PP 1랩 {info['t']:.1f}s, env {1e3 * (time.perf_counter() - t0) / n:.1f} ms/step")


def test_numpy_brain_speed():
    p = init_params(n_inputs())
    mask, sign = make_dn_wiring(n_inputs())
    x = np.random.default_rng(0).uniform(0, 1, (1, 3, 16, 64)).astype(np.float32)
    forward_numpy(x, p, mask, sign)
    t0 = time.perf_counter()
    for _ in range(200):
        forward_numpy(x, p, mask, sign)
    ms = (time.perf_counter() - t0) / 200 * 1e3
    print(f"   회로 numpy {ms:.2f} ms/프레임 (30 Hz 예산 33 ms)")
    assert ms < 10


# ------------------------------------------------------------------ torch
def _torch_ok():
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_brain_torch_equals_numpy():
    import torch
    from .brain import DepthFlyBrain
    m = DepthFlyBrain()
    with torch.no_grad():
        for q in (m.g_t4, m.g_t5, m.g_lp, m.b_dn):
            q.normal_(0, 0.5)
    x = np.random.default_rng(0).uniform(0, 1, (5, 3, 16, 64)).astype(np.float32)
    ref = forward_numpy(x, m.numpy_params(), m.mask.numpy(), m.sign.numpy())
    with torch.no_grad():
        out = m(torch.from_numpy(x)).numpy()
    assert np.allclose(ref, out, atol=1e-4), np.abs(ref - out).max()


def test_sac_loop_and_numpy_actor():
    import torch
    from stable_baselines3 import SAC
    from mapless40.policy import AsymSACPolicy
    from .env import DepthFlyEnv
    from .np_actor import NearNumpyActor
    from .policy import NearDeterministicActor, NearFeatures
    env = DepthFlyEnv(maps=("ifac",), cfg=DepthFlyConfig(), seed=0)
    model = SAC(AsymSACPolicy, env, learning_starts=32, batch_size=32, buffer_size=500, device="cpu",
                policy_kwargs=dict(features_extractor_class=NearFeatures, net_arch=dict(pi=[], qf=[64, 64])))
    model.learn(96)
    assert not model.policy.actor.features_extractor.use_priv and model.policy.critic.features_extractor.use_priv
    obs, _ = env.reset(seed=1)
    with tempfile.TemporaryDirectory() as d:
        model.save(f"{d}/m")
        na = NearNumpyActor(f"{d}/m.zip")
    det = NearDeterministicActor(model.policy.actor).eval()
    with torch.no_grad():
        a_t = det(torch.from_numpy(obs["near"].astype(np.float32))[None],
                  torch.from_numpy(obs["state"])[None])[0].numpy()
    a_n = na(obs)
    assert np.allclose(a_t, a_n, atol=1e-4), (a_t, a_n)


def main():
    tests = [test_sim_sensor_geometry, test_floor_leak_pitch_error, test_hole_hold, test_real_sensor_synthetic_wall,
             test_brain_selectivity, test_numpy_brain_speed, test_env_obs_and_pp_lap]
    if _torch_ok():
        tests += [test_brain_torch_equals_numpy, test_sac_loop_and_numpy_actor]
    else:
        print("(torch / stable-baselines3 없음 → torch 테스트 건너뜀)")
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
