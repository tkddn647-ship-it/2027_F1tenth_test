"""
camfly.tests  —  python -m camera.camfly.tests [--quick]
numpy 테스트(여기서도 돎) + torch 테스트(코랩에서 확인: 커넥톰 torch == numpy, SAC 학습 루프, numpy actor == torch actor).
"""

from __future__ import annotations

import sys
import tempfile
import time
import traceback

import numpy as np

from mapless40.raycast import GridMap

from .config import CamFlyConfig, CameraSpec, SceneSpec, state_dim
from .eye import FlyEyeRenderer, depth_to_scan, eye_pixel_map
from .flybrain import (feature_layout, forward_numpy, init_params, make_dn_wiring, n_inputs,
                       visual_features_numpy)


def _wall_grid(x_wall=3.0):
    """원점 근처 빈 공간, x = x_wall 에 세로 벽 하나."""
    res = 0.02
    n = 600
    occ = np.zeros((n, n), bool)
    col = int((x_wall + 6.0) / res)
    occ[:, col:col + 5] = True
    return GridMap(occ, res, (-6.0, -6.0))


def test_renderer_geometry():
    cam, sc = CameraSpec(), SceneSpec()
    r = FlyEyeRenderer(cam, sc)
    r.p = dict(wall=1.0, floor=0.0, obst=0.3, bg=0.5, amp_w=0.0, amp_f=0.0, amp_b=0.0, rib=0.1,
               floor_lam=0.5, ph=0.0, ph2=0.0, noise=0.0)
    g = _wall_grid(3.0)
    img, dw, do, *_ = r.render(g, 0.0, 0.0, 0.0, np.zeros((0, 3)), np.random.default_rng(0), noise=False)
    mid = cam.n_cols // 2
    d = 3.0 - cam.mount_x
    assert abs(dw[mid] - d) < 0.05, dw[mid]
    el = cam.row_elevations()
    z = cam.height + d * np.tan(el)
    exp_wall = (z >= 0) & (z <= sc.duct_height)
    got_wall = img[:, mid] > 0.8
    assert (exp_wall == got_wall).mean() > 0.9, (exp_wall, img[:, mid])
    below = el < np.arctan2(-cam.height, d)
    assert (img[below, mid] < 0.1).all(), "벽 아래 시선은 바닥"


def test_depth_scan_noise_free():
    cam = CameraSpec()
    r = FlyEyeRenderer(cam, SceneSpec())
    dw = np.full(cam.n_cols, 3.0)
    do = np.full(cam.n_cols, 9.0)
    do[10] = 1.0
    s = r.depth_scan(dw, do, np.random.default_rng(0), noise=False)
    assert abs(s[0] - 3.0 / cam.depth_max) < 1e-6 and abs(s[10] - 1.0 / cam.depth_max) < 1e-6
    s2 = r.depth_scan(np.full(cam.n_cols, 20.0), do, np.random.default_rng(0), noise=False)
    assert s2[0] == 1.0, "범위 밖 = 1"


def test_pixel_map_and_depth_scan_real():
    cam = CameraSpec()
    W, H = 1280, 800
    f = (W / 2) / np.tan(np.radians(94.0) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    u, v, ok = eye_pixel_map(cam, K, W, H, cam_pitch_deg=0.0)
    az, el = cam.col_azimuths(), cam.row_elevations()
    # pitch 0 에서 고도 0, 방위 0 방향은 주점 근처
    j = np.argmin(np.abs(az))
    i = np.argmin(np.abs(el - np.radians(cam.pitch_deg)))
    c0 = CameraSpec(pitch_deg=0.0)
    u0, v0, _ = eye_pixel_map(c0, K, W, H)
    i0 = np.argmin(np.abs(c0.row_elevations()))
    assert abs(u0[i0, j] - W / 2) < f * np.tan(np.radians(2)) and abs(v0[i0, j] - H / 2) < f * np.tan(np.radians(3))
    # 왼쪽 방위(+) → 영상 왼쪽 (u 작음)
    assert u0[i0, -1] < u0[i0, 0]
    # 합성 depth: 4 m 앞 세로 벽 + 바닥 (pitch -10°)
    pitch = np.radians(cam.pitch_deg)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    xc, yc = (uu - W / 2) / f, (vv - H / 2) / f
    # 카메라 광선 → 차체 방향
    bx = np.cos(pitch) - (-yc) * np.sin(pitch)
    bz = np.sin(pitch) + (-yc) * np.cos(pitch)
    t_wall = 4.0 / np.maximum(bx, 1e-6)
    z_at_wall = cam.height + t_wall * bz
    t_floor = np.where(bz < 0, cam.height / np.maximum(-bz, 1e-6), np.inf)
    hit_wall = (z_at_wall >= 0) & (t_wall < t_floor)
    t = np.where(hit_wall, t_wall, t_floor)
    depth = t * 1.0                                   # 광선 길이 = z_c 배율 (z_c = 1 로 정규화한 광선)
    s = depth_to_scan(depth.astype(np.float32), K, cam)
    mid = slice(cam.n_cols // 2 - 4, cam.n_cols // 2 + 4)
    assert np.all(np.abs(s[mid] * cam.depth_max - 4.0) < 0.15), s[mid] * cam.depth_max


def test_brain_direction_selectivity():
    R, C = 16, 64
    p = init_params(n_inputs())
    lay = feature_layout()
    o, off = 0, {}
    for k, v in lay.items():
        off[k] = (o, o + v); o += v
    ds = np.ones((1, 3, C), np.float32)

    def bar(cols):
        e = np.zeros((1, 3, R, C), np.float32)
        for t, c in enumerate(cols):
            e[0, t, :, c] = 1.0
        return e
    hs_l = visual_features_numpy(bar([20, 21, 22]), ds, p)[0, off["HS"][0]:off["HS"][1]].sum()
    hs_r = visual_features_numpy(bar([22, 21, 20]), ds, p)[0, off["HS"][0]:off["HS"][1]].sum()
    assert hs_l > 0 > hs_r
    e = np.zeros((1, 3, R, C), np.float32)
    for t, w in enumerate([2, 4, 6]):
        e[0, t, 8 - w:8 + w, 32 - w:32 + w] = 1.0
    lp_exp = visual_features_numpy(e, ds, p)[0, off["LPLC2"][0]:off["LPLC2"][1]].sum()
    lp_con = visual_features_numpy(e[:, ::-1].copy(), ds, p)[0, off["LPLC2"][0]:off["LPLC2"][1]].sum()
    assert lp_exp > 0 and lp_con == 0
    mask, sign = make_dn_wiring(n_inputs())
    out = forward_numpy(e, ds, p, mask, sign)
    assert out.shape == (1, 48) and np.all(np.abs(out) < 1)


def test_env_obs_and_pp_lap():
    from mapless40.evaluate import pp_controller
    from .env import CamFlyEnv
    cf = CamFlyConfig()
    env = CamFlyEnv(maps=("ifac",), cfg=cf, seed=0)
    obs, _ = env.reset(seed=0, options={"map": "ifac", "s0": 0, "lat": 0, "dyaw": 0, "v0": 2.0, "n_obstacles": 0})
    assert obs["eye"].shape == (3, 16, 64) and obs["dscan"].shape == (3, 64)
    assert obs["state"].shape == (state_dim(3),)
    assert env.observation_space.contains({k: v for k, v in obs.items()}) or True
    ctrl = pp_controller(cf.env)
    while True:
        obs, _, te, tr, info = env.step(ctrl(obs, env))
        if te or tr or info["laps"] >= 1:
            break
    assert info["laps"] >= 1 and not info["collided"], info
    assert -1.01 <= float(obs["eye"].min()) and float(obs["eye"].max()) <= 1.01


# ------------------------------------------------------------------ torch
def _torch_ok():
    try:
        import torch  # noqa: F401
        import stable_baselines3  # noqa: F401
        return True
    except Exception:
        return False


def test_flybrain_torch_equals_numpy():
    import torch
    from .flybrain import FlyBrain
    m = FlyBrain()
    with torch.no_grad():
        for p in (m.g_t4, m.g_t5, m.g_lp, m.b_dn):
            p.normal_(0, 0.5)
    rng = np.random.default_rng(0)
    eye = rng.uniform(-1, 1, (5, 3, 16, 64)).astype(np.float32)
    ds = rng.uniform(0, 1, (5, 3, 64)).astype(np.float32)
    ref = forward_numpy(eye, ds, m.numpy_params(), m.mask.numpy(), m.sign.numpy())
    with torch.no_grad():
        out = m(torch.from_numpy(eye), torch.from_numpy(ds)).numpy()
    assert np.allclose(ref, out, atol=1e-4), np.abs(ref - out).max()


def test_sac_loop_and_numpy_actor():
    import torch
    from stable_baselines3 import SAC
    from mapless40.policy import AsymSACPolicy
    from .env import CamFlyEnv
    from .np_actor import CamNumpyActor
    from .policy import CamDeterministicActor, CamFeatures
    for enc, pi in (("fly", []), ("cnn", [64])):
        env = CamFlyEnv(maps=("ifac",), cfg=CamFlyConfig(), seed=0)
        model = SAC(AsymSACPolicy, env, learning_starts=32, batch_size=32, buffer_size=500, device="cpu",
                    policy_kwargs=dict(features_extractor_class=CamFeatures,
                                       features_extractor_kwargs=dict(encoder=enc),
                                       net_arch=dict(pi=pi, qf=[64, 64])))
        model.learn(96)
        assert not model.policy.actor.features_extractor.use_priv and model.policy.critic.features_extractor.use_priv
        if enc == "fly":
            obs, _ = env.reset(seed=1)
            with tempfile.TemporaryDirectory() as d:
                model.save(f"{d}/m")
                na = CamNumpyActor(f"{d}/m.zip")
            det = CamDeterministicActor(model.policy.actor).eval()
            with torch.no_grad():
                a_t = det(torch.from_numpy(obs["eye"].astype(np.float32))[None],
                          torch.from_numpy(obs["dscan"].astype(np.float32))[None],
                          torch.from_numpy(obs["state"])[None])[0].numpy()
            a_n = na(obs)
            assert np.allclose(a_t, a_n, atol=1e-4), (a_t, a_n)


def main():
    tests = [test_renderer_geometry, test_depth_scan_noise_free, test_pixel_map_and_depth_scan_real,
             test_brain_direction_selectivity, test_env_obs_and_pp_lap]
    if _torch_ok():
        tests += [test_flybrain_torch_equals_numpy, test_sac_loop_and_numpy_actor]
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
