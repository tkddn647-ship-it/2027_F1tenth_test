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

from camera.depthfly.brain import init_params, make_dn_wiring_bilateral, n_inputs, symmetric_init

from .brain import (
    BEAM_W, NEAR_REF_M, N_COLS, feature_layout, feature_meta, features_numpy, forward_numpy,
    mirror_dn_index, scans_to_near_np,
)
from .config import (
    COLLISION, FOV_DEG, HIST, N_BEAMS, OVER_MULT, SERVO_TAU_S, STEER_MAX_RAD, V_MAX, V_MIN,
    W_ALAT, W_EY, W_PROGRESS, W_V_TRACK, W_WALL, lidarfly_config, state_dim,
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


def test_reward_weights_arbitrary_init():
    """보상 계수는 config 한곳. 진행·v_ref 추종은 양수 계수, 충돌은 음수."""
    c = lidarfly_config()
    assert c.reward.w_progress == W_PROGRESS > 0
    assert c.reward.w_ey == W_EY > 0
    assert c.reward.w_v == W_V_TRACK > 0          # v≤v_ref 편차 감점 → 추종할수록 보상 증가
    assert c.reward.over_mult == OVER_MULT > 1    # v>v_ref (곡률이 정한 속도보다 빠름) 배율
    assert c.reward.w_alat == W_ALAT > 0
    assert c.reward.w_wall == W_WALL > 0
    assert c.reward.collision == COLLISION < 0


def _feat_slices():
    o, g = 0, {}
    for k, n in feature_layout().items():
        g[k] = slice(o, o + n)
        o += n
    return g


def test_pool_near_and_numpy_forward():
    """구간 min/median → 1/거리 가까움, ON/OFF 방향, DN forward."""
    col = np.linspace(0.2, 0.8, BEAM_W).astype(np.float32)
    scan = np.ones((1, 3, N_BEAMS), np.float32)
    scan[0, :, :BEAM_W] = col
    rm = 10.0
    near = scans_to_near_np(scan, rm)
    assert near.shape == (1, 3, 2, N_COLS)
    d_min, d_med = float(col.min()), float(np.median(col))
    assert d_min < d_med
    assert np.isclose(near[0, 0, 1, 0], np.clip(NEAR_REF_M / (d_min * rm), 0, 1))
    assert np.isclose(near[0, 0, 0, 0], np.clip(NEAR_REF_M / (d_med * rm), 0, 1))
    assert near[0, 0, 1, 0] > near[0, 0, 0, 0]
    assert near.min() >= 0 and near.max() <= 1

    p = init_params(n_inputs())
    off = _feat_slices()

    def blob(cols):
        s = np.ones((1, 3, N_BEAMS), np.float32)
        for t, c0 in enumerate(cols):
            a = int(c0) * BEAM_W
            s[0, t, a:a + 2 * BEAM_W] = 0.08
        return s

    hs_l = features_numpy(blob([20, 21, 22]), p, rm)[0, off["HS"]].sum()
    hs_r = features_numpy(blob([22, 21, 20]), p, rm)[0, off["HS"]].sum()
    assert hs_l > 0 > hs_r, (hs_l, hs_r)
    still = features_numpy(blob([20, 20, 20]), p, rm)[0]
    assert abs(still[off["HS"]].sum()) == 0 and still[off["LPLC2"]].sum() == 0

    approach = np.full((1, 3, N_BEAMS), 0.9, np.float32)
    retreat = approach.copy()
    for t, d in enumerate((0.70, 0.45, 0.20)):
        approach[0, t, 36 * BEAM_W:45 * BEAM_W] = d
        retreat[0, 2 - t, 36 * BEAM_W:45 * BEAM_W] = d
    lp_a = features_numpy(approach, p, rm)[0, off["LPLC2"]][4]
    lp_r = features_numpy(retreat, p, rm)[0, off["LPLC2"]][4]
    assert lp_a > 0 and lp_a > lp_r, (lp_a, lp_r)

    mask, sign = make_dn_wiring_bilateral()
    out = forward_numpy(approach, p, mask, sign, rm)
    assert out.shape == (1, 48) and np.all(np.abs(out) < 1)


def test_bilateral_polarity():
    """같은 쪽 흥분·반대쪽 억제. 세기만 학습(부호 버퍼). 좌우 뒤집으면 짝 DN 이 같다."""
    meta, sect, mirror = feature_meta()
    is_hs = np.array([m[0] == "HS" for m in meta])
    is_left = sect >= 4
    n_dn, n_mid, fan_in = 48, 8, 12
    n_pair = (n_dn - n_mid) // 2
    mask, sign = make_dn_wiring_bilateral(n_dn, fan_in, seed=0, n_mid=n_mid)
    for i in range(n_pair):
        idx = np.flatnonzero(mask[i] > 0)
        assert idx.size == fan_in
        for j in idx:
            if is_hs[j]:
                assert sign[i, j] == 1.0
            else:
                assert sign[i, j] == (1.0 if is_left[j] else -1.0)
        j_dn = n_pair + i
        midx = mirror[idx]
        assert np.all(mask[j_dn, midx] == 1.0)
        for a, b in zip(idx, midx):
            if is_hs[a]:
                assert sign[j_dn, b] == -sign[i, a]
            else:
                assert sign[j_dn, b] == sign[i, a]
    for k in range(n_mid):
        i = 2 * n_pair + k
        idx = np.flatnonzero(mask[i] > 0)
        assert idx.size == fan_in
        left = idx[is_left[idx]]
        assert left.size == fan_in // 2
        assert set(mirror[left].tolist()) == set(idx[~is_left[idx]].tolist())
        for j in idx:
            if is_hs[j]:
                assert sign[i, j] == (1.0 if is_left[j] else -1.0)
            else:
                assert sign[i, j] == -1.0

    p = symmetric_init(init_params(n_inputs(), n_dn, seed=0), n_dn, n_mid)
    rng = np.random.default_rng(0)
    scan = rng.uniform(0.05, 1.0, (2, 3, N_BEAMS)).astype(np.float32)
    y = forward_numpy(scan, p, mask, sign, 10.0)
    y_flip = forward_numpy(scan[..., ::-1].copy(), p, mask, sign, 10.0)
    err = np.max(np.abs(y_flip[:, mirror_dn_index(n_dn, n_mid)] - y))
    assert err < 1e-5, err


def _torch_ok():
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_brain_torch_equals_numpy():
    import torch
    from .brain import LidarFlyBrain
    m = LidarFlyBrain(seed=0)
    mask_np, sign_np = make_dn_wiring_bilateral(seed=0)
    assert np.array_equal(m.mask.detach().cpu().numpy(), mask_np)
    assert np.array_equal(m.sign.detach().cpu().numpy(), sign_np)
    assert not m.sign.requires_grad and m.core.w_dn.requires_grad
    x = np.random.default_rng(1).uniform(0.05, 1.0, (4, 3, N_BEAMS)).astype(np.float32)
    rm = float(m.range_max)
    ref = forward_numpy(x, m.numpy_params(), m.mask.detach().cpu().numpy(),
                        m.sign.detach().cpu().numpy(), rm)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        out = m(xt).numpy()
    assert np.allclose(ref, out, atol=1e-4), float(np.abs(ref - out).max())
    y = m(xt).sum()
    y.backward()
    g = m.core.w_dn.grad
    assert g is not None and float(g.abs().sum()) > 0


def test_train_cli_and_short_learn():
    """--save-dir / --resume auto / --v-max 7 / pi=[] .  정책 생성 후 몇 스텝만 업데이트."""
    import torch
    from torch import nn
    from stable_baselines3 import SAC
    from mapless40.config import PRIV_DIM
    from mapless40.policy import AsymSACPolicy
    from .train import apply_train_cfg, build_parser, policy_kwargs

    args = build_parser().parse_args(["--save-dir", "runs/lidarfly_smoke", "--resume", "auto", "--v-max", "7"])
    assert args.save_dir == "runs/lidarfly_smoke" and args.resume == "auto"
    assert args.v_max == 7.0 and args.pi_net == ""
    cfg = apply_train_cfg(args)
    assert cfg.action.v_max == 7.0 and cfg.raceline.v_max == 7.0
    kw = policy_kwargs(args, cfg)
    assert kw["net_arch"]["pi"] == [] and kw["net_arch"]["qf"] == [256, 256]

    env = LidarFlyEnv(maps="ifac", seed=0, sensor_noise=False, randomize=False)
    smoke = dict(kw)
    smoke["net_arch"] = dict(pi=[], qf=[32, 32])
    model = SAC(AsymSACPolicy, env, learning_starts=8, batch_size=8, buffer_size=64,
                gradient_steps=1, train_freq=1, policy_kwargs=smoke, verbose=0,
                seed=0, device="cpu")
    actor, critic = model.policy.actor, model.policy.critic
    assert not actor.features_extractor.use_priv and critic.features_extractor.use_priv
    assert actor.features_extractor.features_dim + PRIV_DIM == critic.features_extractor.features_dim
    assert PRIV_DIM == 27
    assert isinstance(actor.mu, nn.Linear) and actor.mu.out_features == 2
    assert actor.mu.in_features == 48 + state_dim(3)
    assert sum(p.numel() for p in actor.latent_pi.parameters()) == 0
    assert not any(isinstance(mod, (nn.Conv1d, nn.Conv2d)) for mod in actor.modules())
    obs, _ = env.reset(seed=0, options={"map": "ifac", "s0": 1.0, "lat": 0.0, "dyaw": 0.0,
                                        "v0": 2.0, "n_obstacles": 0})
    batch = {k: torch.from_numpy(np.asarray(v)[None]) for k, v in obs.items()}
    with torch.no_grad():
        a1 = actor.features_extractor(batch)
        bumped = {k: v.clone() for k, v in batch.items()}
        bumped["priv"] = bumped["priv"] + 1
        a2 = actor.features_extractor(bumped)
        c1 = critic.features_extractor(batch)
        c2 = critic.features_extractor(bumped)
    assert torch.allclose(a1, a2)
    assert not torch.allclose(c1, c2)
    before = actor.features_extractor.brain.core.w_dn.detach().clone()
    model.learn(total_timesteps=20)
    after = actor.features_extractor.brain.core.w_dn.detach()
    assert not torch.equal(before, after)


def main():
    tests = [
        test_mapless40_defaults_unchanged,
        test_lidarfly_config_sources,
        test_parse_weights,
        test_random_track_raceline,
        test_env_three_maps_obs_shape,
        test_reward_weights_arbitrary_init,
        test_pool_near_and_numpy_forward,
        test_bilateral_polarity,
    ]
    if _torch_ok():
        tests += [test_brain_torch_equals_numpy, test_train_cli_and_short_learn]
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
