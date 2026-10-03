"""
depthfly.brain
==============
depth 카메라 한 대 → 초파리 시각엽 회로 → 하행 뉴런(DN).  커넥톰 회로만 쓴다 (CNN·MLP·별도 depth 경로 없음).

입력: near (B, 3, R, C) — '가까움' 영상 프레임 n-2, n-1, n.  가까움 = 0.25 m / 거리 ∈ (0, 1], 측정 없음 = 0.
      (파리 광수용체가 받는 밝기 자리에 가까움을 넣는다: 가까운 벽·장애물이 밝게 보이는 눈)

  lamina   L_ON/L_OFF = ±Δ가까움 (빠른 갈래 n−(n−1), 느린 갈래 (n−1)−(n−2))
           L3 지속 경로 = 현재 가까움 자체
  T4/T5    Reichardt 상관기 4방향 (왼·오·위·아래), 방향별 이득 학습
  HS/VS    섹터 8 × 밴드 2 의 수평·수직 흐름
  LPLC2    섹터별 바깥쪽 흐름 = 다가옴
  LC       섹터·밴드별 가장자리 (물체 경계)
  L3→LC    섹터·밴드별 평균 가까움, 섹터별 최대 가까움 (가장 가까운 물체)
  DN 48    고정 희소 배선 (입력 12개, 부호 고정) × 학습되는 세기  → tanh

학습: 이득 5개(T4·T5 방향별 8 + 경로 이득 5) + DN 세기·바이어스.  numpy 참조 / torch 모듈 같은 수식.
"""

from __future__ import annotations

import numpy as np

from camfly.flybrain import DIR_NAMES, DIRS, _pool_np, _shift_np, _softplus, make_dn_wiring

N_SECT, N_BAND = 8, 2


def feature_layout(n_sect: int = N_SECT, n_band: int = N_BAND) -> dict[str, int]:
    sb = n_sect * n_band
    return {"HS": sb, "VS": sb, "LPLC2": n_sect, "LC": 2 * sb, "LUM": sb, "NEAR": n_sect}


def n_inputs(n_sect: int = N_SECT, n_band: int = N_BAND) -> int:
    return sum(feature_layout(n_sect, n_band).values())


def init_params(n_in: int, n_dn: int = 48, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    return {"g_t4": np.zeros(4, np.float32), "g_t5": np.zeros(4, np.float32),
            "g_lp": np.zeros(5, np.float32),                 # HS, VS, LPLC2, LC, L3
            "w_dn": rng.normal(0.0, 0.5, (n_dn, n_in)).astype(np.float32),
            "b_dn": np.zeros(n_dn, np.float32)}


PARAM_KEYS = ("g_t4", "g_t5", "g_lp", "w_dn", "b_dn")


def features_numpy(near, p, n_sect=N_SECT, n_band=N_BAND):
    f0, f1, f2 = near[:, 0], near[:, 1], near[:, 2]
    dn, dp = f2 - f1, f1 - f0
    on_f, on_s, off_f, off_s = np.maximum(dn, 0), np.maximum(dp, 0), np.maximum(-dn, 0), np.maximum(-dp, 0)
    g4, g5, gl = _softplus(p["g_t4"]), _softplus(p["g_t5"]), _softplus(p["g_lp"])
    mot = {}
    for k, name in enumerate(DIR_NAMES):
        dr, dc = DIRS[name]
        t4 = np.maximum(g4[k] * (on_s * _shift_np(on_f, dr, dc) - on_f * _shift_np(on_s, dr, dc)), 0)
        t5 = np.maximum(g5[k] * (off_s * _shift_np(off_f, dr, dc) - off_f * _shift_np(off_s, dr, dc)), 0)
        mot[name] = t4 + t5
    B, R, C = f2.shape
    hs = _pool_np(mot["left"] - mot["right"], n_sect, n_band) * gl[0]
    vs = _pool_np(mot["down"] - mot["up"], n_sect, n_band) * gl[1]
    half = np.concatenate([mot["right"][..., :C // 2], mot["left"][..., C // 2:]], axis=-1)
    vert = np.concatenate([mot["up"][..., :R // 2, :], mot["down"][..., R // 2:, :]], axis=-2)
    lplc2 = _pool_np(half + vert, n_sect, 1)[:, 0, :] * gl[2]
    ex, ey = np.abs(f2 - _shift_np(f2, 0, 1)), np.abs(f2 - _shift_np(f2, 1, 0))
    lc = np.concatenate([_pool_np(ex, n_sect, n_band), _pool_np(ey, n_sect, n_band)], axis=1) * gl[3]
    lum = _pool_np(f2, n_sect, n_band) * gl[4]
    near_max = f2.reshape(B, R, n_sect, C // n_sect).max(axis=(1, 3)) * gl[4]
    return np.concatenate([hs.reshape(B, -1), vs.reshape(B, -1), lplc2, lc.reshape(B, -1),
                           lum.reshape(B, -1), near_max], axis=1)


def forward_numpy(near, p, mask, sign, n_sect=N_SECT, n_band=N_BAND):
    x = features_numpy(near.astype(np.float32), p, n_sect, n_band)
    W = sign * mask * _softplus(p["w_dn"])
    return np.tanh(x @ W.T + p["b_dn"])


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from camfly.flybrain import _pool_t, _shift_t
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:

    class DepthFlyBrain(nn.Module):
        def __init__(self, n_dn: int = 48, fan_in: int = 12, seed: int = 0, n_sect: int = N_SECT,
                     n_band: int = N_BAND):
            super().__init__()
            self.n_sect, self.n_band = n_sect, n_band
            n_in = n_inputs(n_sect, n_band)
            mask, sign = make_dn_wiring(n_in, n_dn, fan_in, seed)
            p = init_params(n_in, n_dn, seed)
            self.register_buffer("mask", torch.from_numpy(mask))
            self.register_buffer("sign", torch.from_numpy(sign))
            for k in PARAM_KEYS:
                setattr(self, k, nn.Parameter(torch.from_numpy(p[k])))
            self.features_dim = n_dn

        def numpy_params(self) -> dict:
            return {k: getattr(self, k).detach().cpu().numpy() for k in PARAM_KEYS}

        def features(self, near):
            f0, f1, f2 = near[:, 0], near[:, 1], near[:, 2]
            dn, dp = f2 - f1, f1 - f0
            on_f, on_s, off_f, off_s = F.relu(dn), F.relu(dp), F.relu(-dn), F.relu(-dp)
            g4, g5, gl = F.softplus(self.g_t4), F.softplus(self.g_t5), F.softplus(self.g_lp)
            mot = {}
            for k, name in enumerate(DIR_NAMES):
                dr, dc = DIRS[name]
                t4 = F.relu(g4[k] * (on_s * _shift_t(on_f, dr, dc) - on_f * _shift_t(on_s, dr, dc)))
                t5 = F.relu(g5[k] * (off_s * _shift_t(off_f, dr, dc) - off_f * _shift_t(off_s, dr, dc)))
                mot[name] = t4 + t5
            B, R, C = f2.shape
            ns, nb = self.n_sect, self.n_band
            hs = _pool_t(mot["left"] - mot["right"], ns, nb) * gl[0]
            vs = _pool_t(mot["down"] - mot["up"], ns, nb) * gl[1]
            half = torch.cat([mot["right"][..., :C // 2], mot["left"][..., C // 2:]], dim=-1)
            vert = torch.cat([mot["up"][..., :R // 2, :], mot["down"][..., R // 2:, :]], dim=-2)
            lplc2 = _pool_t(half + vert, ns, 1)[:, 0, :] * gl[2]
            ex, ey = (f2 - _shift_t(f2, 0, 1)).abs(), (f2 - _shift_t(f2, 1, 0)).abs()
            lc = torch.cat([_pool_t(ex, ns, nb), _pool_t(ey, ns, nb)], dim=1) * gl[3]
            lum = _pool_t(f2, ns, nb) * gl[4]
            near_max = f2.reshape(B, R, ns, C // ns).amax(dim=(1, 3)) * gl[4]
            return torch.cat([hs.reshape(B, -1), vs.reshape(B, -1), lplc2, lc.reshape(B, -1),
                              lum.reshape(B, -1), near_max], dim=1)

        def forward(self, near):
            x = self.features(near.float())
            W = self.sign * self.mask * F.softplus(self.w_dn)
            return torch.tanh(x @ W.t() + self.b_dn)
