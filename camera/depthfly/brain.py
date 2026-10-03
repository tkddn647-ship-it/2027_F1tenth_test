"""
depthfly.brain
==============
depth 카메라 한 대 → 초파리 시각엽 회로 → 하행 뉴런(DN).  커넥톰 회로만 쓴다 (CNN·MLP·별도 depth 경로 없음).

입력: near (B, 3, R, C) — '가까움' 영상 프레임 n-2, n-1, n.  가까움 = 0.25 m / 거리 ∈ (0, 1], 측정 없음 = 0.
      (파리 광수용체가 받는 밝기 자리에 가까움을 넣는다: 가까운 벽·장애물이 밝게 보이는 눈)

  lamina   L_ON/L_OFF = ±Δ가까움 (빠른 갈래 n−(n−1), 느린 갈래 (n−1)−(n−2)), 대비 적응 Δ/(|Δ|+0.01)
           L3 지속 경로 = 현재 가까움 자체
  T4/T5    Reichardt 상관기 4방향 (왼·오·위·아래), 방향별 이득 학습
  HS/VS    섹터 8 × 밴드 2 의 수평·수직 흐름
  LPLC2    섹터별 다가옴 = 가까움 증가율 Δn/n (= 시각 크기 팽창률; depth 입력에서는 바깥쪽 흐름보다 정확)
  LC       섹터·밴드별 가장자리 (물체 경계)
  L3→LC    섹터·밴드별 평균 가까움, 섹터별 최대 가까움 (가장 가까운 물체)
  DN 48    고정 희소 배선 (입력 12개) × 학습되는 세기 → tanh.  부호는 좌우 대칭 규칙 (make_dn_wiring_bilateral):
           같은 쪽 시야 흥분·반대쪽 억제, 왼쪽/오른쪽 DN 20쌍 거울상 + 가운데 8개 (v2 까지는 부호 랜덤 60:40)

학습: 이득 5개(T4·T5 방향별 8 + 경로 이득 5) + DN 세기·바이어스.  numpy 참조 / torch 모듈 같은 수식.
"""

from __future__ import annotations

import numpy as np

from camera.camfly.flybrain import DIR_NAMES, DIRS, _pool_np, _shift_np, _softplus, make_dn_wiring

N_SECT, N_BAND = 8, 2
# 경로별 고정 이득 (회로 설계값, 학습 안 함): HS, VS, LPLC2, LC, L3.  평균 풀링이 묽어지는 것을 보정해
# ifac·팀 맵 주행에서 각 경로 평균 크기가 ~0.03~0.1 로 비슷해지게 맞춤 (행을 지평선 ±8° 에 몬 v2 격자 기준;
# 균일 격자 때 값 50, 50, 20, 10, 1).  학습되는 이득 softplus(g_lp)/ln2 는 1 에서 시작.
GAIN0 = (15.0, 15.0, 7.0, 10.0, 1.0)
LN2 = float(np.log(2.0))
SIGMA_LAM = 0.01     # lamina 대비 적응: Δ → Δ / (|Δ| + σ).  주행 중 |Δ가까움| 평균 0.003 → 움직임 신호가 가까움과 비슷한 크기로


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


def feature_meta(n_sect: int = N_SECT, n_band: int = N_BAND):
    """특징마다 (경로, 섹터, 부호있음).  섹터 0~3 = 오른쪽, 4~7 = 왼쪽.  좌우 거울상 특징 인덱스도 돌려줌."""
    meta = []
    for g, n in feature_layout(n_sect, n_band).items():
        for i in range(n):
            meta.append((g, i % n_sect, g in ("HS",)))   # HS 만 좌우 방향 부호가 있음 (VS 는 위아래)
    meta_arr = np.array([m[1] for m in meta])
    mirror = np.empty(len(meta), np.int64)
    o = 0
    for g, n in feature_layout(n_sect, n_band).items():
        idx = np.arange(n)
        mirror[o + idx] = o + (idx // n_sect) * n_sect + (n_sect - 1 - idx % n_sect)
        o += n
    return meta, meta_arr, mirror


def make_dn_wiring_bilateral(n_dn: int = 48, fan_in: int = 12, seed: int = 0, n_mid: int = 8):
    """좌우 대칭 DN 배선 (부호에 의미가 있음).  실제 초파리 연결 데이터는 아니고 다음 원칙을 따른다:
      - DN 은 왼쪽·오른쪽 쌍 (곤충 하행 뉴런은 좌우 쌍).  오른쪽 DN = 왼쪽 DN 의 거울상 배선.
      - 같은 쪽(ipsi) 시야 입력은 흥분 +, 반대쪽(contra) 은 억제 −  (가까움·다가옴·가장자리·VS).
      - HS (좌우 흐름) 는 '왼쪽으로 흐름' 이 왼쪽 DN 을 흥분 (= 오른쪽으로 도는 회전 감지), 거울상 DN 은 반대.
      - 가운데 DN n_mid 개 (좌우 대칭 입력): 가까움·다가옴·가장자리 → 억제 −, 양옆 바깥쪽 흐름(전진 속도) → 흥분 +.
    순서: [왼쪽 n_pair | 오른쪽 n_pair | 가운데 n_mid].  (n_dn, n_in) 마스크·부호."""
    meta, sect, mirror = feature_meta()
    n_in = len(meta)
    half = N_SECT // 2
    is_left = sect >= half
    is_hs = np.array([m[0] == "HS" for m in meta])
    n_pair = (n_dn - n_mid) // 2
    assert 2 * n_pair + n_mid == n_dn
    rng = np.random.default_rng(seed)
    mask = np.zeros((n_dn, n_in), np.float32)
    sign = np.zeros((n_dn, n_in), np.float32)
    for i in range(n_pair):
        idx = rng.choice(n_in, size=fan_in, replace=False)
        sg = np.where(is_left[idx], 1.0, -1.0)                 # ipsi + / contra −
        sg = np.where(is_hs[idx], 1.0, sg)                     # HS: 왼쪽으로 흐름 → +
        mask[i, idx], sign[i, idx] = 1.0, sg
        j = n_pair + i                                         # 거울상 오른쪽 DN
        mask[j, mirror[idx]] = 1.0
        sign[j, mirror[idx]] = np.where(is_hs[idx], -sg, sg)
    left_idx = np.nonzero(is_left)[0]
    for k in range(n_mid):
        i = 2 * n_pair + k
        idx = rng.choice(left_idx, size=fan_in // 2, replace=False)
        both = np.concatenate([idx, mirror[idx]])
        sg = np.where(is_hs[both], np.where(is_left[both], 1.0, -1.0), -1.0)   # 바깥쪽 흐름 + / 가까움 −
        mask[i, both], sign[i, both] = 1.0, sg
    return mask, sign


def mirror_dn_index(n_dn: int = 48, n_mid: int = 8) -> np.ndarray:
    """DN 거울상 짝 인덱스 (가운데 DN 은 자기 자신)."""
    n_pair = (n_dn - n_mid) // 2
    m = np.arange(n_dn)
    m[:n_pair], m[n_pair:2 * n_pair] = np.arange(n_pair, 2 * n_pair), np.arange(n_pair)
    return m


def symmetric_init(p: dict, n_dn: int = 48, n_mid: int = 8) -> dict:
    """오른쪽 DN 세기·바이어스를 왼쪽 짝과 같게 (학습 시작이 좌우 대칭), 가운데 DN 은 거울상 입력끼리 같게."""
    n_pair = (n_dn - n_mid) // 2
    _, _, mirror = feature_meta()
    w = p["w_dn"].copy()
    w[n_pair:2 * n_pair] = w[:n_pair][:, np.argsort(mirror)]
    w[2 * n_pair:] = 0.5 * (w[2 * n_pair:] + w[2 * n_pair:][:, np.argsort(mirror)])
    b = p["b_dn"].copy()
    b[n_pair:2 * n_pair] = b[:n_pair]
    return {**p, "w_dn": w.astype(np.float32), "b_dn": b.astype(np.float32)}


def make_wiring(kind: str, n_in: int, n_dn: int = 48, fan_in: int = 12, seed: int = 0):
    if kind == "bilateral":
        return make_dn_wiring_bilateral(n_dn, fan_in, seed)
    return make_dn_wiring(n_in, n_dn, fan_in, seed)                    # 'random' (v2 까지)


LOOM_EPS = 0.02      # 다가옴 = (Δ가까움 − 문턱) / (가까움 + ε)
LOOM_TH = 0.005      # 가까움 잡음 σ = 0.25·k = 0.0025 (k=0.01, 거리 무관) 의 두 프레임 차 ≈ 0.0035 → 문턱 0.005


def _loom_np(f1, f2, n_sect):
    """LPLC2 (depth 판): 섹터별 '가까움이 커지는 비율' 평균 (두 프레임 모두 값이 있는 칸만).
    크기가 고정된 물체의 시각 크기 θ ∝ 가까움 이므로 Δθ/θ = Δ가까움/가까움 — LPLC2 가 재는 '팽창률' 과 같은 양.
    (T4/T5 바깥쪽 흐름으로 재면 2 m 밖 물체는 가장자리가 프레임당 1칸도 안 움직여 방향이 거꾸로 나옴 — polarity.py)"""
    B, R, C = f2.shape
    v = ((f1 > 0) & (f2 > 0)).astype(np.float32)
    lo = np.clip(np.maximum(f2 - f1 - LOOM_TH, 0) / (f1 + LOOM_EPS), 0, 1) * v
    s = lo.reshape(B, R, n_sect, C // n_sect).sum(axis=(1, 3))
    n = v.reshape(B, R, n_sect, C // n_sect).sum(axis=(1, 3))
    return s / np.maximum(n, 1.0)


def features_numpy(near, p, n_sect=N_SECT, n_band=N_BAND):
    f0, f1, f2 = near[:, 0], near[:, 1], near[:, 2]
    dn, dp = f2 - f1, f1 - f0
    if SIGMA_LAM:
        dn, dp = dn / (np.abs(dn) + SIGMA_LAM), dp / (np.abs(dp) + SIGMA_LAM)
    on_f, on_s, off_f, off_s = np.maximum(dn, 0), np.maximum(dp, 0), np.maximum(-dn, 0), np.maximum(-dp, 0)
    g4, g5 = _softplus(p["g_t4"]), _softplus(p["g_t5"])
    gl = _softplus(p["g_lp"]) / LN2 * np.asarray(GAIN0, np.float32)
    mot = {}
    for k, name in enumerate(DIR_NAMES):
        dr, dc = DIRS[name]
        t4 = np.maximum(g4[k] * (on_s * _shift_np(on_f, dr, dc) - on_f * _shift_np(on_s, dr, dc)), 0)
        t5 = np.maximum(g5[k] * (off_s * _shift_np(off_f, dr, dc) - off_f * _shift_np(off_s, dr, dc)), 0)
        mot[name] = t4 + t5
    B, R, C = f2.shape
    hs = _pool_np(mot["left"] - mot["right"], n_sect, n_band) * gl[0]
    vs = _pool_np(mot["down"] - mot["up"], n_sect, n_band) * gl[1]
    lplc2 = _loom_np(f1, f2, n_sect) * gl[2]
    ex = 0.5 * (np.abs(f2 - _shift_np(f2, 0, 1)) + np.abs(f2 - _shift_np(f2, 0, -1)))   # 좌우 대칭 가장자리
    ey = 0.5 * (np.abs(f2 - _shift_np(f2, 1, 0)) + np.abs(f2 - _shift_np(f2, -1, 0)))
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

    from camera.camfly.flybrain import _pool_t, _shift_t
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:

    def _loom_t(f1, f2, n_sect):
        B, R, C = f2.shape
        v = ((f1 > 0) & (f2 > 0)).float()
        lo = torch.clamp(F.relu(f2 - f1 - LOOM_TH) / (f1 + LOOM_EPS), 0, 1) * v
        s = lo.reshape(B, R, n_sect, C // n_sect).sum(dim=(1, 3))
        n = v.reshape(B, R, n_sect, C // n_sect).sum(dim=(1, 3))
        return s / torch.clamp(n, min=1.0)


if torch is not None:

    class DepthFlyBrain(nn.Module):
        def __init__(self, n_dn: int = 48, fan_in: int = 12, seed: int = 0, n_sect: int = N_SECT,
                     n_band: int = N_BAND, wiring: str = "bilateral"):
            super().__init__()
            self.n_sect, self.n_band = n_sect, n_band
            n_in = n_inputs(n_sect, n_band)
            mask, sign = make_wiring(wiring, n_in, n_dn, fan_in, seed)
            p = init_params(n_in, n_dn, seed)
            if wiring == "bilateral":
                p = symmetric_init(p, n_dn)
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
            dn, dp = dn / (dn.abs() + SIGMA_LAM), dp / (dp.abs() + SIGMA_LAM)
            on_f, on_s, off_f, off_s = F.relu(dn), F.relu(dp), F.relu(-dn), F.relu(-dp)
            g4, g5 = F.softplus(self.g_t4), F.softplus(self.g_t5)
            gl = F.softplus(self.g_lp) / LN2 * self.g_lp.new_tensor(GAIN0)
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
            lplc2 = _loom_t(f1, f2, ns) * gl[2]
            ex = 0.5 * ((f2 - _shift_t(f2, 0, 1)).abs() + (f2 - _shift_t(f2, 0, -1)).abs())
            ey = 0.5 * ((f2 - _shift_t(f2, 1, 0)).abs() + (f2 - _shift_t(f2, -1, 0)).abs())
            lc = torch.cat([_pool_t(ex, ns, nb), _pool_t(ey, ns, nb)], dim=1) * gl[3]
            lum = _pool_t(f2, ns, nb) * gl[4]
            near_max = f2.reshape(B, R, ns, C // ns).amax(dim=(1, 3)) * gl[4]
            return torch.cat([hs.reshape(B, -1), vs.reshape(B, -1), lplc2, lc.reshape(B, -1),
                              lum.reshape(B, -1), near_max], dim=1)

        def forward(self, near):
            x = self.features(near.float())
            W = self.sign * self.mask * F.softplus(self.w_dn)
            return torch.tanh(x @ W.t() + self.b_dn)
