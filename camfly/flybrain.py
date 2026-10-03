"""
camfly.flybrain
===============
초파리 시각엽 → 하행 뉴런(DN) 회로 (합성 커넥톰 1단계).  CNN·MLP 없이 회로 구조로만 특징을 만든다.

입력: eye (B, 3, R, C) — 파리 눈 프레임 n-2, n-1, n (광수용체 적응 후),  dscan (B, 3, C) — depth 가짜 스캔

  lamina      L_ON  = relu(Δ),  L_OFF = relu(−Δ)              Δ_now = f_n − f_{n−1},  Δ_prev = f_{n−1} − f_{n−2}
  medulla     빠른 갈래 = Δ_now 경로, 느린 갈래 = Δ_prev 경로 (Mi1/Tm3 vs Mi4/Mi9 의 시간 지연 역할)
  T4 (ON) / T5 (OFF)  방향 선택 운동 검출 (Hassenstein–Reichardt 상관기), 4방향 (왼·오·위·아래)
              T_d(x) = relu( g_d · [ slow(x)·fast(x+e_d) − fast(x)·slow(x+e_d) ] )
  lobula plate
      HS  (수평 계열)   섹터·밴드별  (왼쪽으로 흐름 − 오른쪽으로 흐름)
      VS  (수직 계열)   섹터·밴드별  (아래 − 위)
      LPLC2 (다가옴)   섹터별 바깥쪽 흐름 (왼쪽 시야의 왼쪽 흐름 + 오른쪽 시야의 오른쪽 흐름 + 위·아래로 벌어짐)
  lobula      LC  섹터·밴드별 공간 대비 (가장자리) — 벽·장애물 위치
  depth       섹터별 최근접 거리, 다가오는 속도 (파리에는 없는 감각 — DN 으로 바로)
  DN          고정 희소 연결(부호 고정) × 학습되는 세기  (FLYNN 방식: 토폴로지 고정, 세기만 학습)

학습되는 것: T4/T5 방향별 이득 8개, LPTC·LC 이득 4개, DN 연결 세기(마스크 안의 값), DN 바이어스.
출력: DN 활동 (n_dn) — 정책은 여기서 선형으로 (조향, 속도) 를 읽는다.

numpy 참조(forward_numpy) 와 torch 모듈(FlyBrain) 이 같은 수식 — tests 에서 비교.
"""

from __future__ import annotations

import numpy as np

N_SECT = 8          # 가로 섹터 (64 열 / 8)
N_BAND = 2          # 세로 밴드 (위: 덕트·배경, 아래: 바닥·가까운 물체)
# 방향: (행 이동, 열 이동). 열 0 = 오른쪽 끝, 열 증가 = 왼쪽.  행 0 = 위.
DIRS = {"left": (0, 1), "right": (0, -1), "down": (1, 0), "up": (-1, 0)}
DIR_NAMES = list(DIRS)


def feature_layout(n_sect: int = N_SECT, n_band: int = N_BAND) -> dict[str, int]:
    return {"HS": n_sect * n_band, "VS": n_sect * n_band, "LPLC2": n_sect,
            "LC": 2 * n_sect * n_band, "DEPTH": 2 * n_sect}


def n_inputs(n_sect: int = N_SECT, n_band: int = N_BAND) -> int:
    return sum(feature_layout(n_sect, n_band).values())


def make_dn_wiring(n_in: int, n_dn: int = 48, fan_in: int = 12, seed: int = 0):
    """합성 DN 배선: DN 마다 입력 fan_in 개, 부호 ±1 (흥분/억제) 고정. (n_dn, n_in) 마스크·부호."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((n_dn, n_in), np.float32)
    sign = np.zeros((n_dn, n_in), np.float32)
    for i in range(n_dn):
        idx = rng.choice(n_in, size=min(fan_in, n_in), replace=False)
        mask[i, idx] = 1.0
        sign[i, idx] = rng.choice([-1.0, 1.0], size=idx.size, p=[0.4, 0.6])
    return mask, sign


def init_params(n_in: int, n_dn: int = 48, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    return {
        "g_t4": np.zeros(4, np.float32), "g_t5": np.zeros(4, np.float32),   # softplus(0)=0.69
        "g_lp": np.zeros(4, np.float32),                                    # HS, VS, LPLC2, LC 이득
        "w_dn": rng.normal(0.0, 0.5, (n_dn, n_in)).astype(np.float32),      # 세기 = softplus(w)
        "b_dn": np.zeros(n_dn, np.float32),
    }


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _shift_np(x, dr, dc):
    """x[..., r, c] → x[..., r+dr, c+dc] 위치의 값 (밖은 0). 마지막 두 축 = (R, C)."""
    out = np.zeros_like(x)
    R, C = x.shape[-2], x.shape[-1]
    rs, re = max(0, -dr), min(R, R - dr)
    cs, ce = max(0, -dc), min(C, C - dc)
    out[..., rs:re, cs:ce] = x[..., rs + dr:re + dr, cs + dc:ce + dc]
    return out


def _pool_np(x, n_sect, n_band, how="mean"):
    """(B, R, C) → (B, n_band, n_sect)."""
    B, R, C = x.shape
    y = x.reshape(B, n_band, R // n_band, n_sect, C // n_sect)
    return y.mean(axis=(2, 4)) if how == "mean" else y.min(axis=(2, 4))


def visual_features_numpy(eye, dscan, p, n_sect=N_SECT, n_band=N_BAND):
    """eye (B,3,R,C), dscan (B,3,C) → (B, n_inputs) 시각·depth 특징."""
    f0, f1, f2 = eye[:, 0], eye[:, 1], eye[:, 2]
    dn, dp = f2 - f1, f1 - f0
    on_f, on_s = np.maximum(dn, 0), np.maximum(dp, 0)
    off_f, off_s = np.maximum(-dn, 0), np.maximum(-dp, 0)
    g4, g5, gl = _softplus(p["g_t4"]), _softplus(p["g_t5"]), _softplus(p["g_lp"])
    mot = {}
    for k, name in enumerate(DIR_NAMES):
        dr, dc = DIRS[name]
        t4 = np.maximum(g4[k] * (on_s * _shift_np(on_f, dr, dc) - on_f * _shift_np(on_s, dr, dc)), 0)
        t5 = np.maximum(g5[k] * (off_s * _shift_np(off_f, dr, dc) - off_f * _shift_np(off_s, dr, dc)), 0)
        mot[name] = t4 + t5
    B = eye.shape[0]
    hs = _pool_np(mot["left"] - mot["right"], n_sect, n_band) * gl[0]
    vs = _pool_np(mot["down"] - mot["up"], n_sect, n_band) * gl[1]
    # 다가옴: 시야 왼쪽 절반(열 큰 쪽)은 왼쪽 흐름, 오른쪽 절반은 오른쪽 흐름 + 위 밴드 위로, 아래 밴드 아래로
    C = eye.shape[-1]
    half = np.zeros_like(mot["left"])
    half[..., C // 2:] = mot["left"][..., C // 2:]
    half[..., :C // 2] = mot["right"][..., :C // 2]
    R = eye.shape[-2]
    vert = np.concatenate([mot["up"][..., :R // 2, :], mot["down"][..., R // 2:, :]], axis=-2)
    lplc2 = _pool_np(half + vert, n_sect, 1)[:, 0, :] * gl[2]
    ex = np.abs(f2 - _shift_np(f2, 0, 1))
    ey = np.abs(f2 - _shift_np(f2, 1, 0))
    lc = np.concatenate([_pool_np(ex, n_sect, n_band), _pool_np(ey, n_sect, n_band)], axis=1) * gl[3]
    d_now = dscan[:, -1].reshape(B, n_sect, -1).min(axis=-1)
    d_prev = dscan[:, -2].reshape(B, n_sect, -1).min(axis=-1)
    depth = np.concatenate([d_now, (d_prev - d_now) * 10.0], axis=1)        # 거리, 다가오는 속도
    return np.concatenate([hs.reshape(B, -1), vs.reshape(B, -1), lplc2, lc.reshape(B, -1), depth], axis=1)


def forward_numpy(eye, dscan, p, mask, sign, n_sect=N_SECT, n_band=N_BAND):
    """→ DN 활동 (B, n_dn) ∈ (−1, 1)."""
    x = visual_features_numpy(eye.astype(np.float32), dscan.astype(np.float32), p, n_sect, n_band)
    W = sign * mask * _softplus(p["w_dn"])
    return np.tanh(x @ W.T + p["b_dn"])


# ------------------------------------------------------------------------- torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:

    def _shift_t(x, dr, dc):
        """out[..., r, c] = x[..., r+dr, c+dc] (밖은 0) — _shift_np 와 같음."""
        R, C = x.shape[-2], x.shape[-1]
        P = max(abs(dr), abs(dc))
        xp = F.pad(x, (P, P, P, P))
        return xp[..., P + dr:P + dr + R, P + dc:P + dc + C]

    def _pool_t(x, n_sect, n_band):
        B, R, C = x.shape
        return x.reshape(B, n_band, R // n_band, n_sect, C // n_sect).mean(dim=(2, 4))

    class FlyBrain(nn.Module):
        """합성 커넥톰 시각엽 + DN.  forward(eye, dscan) → (B, n_dn)."""

        def __init__(self, n_dn: int = 48, fan_in: int = 12, seed: int = 0,
                     n_sect: int = N_SECT, n_band: int = N_BAND):
            super().__init__()
            self.n_sect, self.n_band, self.n_dn = n_sect, n_band, n_dn
            n_in = n_inputs(n_sect, n_band)
            mask, sign = make_dn_wiring(n_in, n_dn, fan_in, seed)
            p = init_params(n_in, n_dn, seed)
            self.register_buffer("mask", torch.from_numpy(mask))
            self.register_buffer("sign", torch.from_numpy(sign))
            self.g_t4 = nn.Parameter(torch.from_numpy(p["g_t4"]))
            self.g_t5 = nn.Parameter(torch.from_numpy(p["g_t5"]))
            self.g_lp = nn.Parameter(torch.from_numpy(p["g_lp"]))
            self.w_dn = nn.Parameter(torch.from_numpy(p["w_dn"]))
            self.b_dn = nn.Parameter(torch.from_numpy(p["b_dn"]))
            self.features_dim = n_dn

        def numpy_params(self) -> dict:
            return {k: getattr(self, k).detach().cpu().numpy() for k in ("g_t4", "g_t5", "g_lp", "w_dn", "b_dn")}

        def visual_features(self, eye, dscan):
            f0, f1, f2 = eye[:, 0], eye[:, 1], eye[:, 2]
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
            ex = (f2 - _shift_t(f2, 0, 1)).abs()
            ey = (f2 - _shift_t(f2, 1, 0)).abs()
            lc = torch.cat([_pool_t(ex, ns, nb), _pool_t(ey, ns, nb)], dim=1) * gl[3]
            d_now = dscan[:, -1].reshape(B, ns, -1).amin(dim=-1)
            d_prev = dscan[:, -2].reshape(B, ns, -1).amin(dim=-1)
            depth = torch.cat([d_now, (d_prev - d_now) * 10.0], dim=1)
            return torch.cat([hs.reshape(B, -1), vs.reshape(B, -1), lplc2, lc.reshape(B, -1), depth], dim=1)

        def forward(self, eye, dscan):
            x = self.visual_features(eye.float(), dscan.float())
            W = self.sign * self.mask * F.softplus(self.w_dn)
            return torch.tanh(x @ W.t() + self.b_dn)

    class SmallCNN(nn.Module):
        """비교용 일반 CNN (같은 입력)."""

        def __init__(self, out_dim: int = 48, n_rows: int = 16, n_cols: int = 64):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
            )
            flat = 32 * (n_rows // 4) * (n_cols // 4)
            self.fc = nn.Sequential(nn.Linear(flat + 2 * n_cols, out_dim), nn.ReLU())
            self.features_dim = out_dim

        def forward(self, eye, dscan):
            z = self.cnn(eye.float())
            d = dscan[:, -2:].float().reshape(dscan.shape[0], -1)
            return self.fc(torch.cat([z, d], dim=1))
