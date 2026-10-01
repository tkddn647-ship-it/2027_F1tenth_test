"""
mapless40.np_actor
==================
torch 없이 SB3 SAC zip(best_model.zip · last_model.zip · checkpoints/*.zip)을 읽어
배포용 actor(결정적 tanh(μ))를 numpy 로 돌린다.  encoder = conv1d 전용.

  from mapless40.np_actor import NumpyActor
  act = NumpyActor("best_model.zip")
  a = act(obs)           # obs: env 의 Dict 관측 (scan (4,1125), state (17,))

용도: torch 없는 PC 에서 시뮬 평가 / 결과 확인.  (실차 노드는 TorchScript 를 쓴다.)
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from collections import OrderedDict

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
    "BFloat16Storage": None,
}


class _Storage:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *args, **kw):
    arr = storage
    if len(size) == 0:
        return arr[offset].copy()
    item = arr.itemsize
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


def load_torch_state_dict(raw: bytes) -> dict:
    """torch.save(dict) 로 쓴 바이트 → {이름: np.ndarray}.  torch 불필요 (zip 직렬화 형식)."""
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    pkl = next(n for n in names if n.endswith("data.pkl"))
    prefix = pkl[: -len("data.pkl")]
    cache: dict[str, np.ndarray] = {}

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, requires_grad, hooks, *a: data
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            if module == "torch" and name.endswith("Storage"):
                return _Storage(name)
            if module == "torch" and name in ("float32", "float16", "float64", "int64", "bool"):
                return name
            return super().find_class(module, name)

        def persistent_load(self, pid):
            # ('storage', storage_type, key, location, numel)
            _, stype, key, _loc, numel = pid
            if key not in cache:
                name = stype.name if isinstance(stype, _Storage) else str(stype)
                dt = _DTYPES.get(name, np.float32)
                if dt is None:
                    raise ValueError("bfloat16 저장값은 지원 안 함")
                buf = zf.read(f"{prefix}data/{key}")
                cache[key] = np.frombuffer(buf, dtype=dt, count=numel)
            return cache[key]

    return dict(U(io.BytesIO(zf.read(pkl))).load())


def model_obs_cfg(path: str) -> tuple[dict | None, dict | None]:
    """SB3 zip 에 기록된 학습 때 LiDAR·정규화 설정 (lidar_cfg, norm_cfg). torch 불필요.

    range_max 등이 바뀐 뒤 옛 모델을 돌릴 때 입력을 학습 때와 똑같이 만들려고 쓴다.
    """
    try:
        with zipfile.ZipFile(path) as z:
            d = json.loads(z.read("data"))
        kw = d["policy_kwargs"]["features_extractor_kwargs"]
        return kw.get("lidar_cfg"), kw.get("norm_cfg")
    except Exception:
        return None, None


def apply_model_cfg(cfg, path: str) -> None:
    """EnvConfig 의 lidar·norm 을 모델이 학습된 값으로 덮어쓴다 (.zip 만)."""
    if not str(path).endswith(".zip"):
        return
    lc, nc = model_obs_cfg(path)
    for k, v in (lc or {}).items():
        if hasattr(cfg.lidar, k):
            setattr(cfg.lidar, k, v)
    for k, v in (nc or {}).items():
        if hasattr(cfg.norm, k):
            setattr(cfg.norm, k, v)


def load_sb3_zip(path: str) -> tuple[dict, dict]:
    """SB3 zip → (policy state_dict numpy, data json)."""
    with zipfile.ZipFile(path) as z:
        sd = load_torch_state_dict(z.read("policy.pth"))
        try:
            data = json.loads(z.read("data"))
        except Exception:
            data = {}
    return sd, data


class NumpyActor:
    """policy.DeterministicActor 의 numpy 판 (encoder=conv1d)."""

    def __init__(self, path: str):
        from .policy_spec import CONV1D_LAYERS
        sd, self.data = load_sb3_zip(path)
        self.num_timesteps = int(self.data.get("num_timesteps", 0))
        p = "actor.features_extractor."
        if f"{p}scan_enc.conv.0.weight" not in sd:
            raise ValueError("conv1d 인코더 모델만 지원 (bev/both 는 torch 로 평가)")
        g = lambda k: sd[k].astype(np.float32)  # noqa: E731
        self.convs = []
        for i, (_, _, s, pad) in enumerate(CONV1D_LAYERS):
            self.convs.append((g(f"{p}scan_enc.conv.{2 * i}.weight"), g(f"{p}scan_enc.conv.{2 * i}.bias"), s, pad))
        self.fc = (g(f"{p}scan_enc.fc.1.weight"), g(f"{p}scan_enc.fc.1.bias"))
        self.st = (g(f"{p}state_enc.0.weight"), g(f"{p}state_enc.0.bias"))
        self.mlp = []
        i = 0
        while f"actor.latent_pi.{i}.weight" in sd:
            self.mlp.append((g(f"actor.latent_pi.{i}.weight"), g(f"actor.latent_pi.{i}.bias")))
            i += 2
        self.mu = (g("actor.mu.weight"), g("actor.mu.bias"))
        c_last = self.convs[-1][0].shape[0]
        self.pool_n = self.fc[0].shape[1] // c_last

    @staticmethod
    def _conv(x, w, b, stride, pad):
        # x (N, C, L) 배치 conv1d
        n, c, L = x.shape
        o, _, k = w.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
        Lo = (L + 2 * pad - k) // stride + 1
        idx = np.arange(Lo)[:, None] * stride + np.arange(k)[None, :]
        cols = xp[:, :, idx].transpose(0, 1, 3, 2).reshape(n, c * k, Lo)   # (N, C·K, Lo)
        return np.matmul(w.reshape(o, c * k)[None], cols) + b[None, :, None]

    def __call__(self, obs, env=None) -> np.ndarray:
        scan = np.asarray(obs["scan"], np.float32)             # (T, N)
        x = scan[:, None, :]
        for w, b, s, pad in self.convs:
            x = np.maximum(self._conv(x, w, b, s, pad), 0.0)
        T, C, L = x.shape
        k = max(1, L // 8)
        n = (L - k) // k + 1
        assert n == self.pool_n, (n, self.pool_n)
        pooled = x[:, :, :k * n].reshape(T, C, n, k).mean(-1).reshape(T, -1)
        z = np.maximum(pooled @ self.fc[0].T + self.fc[1], 0.0).reshape(-1)       # (T·48,)
        s = np.maximum(self.st[0] @ np.asarray(obs["state"], np.float32) + self.st[1], 0.0)
        h = np.concatenate([z, s])
        for w, b in self.mlp:
            h = np.maximum(w @ h + b, 0.0)
        return np.tanh(self.mu[0] @ h + self.mu[1]).astype(np.float32)
