"""
mapless40.export
================
학습된 SAC zip → 실차용 결정적 actor.

  python -m mapless40.export runs/mapless40_conv1d_xxx/best_model.zip [--onnx]

산출물 (zip 과 같은 폴더):
  actor.ts.pt      TorchScript  (scan[1,4,N] float32, state[1,17] float32) → a[1,2] ∈ [−1,1]
  actor.onnx       (--onnx)     TensorRT 변환용
  actor_meta.json  실차 노드가 읽는 전처리·정규화·명령 변환 상수 (sim 과 동일해야 함)

Jetson 에는 actor.ts.pt 와 actor_meta.json 두 파일만 복사하면 된다 (SB3 불필요, 인터넷 불필요).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .config import EnvConfig
from .policy import DeterministicActor


def export_actor(model_zip: str | Path, out_dir: str | Path | None = None, onnx: bool = False) -> Path:
    model_zip = Path(model_zip)
    out_dir = Path(out_dir) if out_dir else model_zip.parent
    model = SAC.load(str(model_zip), device="cpu")
    actor = DeterministicActor(model.policy.actor).eval()

    obs_space = model.observation_space
    hist, n_beams = obs_space["scan"].shape
    scan = torch.rand(1, hist, n_beams)
    state = torch.randn(1, obs_space["state"].shape[0]) * 0.3

    # SB3 predict 와 일치하는지 확인
    with torch.no_grad():
        a_ref, _ = model.predict({"scan": scan[0].numpy().astype(np.float16),
                                  "state": state[0].numpy(),
                                  "priv": np.zeros(obs_space["priv"].shape, np.float32)},
                                 deterministic=True)
        a_exp = actor(scan.half().float(), state)[0].numpy()
    err = float(np.abs(a_ref - a_exp).max())
    print(f"[export] actor vs SB3.predict max|Δa| = {err:.2e}")
    assert err < 1e-4, "export actor 가 SB3 정책과 다르다"

    ts_path = out_dir / "actor.ts.pt"
    traced = torch.jit.trace(actor, (scan, state), check_trace=False)
    traced.save(str(ts_path))
    print(f"[export] {ts_path}")

    if onnx:
        onnx_path = out_dir / "actor.onnx"
        torch.onnx.export(actor, (scan, state), str(onnx_path), input_names=["scan", "state"],
                          output_names=["action"], opset_version=17)
        print(f"[export] {onnx_path}")

    cfg_json = model_zip.parent / "config.json"
    env_cfg = json.loads(cfg_json.read_text(encoding="utf-8"))["env"] if cfg_json.exists() \
        else EnvConfig().to_dict()
    meta = {
        "hist": hist, "n_beams": n_beams,
        "lidar": env_cfg["lidar"], "norm": env_cfg["norm"],
        "act": env_cfg["act"], "action": env_cfg["action"],
        "encoder": model.policy.features_extractor_kwargs.get("encoder", "conv1d"),
        "source": str(model_zip),
    }
    (out_dir / "actor_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] {out_dir / 'actor_meta.json'}")
    return ts_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_zip")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--onnx", action="store_true")
    a = p.parse_args()
    export_actor(a.model_zip, a.out_dir, a.onnx)
