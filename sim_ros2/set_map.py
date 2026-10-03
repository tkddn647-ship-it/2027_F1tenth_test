"""
f1tenth_gym_ros 시뮬 맵을 mapless40 학습 맵과 같게 맞춘다.

  python3 sim_ros2/set_map.py ifac              # 또는 roboracer_0817, Spielberg ...
  python3 sim_ros2/set_map.py ifac --s0 12.5    # 레이싱라인 위 s0 [m] 지점에서 출발

- 학습 env(load_track)가 쓰는 점유 격자를 그대로 흑백 PNG 로 저장 → gym 의 128 임계값 차이·
  Cartographer trinary(미탐색=벽) 처리 차이가 없도록.
- 출발 자세 = 레이싱라인 s0 지점 (x, y, 진행 방향) → 학습 때와 같은 주행 방향.
- sim.yaml 의 map_path / map_img_ext / sx / sy / stheta 를 고친다 (install 은 symlink 라 재빌드 불필요).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GYM_ROS = Path.home() / "f1tenth_ws/src/f1tenth_gym_ros"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map", help="ifac | roboracer_0817 | Spielberg ... (mapless40 --map 과 같은 이름)")
    p.add_argument("--s0", type=float, default=0.0, help="출발 지점 (레이싱라인 길이 방향 [m])")
    p.add_argument("--sim-yaml", default=str(GYM_ROS / "config/sim.yaml"))
    a = p.parse_args()

    from mapless40.config import EnvConfig
    from mapless40.env import load_track

    tr = load_track(a.map, EnvConfig())
    g, line = tr.grid, tr.line
    occ = g.occ

    name = f"mapless_{a.map}"
    out_dir = GYM_ROS / "maps"
    Image.fromarray(np.where(occ, 0, 255).astype(np.uint8)).save(out_dir / f"{name}.png")
    ox, oy = g.ox, g.oy
    (out_dir / f"{name}.yaml").write_text(
        f"image: {name}.png\nresolution: {g.res}\norigin: [{ox}, {oy}, 0.0]\n"
        f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n", encoding="utf-8")

    i = int(round(a.s0 / line.ds)) % line.n
    sx, sy, st = float(line.x[i]), float(line.y[i]), float(line.psi[i])

    sim = Path(a.sim_yaml)
    txt = sim.read_text(encoding="utf-8")
    for key, val in (("map_path", f"'{out_dir / name}'"), ("map_img_ext", "'.png'"),
                     ("sx", f"{sx:.4f}"), ("sy", f"{sy:.4f}"), ("stheta", f"{st:.4f}")):
        txt, n = re.subn(rf"^(\s*{key}:\s*)[^\n#]*", rf"\g<1>{val}", txt, count=1, flags=re.M)
        assert n == 1, f"sim.yaml 에 {key} 없음"
    sim.write_text(txt, encoding="utf-8")

    print(f"[set_map] {a.map}: {occ.shape[1]}x{occ.shape[0]} @ {g.res} m, line {line.length:.1f} m ({tr.line_file})")
    print(f"[set_map] map  → {out_dir / name}.yaml")
    print(f"[set_map] 출발 → sx={sx:.3f} sy={sy:.3f} stheta={st:.3f}  (sim.yaml 갱신)")


if __name__ == "__main__":
    main()
