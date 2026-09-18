"""
racetrack_loader.py
===================
f1tenth_racetracks (실제 F1/DTM 1:10 맵) centerline + 맵 이미지 로드.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

DEFAULT_RACETRACKS_DIR = Path(__file__).resolve().parent / "f1tenth_racetracks"


def list_race_maps(racetracks_dir: Path | str = DEFAULT_RACETRACKS_DIR) -> list[str]:
    root = Path(racetracks_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} 없음. 실행:\n"
            "  git clone https://github.com/f1tenth/f1tenth_racetracks.git f1tenth_racetracks"
        )
    names = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and next(d.glob("*centerline.csv"), None):
            names.append(d.name)
    return names


def load_centerline(map_name: str, racetracks_dir: Path | str = DEFAULT_RACETRACKS_DIR,
                    max_points: int = 600) -> tuple[np.ndarray, np.ndarray, Path | None]:
    """returns (centerline Nx2, half_widths N, map_png_path|None)"""
    root = Path(racetracks_dir) / map_name
    csv = next(root.glob("*centerline.csv"), None)
    if csv is None:
        raise FileNotFoundError(f"{root} 에 centerline.csv 없음")
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    xy = data[:, :2].astype(np.float64)
    # w_right, w_left
    half_w = 0.5 * (data[:, 2] + data[:, 3]).astype(np.float64)
    # 시뮬 동역학이 단순해서 실측 폭보다 약간 여유
    half_w = np.maximum(half_w * 1.6, 1.8)
    if len(xy) > max_points:
        idx = np.linspace(0, len(xy) - 1, max_points).astype(int)
        xy = xy[idx]
        half_w = half_w[idx]
    png = next(root.glob("*map.png"), None)
    return xy, half_w, png


def build_race_track_pool(
    map_names: list[str] | None = None,
    racetracks_dir: Path | str = DEFAULT_RACETRACKS_DIR,
) -> dict[str, dict]:
    """name -> {centerline, half_widths, map_png}"""
    names = map_names or ["Spielberg", "MoscowRaceway", "Silverstone"]
    pool = {}
    for name in names:
        cl, hw, png = load_centerline(name, racetracks_dir)
        pool[name] = {"centerline": cl, "half_widths": hw, "map_png": png}
        L = float(np.sum(np.linalg.norm(np.diff(cl, axis=0, append=cl[:1]), axis=1)))
        print(f"[racetrack_loader] {name}: {len(cl)} pts, length~{L:.0f}m")
    return pool


if __name__ == "__main__":
    print(list_race_maps())
    build_race_track_pool(["Spielberg", "Monza"])
