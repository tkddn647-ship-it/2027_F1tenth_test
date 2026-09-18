#!/usr/bin/env python3
"""
맵 이미지에서 벽(장애물) 사이의 도로 한가운데 센터라인을 자동 추출해 CSV로 저장합니다.

- 입력: ROS 맵 YAML (image, resolution, origin, free_thresh). 어떤 맵을 들고 와도 --map 으로 지정하면 됨.
- 기본 맵: Spielberg_map.yaml (f1tenth_gym_ros/maps)
- 방식: mat(기본)=거리변환(D)+국소최대(ridge)=센터라인→스켈레톤 그래프 가지치기→최대 루프→호길이 리샘플. (검은선=벽, 흰바탕=도로면 --invert-free)
- 출력: x,y CSV (waypoint_follow / centerline_path_node 에서 사용)
- 좌표 수식: ROS와 동일 (x=origin_x+col*resolution, y=origin_y+(height-1-row)*resolution). 수식 문제 아님.
- 정상 CSV(Spielberg_centerline.csv)처럼 첫 점 (0,0)으로 맞추려면 --start-at-zero. RVIZ에서 라인이 벽에 붙으면 --invert-free 시도.
- 원점/차량 기준: 라인 추출 후 시뮬·맵에서 쓰는 원점에 맞게 좌표 재조정하려면 --origin-x, --origin-y (또는 --origin-from-map). 출력이 (x - origin_x, y - origin_y)로 저장됨.

의존성: numpy, scipy, PyYAML, Pillow. (선택) scikit-image → ridge를 1픽셀 두께로 얇게 해서 한 바퀴만 나오게 함.)

절차(기본 mat): 맵 이진화(검은선=벽, 도로=free) → 거리변환 D → D 국소최대=ridge=센터라인
→ 스켈레톤 그래프 가지치기 → 최대 루프 → 호길이 리샘플. (맵에서 흰바탕=도로면 --invert-free)

사용 예:
  python3 scripts/extract_centerline_from_map.py
  python3 scripts/extract_centerline_from_map.py --map /path/to/맵.yaml --out config/센터라인.csv
  python3 scripts/extract_centerline_from_map.py --invert-free   # 검은선=벽, 흰바탕=도로일 때
  python3 scripts/extract_centerline_from_map.py --origin-x 84.85 --origin-y 36.30   # 시뮬 원점 기준으로 재조정
  python3 scripts/extract_centerline_from_map.py --origin-from-map   # 맵 YAML origin 기준 상대 좌표
"""
import argparse
import csv
import os
import sys

try:
    import numpy as np
    import yaml
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, maximum_filter
except ImportError as e:
    print("Missing dependency:", e, file=sys.stderr)
    print("Install with: pip install numpy scipy pyyaml Pillow", file=sys.stderr)
    sys.exit(1)

try:
    from skimage.morphology import skeletonize as _skeletonize_thin
    from skimage.morphology import medial_axis as _medial_axis_thin
except ImportError:
    _skeletonize_thin = None
    _medial_axis_thin = None


def load_map(yaml_path: str, invert_free: bool = False):
    """ROS 맵 YAML + 이미지 로드. free_mask=1 = 도로, 0 = 벽.
    invert_free=True: 이미지에서 밝은 쪽(높은 값)=도로로 해석 (맵에 따라 필요)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    img_path = meta["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    resolution = float(meta["resolution"])
    origin = meta["origin"]
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    free_thresh = float(meta.get("free_thresh", 0.196))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))

    img = np.array(Image.open(img_path).convert("L"))
    if meta.get("negate", 0):
        img = 255 - img
    max_val = 255.0 if img.max() > 100 else 100.0
    # Spielberg 등 많은 맵: 흰색(255)=도로, 검은색(0)=벽. ROS YAML 기본 해석은 반대일 수 있음.
    # 검은선=벽, 흰바탕=도로 → --invert-free. 어두운쪽=도로 → 기본.
    if invert_free:
        free_mask = (img >= (1.0 - free_thresh) * max_val).astype(np.uint8)
    else:
        free_mask = (img <= free_thresh * max_val).astype(np.uint8)
    return free_mask, resolution, origin_x, origin_y, img.shape


def _row_run_center(free_inds: np.ndarray) -> float | None:
    """연속 구간들 중 가장 긴 구간의 중앙 인덱스(실수)."""
    if len(free_inds) == 0:
        return None
    gaps = np.diff(free_inds) > 1
    run_starts = np.concatenate([[0], np.where(gaps)[0] + 1])
    run_ends = np.concatenate([np.where(gaps)[0] + 1, [len(free_inds)]])
    best_len = 0
    best_mid = None
    for i in range(len(run_starts)):
        s, e = run_starts[i], run_ends[i]
        if e - s > best_len:
            best_len = e - s
            best_mid = (free_inds[s] + free_inds[e - 1]) / 2.0
    return best_mid


def row_col_midpoints(free_mask: np.ndarray, use_both: bool = True):
    """행·열 각각에서 free 구간 정중앙. use_both=True면 행+열 합쳐서 곡선에서도 진짜 중심에 가깝게."""
    height, width = free_mask.shape
    points = []
    # 행 기준: (r, c_mid)
    for r in range(height):
        free_cols = np.where(free_mask[r, :] > 0)[0]
        c_mid = _row_run_center(free_cols)
        if c_mid is not None:
            points.append((float(r), c_mid))
    if not use_both:
        return points
    # 열 기준: (r_mid, c) → 곡선 구간에서 행만 썼을 때 생기는 치우침 보정
    for c in range(width):
        free_rows = np.where(free_mask[:, c] > 0)[0]
        r_mid = _row_run_center(free_rows)
        if r_mid is not None:
            points.append((r_mid, float(c)))
    return points


def merge_near_points(points: list, tol: float = 1.5) -> list:
    """거리 tol 이내 점 하나로 합치기 (중심)."""
    if len(points) <= 1:
        return points
    pts = np.array(points)
    used = np.zeros(len(pts), dtype=bool)
    out = []
    for i in range(len(pts)):
        if used[i]:
            continue
        near = np.sum((pts - pts[i]) ** 2, axis=1) <= tol * tol
        used[near] = True
        out.append(tuple(np.mean(pts[near], axis=0)))
    return out


def order_points_nearest(points: list, start_index: int | None = None) -> list:
    """포인트들을 가까운 순서로 연결해 경로 순서로 재배열. start_index 없으면 centroid에서 가장 먼 점부터."""
    if len(points) <= 1:
        return points
    pts = np.array(points)
    if start_index is None:
        cen = np.mean(pts, axis=0)
        dists = np.sum((pts - cen) ** 2, axis=1)
        start_index = int(np.argmax(dists))
    ordered = [start_index]
    used = {start_index}
    for _ in range(len(pts) - 1):
        cur = pts[ordered[-1]]
        dists = np.sum((pts - cur) ** 2, axis=1)
        dists[list(used)] = np.inf
        nxt = int(np.argmin(dists))
        ordered.append(nxt)
        used.add(nxt)
    return [tuple(pts[i]) for i in ordered]


def refocus_centerline(free_mask: np.ndarray, ordered_path: list, tangent_lookahead: int = 15) -> list:
    """경로를 따라 가며 각 점에서 왼쪽 벽·오른쪽 벽 사이 정중앙으로 보정. (도로 한가운데만).
    tangent_lookahead: 접선을 앞뒤 몇 칸 평균할지."""
    height, width = free_mask.shape
    out = []
    n = len(ordered_path)
    L = max(1, tangent_lookahead)
    for i in range(n):
        r, c = float(ordered_path[i][0]), float(ordered_path[i][1])
        i_prev = max(0, i - L)
        i_next = min(n - 1, i + L)
        if i_prev == i_next:
            out.append((r, c))
            continue
        dr = ordered_path[i_next][0] - ordered_path[i_prev][0]
        dc = ordered_path[i_next][1] - ordered_path[i_prev][1]
        norm = np.sqrt(dr * dr + dc * dc)
        if norm < 1e-6:
            out.append((r, c))
            continue
        perp_r = -dc / norm
        perp_c = dr / norm
        # +perp 방향으로 벽(occupied)까지 걸음 수
        k1 = 0
        while True:
            rr = int(round(r + (k1 + 1) * perp_r))
            cc = int(round(c + (k1 + 1) * perp_c))
            if rr < 0 or rr >= height or cc < 0 or cc >= width:
                break
            if free_mask[rr, cc] == 0:  # 벽
                break
            k1 += 1
        k2 = 0
        while True:
            rr = int(round(r - (k2 + 1) * perp_r))
            cc = int(round(c - (k2 + 1) * perp_c))
            if rr < 0 or rr >= height or cc < 0 or cc >= width:
                break
            if free_mask[rr, cc] == 0:
                break
            k2 += 1
        # 벽–벽 정중앙 (도로 한가운데)
        r_center = r + (k1 - k2) * 0.5 * perp_r
        c_center = c + (k1 - k2) * 0.5 * perp_c
        # 이미지 범위 안으로 클램프
        r_center = max(0.0, min(height - 1.0, r_center))
        c_center = max(0.0, min(width - 1.0, c_center))
        out.append((r_center, c_center))
    return out


def centerline_radial(free_mask: np.ndarray, num_angles: int = 720) -> list:
    """각도별: 중심에서 레이를 쏴 free 구간(도로)을 지나면 왼쪽벽·오른쪽벽(첫/끝) 사이 정중앙. (방식 자체는 표준.)"""
    height, width = free_mask.shape
    # 도로 무게중심
    ys, xs = np.where(free_mask > 0)
    if len(xs) == 0:
        return []
    cr = float(np.mean(ys))
    cc = float(np.mean(xs))
    out = []
    for j in range(num_angles):
        theta = 2.0 * np.pi * j / num_angles
        dr, dc = np.sin(theta), np.cos(theta)
        # 레이 상에서 free 구간 [t_in, t_out] 찾기 (픽셀 단위)
        t = 0
        t_in, t_out = None, None
        max_steps = max(height, width) * 2
        while t < max_steps:
            r = int(round(cr + t * dr))
            c = int(round(cc + t * dc))
            if r < 0 or r >= height or c < 0 or c >= width:
                t += 1
                continue
            if free_mask[r, c] > 0:
                if t_in is None:
                    t_in = t
                t_out = t
            else:
                if t_in is not None and t_out is not None:
                    break
            t += 1
        if t_in is not None and t_out is not None:
            t_mid = (t_in + t_out) * 0.5
            r_center = cr + t_mid * dr
            c_center = cc + t_mid * dc
            out.append((r_center, c_center))
    return out


def remove_duplicate_points(points: list, min_dist_px: float = 0.5) -> list:
    """연속된 점이 min_dist_px 미만이면 하나만 유지 (겹침/한줄 제거)."""
    if len(points) <= 1:
        return points
    out = [points[0]]
    for i in range(1, len(points)):
        r0, c0 = out[-1][0], out[-1][1]
        r1, c1 = points[i][0], points[i][1]
        if (r1 - r0) ** 2 + (c1 - c0) ** 2 >= min_dist_px ** 2:
            out.append(points[i])
    return out


def filter_centerline_away_from_walls(
    points: list, free_mask: np.ndarray, min_dist_from_wall_px: float = 2.0
) -> list:
    """벽에서 min_dist_from_wall_px 픽셀 이상 떨어진 점만 유지 → 라인이 벽 위가 아니라 흰 부분(도로)에만 오도록."""
    if len(points) <= 1 or min_dist_from_wall_px <= 0:
        return points
    dist = distance_transform_edt(free_mask > 0)
    out = []
    h, w = free_mask.shape
    for (r, c) in points:
        ri, ci = int(round(r)), int(round(c))
        if ri < 0 or ri >= h or ci < 0 or ci >= w:
            continue
        if dist[ri, ci] >= min_dist_from_wall_px:
            out.append((r, c))
    if len(out) < 2:
        return points
    return out


def medial_axis_from_free_mask(
    free_mask: np.ndarray,
    min_dist: float = 0.5,
    max_dist: float | None = None,
) -> np.ndarray:
    """도로(free=1)에서 거리변환 D(x,y)=가장 가까운 벽까지 거리 → D가 국소 최대인 점 = 센터라인(ridge).
    max_dist: 이 값(픽셀)보다 먼 점은 제외. 검은 두 선 사이만 도로·바깥은 흰 맵에서 트랙 밖 넓은 흰 영역 제거용."""
    dist = distance_transform_edt(free_mask > 0)
    local_max = maximum_filter(dist, size=5, mode="constant", cval=0)
    ridge = (dist >= local_max - 1e-9) & (free_mask > 0) & (dist > min_dist)
    if max_dist is not None and max_dist > 0:
        ridge = ridge & (dist <= max_dist)
    return ridge.astype(np.uint8)


def thin_ridge_to_skeleton(ridge: np.ndarray, use_medial_axis: bool = True) -> np.ndarray:
    """Ridge(두꺼운 줄)를 1픽셀 두께로. use_medial_axis=True면 medial_axis 먼저 시도(연결 유지에 유리)."""
    if _medial_axis_thin is None and _skeletonize_thin is None:
        return ridge
    r = ridge.astype(bool)
    # medial_axis가 루프 끊김을 덜 일으키는 경우가 있음
    if use_medial_axis and _medial_axis_thin is not None:
        out = _medial_axis_thin(r).astype(np.uint8)
        if out.sum() > 0:
            return out
    if _skeletonize_thin is not None:
        out = _skeletonize_thin(r).astype(np.uint8)
        return out
    return ridge


def pixel_to_world(row: float, col: float, height: int, resolution: float, origin_x: float, origin_y: float):
    """Map image (row,col) to map frame (x,y). ROS: row 0 = top, map y positive up."""
    x = origin_x + float(col) * resolution
    y = origin_y + (height - 1 - float(row)) * resolution
    return x, y


def skeleton_to_graph(skel: np.ndarray):
    """스켈레톤 픽셀 → 8-인접 그래프. 반환: pt_set, neighbors(r, c, pt_set)."""
    pts = np.argwhere(skel > 0)
    if len(pts) == 0:
        return set(), lambda r, c, s: []

    def neighbors(r, c, pt_set):
        return [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr != 0 or dc != 0) and (r + dr, c + dc) in pt_set
        ]
    return set(tuple(p) for p in pts), neighbors


def prune_skeleton_tips(pt_set: set, neighbors) -> set:
    """degree=1 인 끝점을 반복 제거(가지치기). 현재 집합 기준 이웃 사용."""
    out = set(pt_set)
    while True:
        to_remove = [p for p in out if len(neighbors(p[0], p[1], out)) <= 1]
        if not to_remove:
            break
        for p in to_remove:
            out.discard(p)
    return out


def extract_cycle_from_start(start, pt_set: set, neighbors) -> list:
    """start에서 시작해 한 바퀴 돌아올 때까지 순서대로 반환."""
    path = [start]
    cur = start
    prev = None
    for _ in range(len(pt_set) + 1):
        ne = [n for n in neighbors(cur[0], cur[1], pt_set) if n != prev]
        if not ne:
            break
        if len(ne) == 1:
            nxt = ne[0]
        else:
            if prev is not None:
                dr0 = cur[0] - prev[0]
                dc0 = cur[1] - prev[1]
                best = None
                best_dot = -2
                for n in ne:
                    dr1 = n[0] - cur[0]
                    dc1 = n[1] - cur[1]
                    norm = np.sqrt(dr1 * dr1 + dc1 * dc1)
                    if norm < 1e-6:
                        continue
                    dot = (dr0 * dr1 + dc0 * dc1) / norm
                    if dot > best_dot:
                        best_dot = dot
                        best = n
                nxt = best if best is not None else ne[0]
            else:
                nxt = ne[0]
        if nxt == start and len(path) > 2:
            break
        prev = cur
        cur = nxt
        path.append(cur)
    return path


def _path_arc_length(path: list, closed: bool = False) -> float:
    """경로(점 리스트)의 총 호길이(픽셀). closed=True면 끝→시작 구간 포함."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        r0, c0 = path[i][0], path[i][1]
        r1, c1 = path[i + 1][0], path[i + 1][1]
        total += np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    if closed and len(path) >= 3:
        r0, c0 = path[-1][0], path[-1][1]
        r1, c1 = path[0][0], path[0][1]
        total += np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    return total


def extract_largest_cycle(pt_set: set, neighbors) -> list:
    """가장 긴 사이클(맵 전체 한 바퀴)을 반환. 호길이(총 거리) 기준으로 선택."""
    if not pt_set:
        return []
    pts_list = list(pt_set)
    n_try = min(50, len(pts_list))
    indices = np.linspace(0, len(pts_list) - 1, n_try, dtype=int)
    starts = [pts_list[i] for i in indices]
    starts.insert(0, min(pts_list, key=lambda p: (p[0], p[1])))
    best_path = []
    best_length = 0.0
    for start in starts:
        path = extract_cycle_from_start(start, pt_set, neighbors)
        if len(path) < 3:
            continue
        length = _path_arc_length(path, closed=True)
        # 꼭짓점 수가 아니라 호길이(한 바퀴 둘레)로 비교 → 직선 구간 왔다갔다 하는 짧은 루프 제외
        if length > best_length:
            best_length = length
            best_path = path
    return best_path


def resample_polyline_by_arc_length(points: list, step: float, closed: bool = False) -> list:
    """polyline을 호길이 기준 step 간격으로 리샘플링. step 단위는 픽셀. closed=True면 끝→시작 구간 포함."""
    if len(points) <= 1 or step <= 0:
        return points
    pts = np.array(points, dtype=float)
    n = len(pts)
    seg_lens = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    if closed and n >= 3:
        close_len = np.sqrt(np.sum((pts[0] - pts[-1]) ** 2))
        seg_lens = np.append(seg_lens, close_len)
    cum = np.zeros(len(seg_lens) + 1)
    cum[1:] = np.cumsum(seg_lens)
    total = cum[-1]
    if total < 1e-9:
        return points
    out = []
    t = 0.0
    n_seg = len(seg_lens)
    while t < total - 1e-9:
        i = np.searchsorted(cum, t, side="right") - 1
        i = max(0, min(i, n_seg - 1))
        if cum[i + 1] - cum[i] < 1e-9:
            frac = 0.0
        else:
            frac = (t - cum[i]) / (cum[i + 1] - cum[i])
        if closed and i == n_seg - 1:
            # 마지막 구간: pts[-1] -> pts[0]
            pt = (1 - frac) * pts[-1] + frac * pts[0]
        else:
            pt = (1 - frac) * pts[i] + frac * pts[i + 1]
        out.append((float(pt[0]), float(pt[1])))
        t += step
    if not out:
        out = [tuple(pts[0])]
    return out


def simplify_backtrack(path: list, dot_thresh: float = -0.3) -> list:
    """경로에서 '같은 구간 왔다갔다' 구간 완화. 연속 두 변위의 내적이 dot_thresh 미만이면 역방향으로 간주해 중간 점 제거. 여러 번 반복해 수렴."""
    if len(path) < 4:
        return path
    for _ in range(5):  # 여러 패스로 역구간 제거
        out = [path[0], path[1]]
        for i in range(2, len(path)):
            dr0 = out[-1][0] - out[-2][0]
            dc0 = out[-1][1] - out[-2][1]
            dr1 = path[i][0] - out[-1][0]
            dc1 = path[i][1] - out[-1][1]
            n0 = np.sqrt(dr0 * dr0 + dc0 * dc0) + 1e-9
            n1 = np.sqrt(dr1 * dr1 + dc1 * dc1) + 1e-9
            dot = (dr0 * dr1 + dc0 * dc1) / (n0 * n1)
            if dot < dot_thresh and len(out) >= 2:
                out.pop()
                if out and out[-1] != path[i]:
                    out.append(path[i])
            else:
                if out[-1] != path[i]:
                    out.append(path[i])
        if len(out) >= len(path):
            break
        path = out
    return path


def smooth_polyline(points: list, window: int = 5) -> list:
    """이동평균으로 경로 부드럽게 (곡률 제한에 도움). window는 양쪽 반창."""
    if len(points) < window * 2 + 1 or window < 1:
        return points
    pts = np.array(points, dtype=float)
    n = len(pts)
    out = np.zeros_like(pts)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        out[i] = np.mean(pts[lo:hi], axis=0)
    return [tuple(out[i]) for i in range(n)]


def get_ordered_skeleton_path(skel: np.ndarray):
    """Skeleton image (0/1) -> ordered list of (row, col). One loop: follow 8-neighbors (기존 동작)."""
    pt_set, neighbors = skeleton_to_graph(skel)
    if not pt_set:
        return []
    pruned = prune_skeleton_tips(pt_set, neighbors)
    if not pruned:
        return list(pt_set)
    return extract_largest_cycle(pruned, neighbors)


def main():
    parser = argparse.ArgumentParser(description="Extract centerline from map image to CSV")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_map = os.path.join(script_dir, "..", "..", "f1tenth_gym_ros", "maps", "Spielberg_map.yaml")
    parser.add_argument(
        "--map",
        default=default_map,
        help="Path to map YAML (image path in YAML is relative to this file)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(script_dir, "..", "config", "Spielberg_centerline_auto.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Keep every Nth point (1 = all)")
    parser.add_argument(
        "--method",
        choices=("mat", "centerline", "radial", "midpoint", "medial"),
        default="mat",
        help="mat=거리변환+ridge(센터라인)+가지치기+최대루프+호길이리샘플(기본). radial/centerline/midpoint/medial=기타.",
    )
    parser.add_argument("--arc-step", type=float, default=1.5, help="MAT: 호길이 리샘플 간격(픽셀).")
    parser.add_argument("--smooth", type=int, default=0, help="MAT: 이동평균 창 크기(0=미적용).")
    parser.add_argument(
        "--invert-free",
        action="store_true",
        help="맵에서 밝은 쪽=도로로 해석 (벽과 벽 사이가 흰색일 때 시도).",
    )
    parser.add_argument(
        "--min-dist-from-wall",
        type=float,
        default=1.0,
        help="벽에서 이만큼(픽셀) 이상 떨어진 점만 유지. 0=비활성.",
    )
    parser.add_argument(
        "--max-dist-from-wall",
        type=float,
        default=0.0,
        metavar="PX",
        help="벽에서 이만큼(픽셀) 이내만 통로로 봄. 검은 두 선 사이=도로/바깥=흰 맵에서 트랙만 쓰려면 20~50 등 지정. 0=비활성.",
    )
    parser.add_argument(
        "--start-at-zero",
        action="store_true",
        help="첫 점을 (0,0)으로 두고 나머지를 상대 좌표로 (정상 Spielberg_centerline.csv 형식).",
    )
    parser.add_argument(
        "--origin-x",
        type=float,
        default=None,
        help="출력 좌표계 원점(맵 프레임 x). 지정 시 모든 좌표를 (x - origin_x, y - origin_y)로 재조정.",
    )
    parser.add_argument(
        "--origin-y",
        type=float,
        default=None,
        help="출력 좌표계 원점(맵 프레임 y). 시뮬/차량이 쓰는 원점 또는 스타트라인 위치.",
    )
    parser.add_argument(
        "--origin-from-map",
        action="store_true",
        help="맵 YAML의 origin을 출력 원점으로 사용 (모든 좌표를 맵 origin 기준 상대값으로).",
    )
    parser.add_argument(
        "--no-thin",
        action="store_true",
        help="Ridge를 1픽셀 두께로 얇게 하지 않음 (같은 구간 반복 시 --no-thin 시도).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.map):
        print(f"Map not found: {args.map}", file=sys.stderr)
        return 1

    print(f"Loading map: {args.map}")
    free_mask, resolution, origin_x, origin_y, (height, width) = load_map(
        args.map, invert_free=args.invert_free
    )
    if args.invert_free:
        print("  (invert_free: 밝은 쪽=도로)")
    print(f"  Free pixels: {free_mask.sum()}, shape {free_mask.shape}")

    if args.method == "mat":
        # 1) 맵 이진화: free=도로, occupied=벽  2) 거리변환 D(x,y)  3) D 국소최대 = ridge = 센터라인
        # 4) 스켈레톤 그래프 가지치기  5) 최대 사이클(한 바퀴)  6) 호길이 리샘플 (ridge가 이미 통로 한가운데라 refocus 없음)
        # 검은 두 선 사이=도로/바깥=흰 맵: --max-dist-from-wall 로 트랙 통로만 남김
        max_dist = float(args.max_dist_from_wall) if args.max_dist_from_wall > 0 else None
        if max_dist is not None:
            print(f"  max_dist_from_wall={max_dist} px (트랙 통로만 사용)")
        skel = medial_axis_from_free_mask(free_mask, max_dist=max_dist)
        skel_no_thin = skel.copy()  # thinning 실패 시 재시도용
        # Ridge가 두꺼우면 직선 구간에 평행한 두 줄이 생겨, 한 바퀴가 같은 구간을 반복함 → 1픽셀 thinning
        use_thin = not getattr(args, "no_thin", False)
        if use_thin and (_medial_axis_thin is not None or _skeletonize_thin is not None):
            skel = thin_ridge_to_skeleton(skel)
            print("  Ridge → 1픽셀 thinning (medial_axis/skeletonize)")
        elif use_thin:
            print("  (skimage 없음: ridge 두껍게 유지 → 경로가 같은 구간 반복될 수 있음. pip install scikit-image 권장)")
        pt_set, ngb = skeleton_to_graph(skel)
        pruned = prune_skeleton_tips(pt_set, ngb)
        used_no_thin_fallback = False
        if not pruned and use_thin and (_skeletonize_thin is not None or _medial_axis_thin is not None):
            print("  Thinning 후 스켈레톤 비어 있음 → ridge 행/열 중점으로 한 줄 경로 생성 (역방향 없음)")
            # 두꺼운 ridge에서 그래프 사이클 쓰면 직선 구간 왔다갔다 함 → 대신 행·열별 ridge 중점만 뽑아 한 바퀴 경로로 만듦
            ridge_as_mask = (skel_no_thin > 0).astype(np.uint8)
            row_cols = row_col_midpoints(ridge_as_mask, use_both=True)
            merged = merge_near_points(row_cols, tol=2.0)
            ordered = order_points_nearest(merged, start_index=None)
            if len(ordered) < 3:
                print("  No path from ridge midpoints.", file=sys.stderr)
                return 1
            # 경로를 닫힌 루프로: 끝↔시작 거리가 전체 둘레에 비해 작으면 이미 루프, 아니면 첫 점을 끝에 추가
            p0, pN = ordered[0], ordered[-1]
            gap = np.sqrt((pN[0] - p0[0]) ** 2 + (pN[1] - p0[1]) ** 2)
            total = _path_arc_length(ordered, closed=False)
            if total > 1e-9 and gap > total * 0.1:
                ordered = ordered + [ordered[0]]
            ordered = resample_polyline_by_arc_length(ordered, args.arc_step, closed=True)
            used_no_thin_fallback = True
        else:
            if not pruned:
                print("  No skeleton after prune.", file=sys.stderr)
                return 1
            raw_path = extract_largest_cycle(pruned, ngb)
            if len(raw_path) < 3:
                print("  No cycle found.", file=sys.stderr)
                return 1
            ordered = resample_polyline_by_arc_length(raw_path, args.arc_step, closed=True)
        if args.smooth > 0:
            ordered = smooth_polyline(ordered, window=args.smooth)
        if used_no_thin_fallback:
            print(f"  MAT: ridge→행/열중점→order→arc_resample(step={args.arc_step}) → {len(ordered)} pts")
        else:
            print(f"  MAT: ridge→prune→cycle→arc_resample(step={args.arc_step}) → {len(ordered)} pts")
        # Some maps collapse to a tiny loop in MAT mode (e.g. very thick/free regions).
        # Auto-fallback to midpoint to avoid unusable CSV outputs.
        if len(ordered) < 100:
            print(
                f"  MAT produced too few points ({len(ordered)}). "
                "Fallback to midpoint method."
            )
            row_cols = row_col_midpoints(free_mask, use_both=True)
            merged = merge_near_points(row_cols, tol=1.5)
            ordered = order_points_nearest(merged, start_index=None)
            print(f"  Fallback midpoint: {len(ordered)} points")
    elif args.method == "centerline":
        # 스켈레톤으로 도로를 따라 가는 순서 확보 → 각 점에서 수직으로 벽 찾아 그 사이 정중앙으로 보정
        skel = medial_axis_from_free_mask(free_mask)
        raw_path = get_ordered_skeleton_path(skel)
        if not raw_path:
            print("  No skeleton path.", file=sys.stderr)
            return 1
        ordered = refocus_centerline(free_mask, raw_path)
        print(f"  Centerline method: {len(ordered)} points (wall–wall midpoint along track)")
    elif args.method == "radial":
        # 각도별: 중심에서 레이 → free 구간의 첫벽·끝벽 사이 정중앙 (표준 방식)
        ordered = centerline_radial(free_mask, num_angles=720)
        print(f"  Radial method: {len(ordered)} points (angle-wise wall–wall center)")
    elif args.method == "midpoint":
        # 행+열 각각 free 구간 정중앙 → 합친 뒤 가까운 점 병합 → 순서 연결
        row_cols = row_col_midpoints(free_mask, use_both=True)
        merged = merge_near_points(row_cols, tol=1.5)
        ordered = order_points_nearest(merged, start_index=None)
        print(f"  Midpoint method: {len(ordered)} points (row+col center, merged)")
    else:
        skel = medial_axis_from_free_mask(free_mask)
        n_skel = int(skel.sum())
        print(f"  Medial axis pixels: {n_skel}")
        ordered = get_ordered_skeleton_path(skel)
        if not ordered:
            print("  No ordered path from skeleton.", file=sys.stderr)
            return 1
        print(f"  Ordered path length: {len(ordered)}")

    # MAT가 아닐 때만 벽 거리 필터 (MAT ridge는 이미 통로 한가운데)
    if args.method != "mat" and args.min_dist_from_wall > 0:
        filtered = filter_centerline_away_from_walls(
            ordered, free_mask, min_dist_from_wall_px=args.min_dist_from_wall
        )
        if len(filtered) >= 2:
            ordered = filtered
            if len(ordered) < 100:
                ordered = resample_polyline_by_arc_length(ordered, step=2.0)
            print(f"  After wall filter (min_dist={args.min_dist_from_wall}px): {len(ordered)} pts")

    # 연속 겹침 제거
    ordered = remove_duplicate_points(ordered, min_dist_px=0.5)
    if len(ordered) < 2:
        print("  Too few points after dedup.", file=sys.stderr)
        return 1

    # To world coordinates (ROS 수식: x=origin_x+col*res, y=origin_y+(height-1-row)*res)
    path_xy = []
    for i, pt in enumerate(ordered):
        if i % args.downsample != 0:
            continue
        r, c = pt[0], pt[1]
        x, y = pixel_to_world(r, c, height, resolution, origin_x, origin_y)
        path_xy.append((x, y))

    # 원점/차량 기준 재조정: 라인 따기 후, 시뮬·맵에서 쓰는 원점에 맞게 좌표 재조정
    ox = args.origin_x
    oy = args.origin_y
    if getattr(args, "origin_from_map", False):
        ox = origin_x
        oy = origin_y
        path_xy = [(x - ox, y - oy) for x, y in path_xy]
        print(f"  Output origin from map: ({ox:.2f}, {oy:.2f}) → 좌표 재조정됨")
    elif ox is not None and oy is not None:
        path_xy = [(x - ox, y - oy) for x, y in path_xy]
        print(f"  Output origin applied: ({ox}, {oy}) → 좌표 재조정됨")

    # 정상 CSV처럼 첫 점을 (0,0)에 두고 나머지를 상대 좌표로 (선택)
    if args.start_at_zero and len(path_xy) >= 2:
        def dist0(p):
            return p[0] ** 2 + p[1] ** 2
        i0 = min(range(len(path_xy)), key=lambda i: dist0(path_xy[i]))
        x0, y0 = path_xy[i0]
        path_xy = [(x - x0, y - y0) for x, y in (path_xy[i0:] + path_xy[:i0])]
        print(f"  start_at_zero: 첫 점 (0,0), {len(path_xy)} pts")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 기존 파일이 있으면 삭제 후 새로 작성 (추가가 아닌 완전 교체)
    if os.path.isfile(args.out):
        os.remove(args.out)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in path_xy:
            w.writerow([x, y])
    print(f"Wrote {len(path_xy)} points to {args.out} (overwrite)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
