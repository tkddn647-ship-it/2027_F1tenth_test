"""
connectome_loader.py
=====================
초파리 hemibrain 커넥톰 로더.

레이싱 목표:
  - 전체 뇌 대신 속도·헤딩 관련 서브서킷(T4/T5 운동감지 + CX + DN)을 우선 사용
  - 토폴로지는 고정, 세기는 PPO가 학습
"""

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 일반 시각 (폴백 / 다운로드 시드)
VISUAL_INPUT_TYPE_PREFIXES = (
    "R1-6", "R7", "R8",
    "L1", "L2", "L3", "L4", "L5",
    "LC", "LPLC", "LLPC", "LT", "LPTC",
    "T4", "T5", "HS", "VS", "H2", "CH",
)

# 레이싱 입력: optic-flow / 운동 감지 계열
MOTION_INPUT_PREFIXES = (
    "T4", "T5", "LPTC", "HS", "VS", "H2", "CH",
    "LC", "LPLC", "LLPC", "LT",  # hemibrain에 T4/T5 없을 때 투영뉴런 폴백
)

# 중심복합체(헤딩·방향 제어)
CX_TYPE_PREFIXES = (
    "PB", "EB", "FB", "NO",
    "PEN", "PEG", "EPG", "DELTA7", "PFN", "PFL", "EL", "ER",
)

DESCENDING_TYPE_PREFIX = "DN"

DEFAULT_CACHE_DIR = Path("hemibrain_cache")
DEFAULT_DATASET = "hemibrain:v1.2.1"


def resolve_token(cli_token: str | None = None) -> str:
    token = (
        cli_token
        or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN")
    )
    if not token:
        raise SystemExit(
            "neuprint 토큰이 없습니다.\n"
            "  PowerShell: $env:NEUPRINT_APPLICATION_CREDENTIALS = \"토큰\"\n"
            "  또는: python download_hemibrain.py --token \"토큰\""
        )
    return token.strip()


def cache_paths(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> dict[str, Path]:
    d = Path(cache_dir)
    return {
        "dir": d,
        "neurons": d / "neurons.parquet",
        "connections": d / "connections.parquet",
        "meta": d / "meta.txt",
    }


def save_hemibrain_cache(neurons_df: pd.DataFrame, conn_df: pd.DataFrame,
                         cache_dir: Path | str = DEFAULT_CACHE_DIR,
                         meta: str = "") -> None:
    paths = cache_paths(cache_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    neurons_df.to_parquet(paths["neurons"], index=False)
    conn_df.to_parquet(paths["connections"], index=False)
    paths["meta"].write_text(meta or f"neurons={len(neurons_df)}, edges={len(conn_df)}\n",
                             encoding="utf-8")
    print(f"[connectome_loader] 캐시 저장: {paths['dir'].resolve()}")


def load_hemibrain_cache(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = cache_paths(cache_dir)
    if not paths["neurons"].exists() or not paths["connections"].exists():
        raise SystemExit(
            f"캐시가 없습니다: {paths['dir']}\n"
            "먼저 실행: python download_hemibrain.py"
        )
    neurons_df = pd.read_parquet(paths["neurons"])
    conn_df = pd.read_parquet(paths["connections"])
    print(f"[connectome_loader] 캐시 로드: 뉴런 {len(neurons_df)}개, 연결 {len(conn_df)}개")
    return neurons_df, conn_df


def _neuprint_imports():
    try:
        from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
    except ImportError as e:
        raise SystemExit(
            "neuprint-python이 설치되어 있지 않습니다.\n"
            "  pip install neuprint-python pyarrow\n"
            f"(원본 에러: {e})"
        )
    return Client, fetch_adjacencies, fetch_neurons, NC


def _type_mask(types: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    return types.str.startswith(prefixes)


def select_speed_relevant_subcircuit(
    neurons_df: pd.DataFrame,
    conn_df: pd.DataFrame,
    max_neurons: int = 800,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """운동감지(T4/T5/…) + 중심복합체(CX) + descending(DN) 중심 서브그래프.

    실제 캐시에 T4/T5/CX가 거의 없으면 LC·LPTC + DN + 그 사이 중계 뉴런으로 폴백.
    """
    types = neurons_df["type"].fillna("").astype(str)
    is_motion = _type_mask(types, MOTION_INPUT_PREFIXES)
    is_cx = _type_mask(types, CX_TYPE_PREFIXES)
    is_dn = types.str.startswith(DESCENDING_TYPE_PREFIX)

    n_motion, n_cx, n_dn = int(is_motion.sum()), int(is_cx.sum()), int(is_dn.sum())
    print(f"[subcircuit] 후보 motion={n_motion}, CX={n_cx}, DN={n_dn}")

    seed_mask = is_motion | is_cx | is_dn
    if seed_mask.sum() < 30:
        is_vis = _type_mask(types, VISUAL_INPUT_TYPE_PREFIXES)
        seed_mask = is_vis | is_dn
        print(f"[subcircuit] 폴백: visual+DN → {int(seed_mask.sum())}개 시드")

    seed_ids = set(neurons_df.loc[seed_mask, "bodyId"].astype(int).tolist())
    if not seed_ids:
        raise SystemExit("속도 관련 서브서킷 시드 뉴런이 없습니다.")

    # 시드 ↔ 이웃 1-hop 포함
    pre = conn_df["bodyId_pre"].astype(int)
    post = conn_df["bodyId_post"].astype(int)
    touch = pre.isin(seed_ids) | post.isin(seed_ids)
    hop_ids = set(pre[touch].tolist()) | set(post[touch].tolist()) | seed_ids

    keep = hop_ids
    if len(keep) > max_neurons:
        # 시드는 유지, 나머지는 degree 높은 순
        deg = pd.concat([pre, post]).value_counts()
        ordered = [int(b) for b in seed_ids]
        for bid in deg.index:
            bid = int(bid)
            if bid in keep and bid not in ordered:
                ordered.append(bid)
            if len(ordered) >= max_neurons:
                break
        keep = set(ordered[:max_neurons])

    neurons_out = neurons_df[neurons_df["bodyId"].astype(int).isin(keep)].copy()
    neurons_out = neurons_out.reset_index(drop=True)
    keep_ids = set(neurons_out["bodyId"].astype(int).tolist())
    conn_out = conn_df[
        conn_df["bodyId_pre"].astype(int).isin(keep_ids)
        & conn_df["bodyId_post"].astype(int).isin(keep_ids)
    ].reset_index(drop=True)

    print(f"[subcircuit] 최종 뉴런 {len(neurons_out)}개, 연결 {len(conn_out)}개")
    return neurons_out, conn_out


def load_hemibrain_real(token: str, min_weight: int = 5,
                        max_neurons: int = 3000,
                        max_visual_seeds: int = 400,
                        dataset: str = DEFAULT_DATASET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """운동/CX/DN 시드 위주로 hemibrain 서브그래프를 받는다."""
    Client, fetch_adjacencies, fetch_neurons, NC = _neuprint_imports()

    print(f"[connectome_loader] neuprint 접속: {dataset}")
    client = Client("neuprint.janelia.org", dataset=dataset, token=token)

    print("[connectome_loader] DN / 운동·CX 시드 조회...")
    dn_df, _ = fetch_neurons(NC(type="DN.*", regex=True, status="Traced"), client=client)

    seed_patterns = [
        "T4.*", "T5.*", "LPTC.*", "HS.*", "VS.*",
        "LC.*", "LPLC.*", "LLPC.*", "LT[0-9].*",
        "PB.*", "EB.*", "FB.*", "EPG.*", "PEN.*", "PEG.*", "PFN.*", "PFL.*",
        "L[1-5].*",
    ]
    parts = []
    for pat in seed_patterns:
        try:
            df, _ = fetch_neurons(NC(type=pat, regex=True, status="Traced"), client=client)
            if df is not None and len(df):
                parts.append(df)
        except Exception as e:
            print(f"  경고: 패턴 {pat} 실패 — {e}")

    if not parts:
        raise SystemExit("시각/운동/CX 시드 뉴런을 하나도 못 찾았습니다.")
    if dn_df is None or len(dn_df) == 0:
        raise SystemExit("DN 뉴런을 하나도 못 찾았습니다.")

    vis_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["bodyId"])
    if len(vis_df) > max_visual_seeds:
        # DN·CX는 가능한 한 남기고 나머지만 샘플
        t = vis_df["type"].fillna("").astype(str)
        must = vis_df[_type_mask(t, CX_TYPE_PREFIXES) | t.str.startswith(MOTION_INPUT_PREFIXES[:6])]
        rest = vis_df.drop(index=must.index, errors="ignore")
        n_rest = max(0, max_visual_seeds - len(must))
        if n_rest > 0 and len(rest) > n_rest:
            rest = rest.sample(n=n_rest, random_state=0)
        vis_df = pd.concat([must, rest], ignore_index=True).drop_duplicates("bodyId")
        print(f"[connectome_loader] 시드 샘플링 → {len(vis_df)}개")

    seed_ids = pd.unique(pd.concat([vis_df["bodyId"], dn_df["bodyId"]], ignore_index=True))
    print(f"[connectome_loader] 시드 합집합 {len(seed_ids)}개")

    print("[connectome_loader] adjacency 다운로드...")
    neurons_df, conn_df = fetch_adjacencies(
        NC(bodyId=list(map(int, seed_ids))), client=client,
    )
    if neurons_df is None or conn_df is None or len(conn_df) == 0:
        raise SystemExit("adjacency 결과가 비었습니다.")

    conn_df = conn_df[conn_df["weight"] >= min_weight].reset_index(drop=True)
    used_ids = pd.unique(
        pd.concat([conn_df["bodyId_pre"], conn_df["bodyId_post"],
                   pd.Series(seed_ids)], ignore_index=True)
    )
    neurons_df = neurons_df[neurons_df["bodyId"].isin(used_ids)].drop_duplicates("bodyId")
    neurons_df = neurons_df.reset_index(drop=True)

    if len(neurons_df) > max_neurons:
        print(f"[connectome_loader] {len(neurons_df)} → max_neurons={max_neurons}")
        seed_set = set(map(int, seed_ids))
        deg = pd.concat([conn_df["bodyId_pre"], conn_df["bodyId_post"]]).value_counts()
        keep = set(seed_set)
        for bid, _ in deg.items():
            if len(keep) >= max_neurons:
                break
            keep.add(int(bid))
        neurons_df = neurons_df[neurons_df["bodyId"].isin(keep)].reset_index(drop=True)
        keep_ids = set(neurons_df["bodyId"].tolist())
        conn_df = conn_df[
            conn_df["bodyId_pre"].isin(keep_ids) & conn_df["bodyId_post"].isin(keep_ids)
        ].reset_index(drop=True)

    keep_cols = [c for c in ["bodyId", "type", "instance", "status", "predictedNt", "celltype"]
                 if c in neurons_df.columns]
    neurons_df = neurons_df[keep_cols].copy()
    conn_df = conn_df[["bodyId_pre", "bodyId_post", "weight"]].copy()
    print(f"[connectome_loader] 최종: 뉴런 {len(neurons_df)}개, 연결 {len(conn_df)}개")
    return neurons_df, conn_df


def build_synthetic_hemibrain(n_neurons: int = 800, n_synapses: int = 20_000,
                               seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """레이싱 서브서킷 형태 합성: T4/T5 → CX → DN (+ 일부 재귀)."""
    rng = np.random.default_rng(seed)
    n_neurons = max(120, n_neurons)
    n_motion = max(40, n_neurons // 5)
    n_cx = max(40, n_neurons // 4)
    n_dn = max(20, n_neurons // 10)
    n_inter = n_neurons - n_motion - n_cx - n_dn
    if n_inter < 0:
        n_inter = 0
        n_motion = n_neurons // 3
        n_cx = n_neurons // 3
        n_dn = n_neurons - n_motion - n_cx

    motion_types = (["T4"] * (n_motion // 3) + ["T5"] * (n_motion // 3)
                    + ["LPTC"] * (n_motion - 2 * (n_motion // 3)))
    cx_types = (["PB"] * (n_cx // 4) + ["EB"] * (n_cx // 4) + ["EPG"] * (n_cx // 4)
                + ["PFN"] * (n_cx - 3 * (n_cx // 4)))
    inter_types = [f"INT_{i % 40}" for i in range(n_inter)]
    dn_types = [f"DNp{i % 30:02d}" for i in range(n_dn)]
    types = motion_types + cx_types + inter_types + dn_types

    neurons_df = pd.DataFrame({
        "bodyId": np.arange(n_neurons),
        "type": types,
        "instance": [f"{t}_{i}" for i, t in enumerate(types)],
    })

    motion_ids = np.arange(0, n_motion)
    cx_ids = np.arange(n_motion, n_motion + n_cx)
    inter_ids = np.arange(n_motion + n_cx, n_motion + n_cx + n_inter)
    dn_ids = np.arange(n_motion + n_cx + n_inter, n_neurons)

    pre, post, weight = [], [], []

    def add_edges(src, dst, n_edges, w_scale=6.0):
        if len(src) == 0 or len(dst) == 0 or n_edges <= 0:
            return
        s = rng.choice(src, size=n_edges)
        d = rng.choice(dst, size=n_edges)
        w = rng.gamma(2.0, w_scale, size=n_edges).astype(int) + 1
        pre.extend(s.tolist()); post.extend(d.tolist()); weight.extend(w.tolist())

    add_edges(motion_ids, cx_ids, int(n_synapses * 0.30))
    add_edges(cx_ids, cx_ids, int(n_synapses * 0.25))
    add_edges(cx_ids, dn_ids, int(n_synapses * 0.20))
    if len(inter_ids):
        add_edges(motion_ids, inter_ids, int(n_synapses * 0.10))
        add_edges(inter_ids, dn_ids, int(n_synapses * 0.10))
        add_edges(inter_ids, inter_ids, int(n_synapses * 0.05))

    conn_df = pd.DataFrame({"bodyId_pre": pre, "bodyId_post": post, "weight": weight})
    conn_df = conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    return neurons_df, conn_df


def identify_io_neurons(neurons_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """입력=운동/시각, 출력=DN."""
    types = neurons_df["type"].fillna("").astype(str)
    is_in = _type_mask(types, MOTION_INPUT_PREFIXES)
    if int(is_in.sum()) == 0:
        is_in = _type_mask(types, VISUAL_INPUT_TYPE_PREFIXES)
    is_out = types.str.startswith(DESCENDING_TYPE_PREFIX)
    return neurons_df.index[is_in].to_numpy(), neurons_df.index[is_out].to_numpy()


def build_adjacency(neurons_df: pd.DataFrame, conn_df: pd.DataFrame,
                     signed: bool = True) -> np.ndarray:
    n = len(neurons_df)
    idx_map = {int(bid): i for i, bid in enumerate(neurons_df["bodyId"])}
    A = np.zeros((n, n), dtype=np.float32)

    if signed and "predictedNt" in neurons_df.columns:
        inhib = {"GABA", "Glu", "glutamate", "gaba"}
        nt = neurons_df["predictedNt"].fillna("").astype(str)
        sign = np.where(nt.str.upper().isin({x.upper() for x in inhib}) | nt.isin(inhib),
                        -1.0, 1.0)
    else:
        rng = np.random.default_rng(1)
        sign = np.where(rng.random(n) < 0.3, -1.0, 1.0) if signed else np.ones(n)

    for pre, post, w in zip(conn_df["bodyId_pre"], conn_df["bodyId_post"], conn_df["weight"]):
        i, j = idx_map.get(int(pre)), idx_map.get(int(post))
        if i is None or j is None:
            continue
        A[i, j] = float(w) * float(sign[i])
    return A


if __name__ == "__main__":
    neurons_df, conn_df = build_synthetic_hemibrain(800)
    neurons_df, conn_df = select_speed_relevant_subcircuit(neurons_df, conn_df)
    inp, out = identify_io_neurons(neurons_df)
    A = build_adjacency(neurons_df, conn_df)
    print(f"합성 서브서킷: N={len(neurons_df)}, in={len(inp)}, out={len(out)}, dens={(A!=0).mean():.4%}")
