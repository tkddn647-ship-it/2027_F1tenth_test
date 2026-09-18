"""
download_hemibrain.py
=====================
neuprint에서 hemibrain 서브그래프를 받아 hemibrain_cache/ 에 저장한다.
토큰은 CLI 또는 환경변수로만 받고, 파일에는 저장하지 않는다.

  $env:NEUPRINT_APPLICATION_CREDENTIALS = "토큰"
  python download_hemibrain.py

  python download_hemibrain.py --token "토큰"
"""

from __future__ import annotations
import argparse

from connectome_loader import (
    DEFAULT_CACHE_DIR, DEFAULT_DATASET,
    resolve_token, load_hemibrain_real, save_hemibrain_cache, identify_io_neurons,
)


def main():
    parser = argparse.ArgumentParser(description="Download hemibrain subgraph to local cache")
    parser.add_argument("--token", type=str, default=None, help="neuprint auth token")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--min-weight", type=int, default=5)
    parser.add_argument("--max-neurons", type=int, default=3000)
    parser.add_argument("--max-visual-seeds", type=int, default=400)
    args = parser.parse_args()

    token = resolve_token(args.token)
    neurons_df, conn_df = load_hemibrain_real(
        token,
        min_weight=args.min_weight,
        max_neurons=args.max_neurons,
        max_visual_seeds=args.max_visual_seeds,
        dataset=args.dataset,
    )
    inp, out = identify_io_neurons(neurons_df)
    meta = (
        f"dataset={args.dataset}\n"
        f"neurons={len(neurons_df)}\n"
        f"edges={len(conn_df)}\n"
        f"visual_inputs={len(inp)}\n"
        f"descending={len(out)}\n"
        f"min_weight={args.min_weight}\n"
        f"max_neurons={args.max_neurons}\n"
    )
    save_hemibrain_cache(neurons_df, conn_df, cache_dir=args.cache_dir, meta=meta)
    print(f"[download_hemibrain] 시각/CX 입력 {len(inp)}개 / DN {len(out)}개")
    print("[download_hemibrain] 완료. 학습:")
    print("  python train_connectome.py --use-cache --timesteps 200000")
    print("  python train_connectome.py --baseline --timesteps 200000")


if __name__ == "__main__":
    main()
