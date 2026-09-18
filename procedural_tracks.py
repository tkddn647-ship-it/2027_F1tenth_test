"""
procedural_tracks.py
======================
f1tenth_racetracks(실제 F1 트랙 20여개)을 설치하기 전에, "여러 트랙 모양에
한 정책이 다 통하는지"를 오프라인으로 먼저 검증하기 위한 절차적 트랙 생성기.

실제 f1tenth_racetracks의 각 트랙도 결국 centerline.csv(x, y, 좌우 폭)
형태로 제공되므로, 이 파일이 만드는 합성 트랙과 인터페이스가 동일하다 —
나중에 f1tenth_gym_wrapper.py로 갈아끼워도 multitrack_env.py 쪽 코드는
그대로 재사용된다.
"""

from __future__ import annotations
import numpy as np


def oval(a: float = 6.0, b: float = 4.5, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


def wavy_loop(base_r: float = 6.0, wave_amp: float = 1.2, wave_freq: int = 3,
              n: int = 400) -> np.ndarray:
    """원형에 파형을 섞어 곡률이 계속 바뀌는 트랙 — 급커브 대응력 테스트용."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = base_r + wave_amp * np.sin(wave_freq * t)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def figure_eight(scale: float = 5.0, n: int = 500) -> np.ndarray:
    """8자 트랙(자기교차 있음) — 급격한 방향전환 대응력 테스트용.
    자기교차 지점은 진행률(progress) 계산에서 별도 처리가 필요할 수 있어,
    학습 초반엔 oval/wavy_loop로 먼저 검증하고 나중에 추가하는 걸 권장."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = scale * np.sin(2 * t)
    y = scale * np.sin(t) * 1.6
    return np.stack([x, y], axis=1)


def asymmetric_oval(a: float = 7.0, b: float = 3.5, skew: float = 0.3, n: int = 400) -> np.ndarray:
    """한쪽으로 치우친 비대칭 트랙 — 좌우 균일하지 않은 코너 대응력 테스트용."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t) + skew * a * np.sin(t) * np.cos(t)
    return np.stack([x, y], axis=1)


TRACK_GENERATORS = {
    "oval": oval,
    "wavy_loop": wavy_loop,
    "figure_eight": figure_eight,
    "asymmetric_oval": asymmetric_oval,
}


def build_track_pool(names: list[str] | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """여러 트랙의 centerline을 미리 만들어 dict로 반환.

    기본 풀은 figure_eight를 제외한다 (자기교차로 progress 계산이 불안정).
    """
    rng = np.random.default_rng(seed)
    default_names = ["oval", "wavy_loop", "asymmetric_oval"]
    names = names or default_names
    pool = {}
    for name in names:
        if name not in TRACK_GENERATORS:
            raise KeyError(f"알 수 없는 트랙: {name}. 가능: {list(TRACK_GENERATORS)}")
        fn = TRACK_GENERATORS[name]
        jitter = rng.uniform(0.9, 1.1)
        if name == "oval":
            pool[name] = fn(a=6.0 * jitter, b=4.5 * jitter)
        elif name == "wavy_loop":
            pool[name] = fn(base_r=6.0 * jitter)
        elif name == "figure_eight":
            pool[name] = fn(scale=5.0 * jitter)
        elif name == "asymmetric_oval":
            pool[name] = fn(a=7.0 * jitter, b=3.5 * jitter)
    return pool


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pool = build_track_pool()
    fig, axes = plt.subplots(1, len(pool), figsize=(4 * len(pool), 4))
    for ax, (name, cl) in zip(axes, pool.items()):
        ax.plot(cl[:, 0], cl[:, 1], "-")
        ax.set_aspect("equal")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("track_pool_preview.png", dpi=100)
    print(f"[procedural_tracks] {len(pool)}개 트랙 생성 완료, track_pool_preview.png 저장")
