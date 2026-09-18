"""
toy_obstacle_env.py
=====================
FLYNN 논문이 MuJoCo로 만든 "원형 장애물이 흩어진 아레나에서 충돌 없이
돌아다니기" 환경을, 훨씬 가벼운 순수 numpy 2D 버전으로 재현했다.

목적: 실제 RoboRacer(F1TENTH Gym)로 가기 전에, "커넥톰 기반 이상한 RNN이
강화학습으로 뭐라도 배우긴 하는가"를 빠르고 가볍게 먼저 확인하기 위함
(GPU 없이 노트북에서 수천 스텝을 몇 초 안에 돌려볼 수 있어야 반복 실험이 가능함).

관측: LiDAR처럼 전방 n_rays 방향의 장애물까지 거리 (정규화 0~1)
행동: [조향각속도, 가속] 연속 2차원 — RoboRacer와 동일한 인터페이스
보상: 전진 거리 - 충돌 페널티 - (벽에 붙어 도는 것 방지용 낮은 속도 페널티)
"""

from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gym = None
    spaces = None

_EnvBase = gym.Env if _HAS_GYM else object


class ToyObstacleEnv(_EnvBase):
    """원형 아레나 + 원형 장애물들. FLYNN 논문의 평가환경을 단순화한 버전."""

    ARENA_RADIUS = 7.0
    N_OBSTACLES = 20
    OBSTACLE_RADIUS = 0.4
    N_RAYS = 16
    RAY_RANGE = 5.0
    MAX_STEPS = 300
    DT = 0.1

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        if _HAS_GYM:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.N_RAYS,), dtype=np.float32)
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]),
                                            dtype=np.float32)
        self.pos = np.zeros(2)
        self.heading = 0.0
        self.speed = 0.0
        self.obstacles = np.zeros((self.N_OBSTACLES, 2))
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2)
        self.heading = float(self.rng.uniform(-np.pi, np.pi))
        self.speed = 0.0
        self.steps = 0

        # 장애물을 원점 근처 제외하고 아레나 안에 무작위 배치
        obs = []
        while len(obs) < self.N_OBSTACLES:
            p = self.rng.uniform(-self.ARENA_RADIUS, self.ARENA_RADIUS, size=2)
            if np.linalg.norm(p) < 1.2 or np.linalg.norm(p) > self.ARENA_RADIUS - 0.3:
                continue
            obs.append(p)
        self.obstacles = np.array(obs)

        obs_vec = self._get_obs()
        info = {}
        if _HAS_GYM:
            return obs_vec, info
        return obs_vec

    def _cast_rays(self) -> np.ndarray:
        angles = self.heading + np.linspace(-np.pi / 2, np.pi / 2, self.N_RAYS)
        dists = np.full(self.N_RAYS, self.RAY_RANGE, dtype=np.float32)

        for i, a in enumerate(angles):
            direction = np.array([np.cos(a), np.sin(a)])
            # 장애물까지 거리(원-광선 교차 근사: 샘플링 방식, 가볍고 충분히 정확)
            for t in np.linspace(0.05, self.RAY_RANGE, 25):
                p = self.pos + direction * t
                if np.linalg.norm(p) > self.ARENA_RADIUS:
                    dists[i] = t
                    break
                if np.any(np.linalg.norm(self.obstacles - p, axis=1) < self.OBSTACLE_RADIUS):
                    dists[i] = t
                    break
        return dists

    def _get_obs(self) -> np.ndarray:
        d = self._cast_rays()
        return (d / self.RAY_RANGE).astype(np.float32)

    def step(self, action):
        steer, accel = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))
        self.heading += steer * 1.5 * self.DT
        self.speed = float(np.clip(self.speed + accel * self.DT, 0.0, 2.0))
        prev_pos = self.pos.copy()
        self.pos = self.pos + self.speed * self.DT * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.steps += 1

        collided = bool(np.any(np.linalg.norm(self.obstacles - self.pos, axis=1) < self.OBSTACLE_RADIUS)
                        or np.linalg.norm(self.pos) > self.ARENA_RADIUS)
        progress = float(np.linalg.norm(self.pos - prev_pos))

        reward = progress - (5.0 if collided else 0.0) - (0.02 if self.speed < 0.1 else 0.0)
        terminated = collided
        truncated = self.steps >= self.MAX_STEPS

        obs = self._get_obs()
        info = {"collided": collided, "pos": self.pos.copy()}
        if _HAS_GYM:
            return obs, reward, terminated, truncated, info
        return obs, reward, (terminated or truncated), info


if __name__ == "__main__":
    print("[toy_obstacle_env] 랜덤 정책으로 동작 확인...")
    env = ToyObstacleEnv(seed=0)
    reset_out = env.reset()
    obs = reset_out[0] if _HAS_GYM else reset_out
    print(f"관측 shape={obs.shape}, 초기 관측 예시={np.round(obs[:5], 2)}")

    total_reward = 0.0
    rng = np.random.default_rng(0)
    for t in range(env.MAX_STEPS):
        action = rng.uniform(-1, 1, size=2)
        result = env.step(action)
        if _HAS_GYM:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        total_reward += reward
        if done:
            print(f"종료: step={t}, collided={info['collided']}, total_reward={total_reward:.2f}")
            break
    else:
        print(f"MAX_STEPS까지 생존, total_reward={total_reward:.2f}")
