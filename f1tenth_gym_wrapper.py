"""
f1tenth_gym_wrapper.py
========================
실제 f1tenth_gym(공식 시뮬레이터, 타이어 슬립 반영 동역학 모델) +
f1tenth_racetracks(실제 F1 트랙 20여 개, Silverstone/Spielberg/Austin 등)를
MultiTrackEnv와 동일한 인터페이스로 감싼 래퍼.

이 개발 환경의 공식 f110_gym(setup: gym==0.19, numpy<=1.22)은 Python 3.14와
호환되지 않는다. 본선은 `f1tenth_mapless_env.py` (racetracks occupancy 폴백)을 사용.

설치 (사용자 로컬 머신, GPU 노트북 등에서):
    git clone https://github.com/f1tenth/f1tenth_gym
    cd f1tenth_gym && pip install -e .
    git clone https://github.com/f1tenth/f1tenth_racetracks

실제 사용 가능한 맵 예시(파일명, f1tenth_racetracks/<맵이름>/ 아래):
    Austin, Catalunya, Hockenheim, IMS, Melbourne, MexicoCity, Montreal,
    Moscow, Nuerburgring, Oschersleben, Sakhir, SaoPaulo, Sepang, Shanghai,
    Silverstone, Sochi, Spa, Spielberg, YasMarina, Zandvoort 등
"""

from __future__ import annotations
import os
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


def load_centerline_csv(map_dir: str) -> np.ndarray:
    """f1tenth_racetracks의 <맵이름>_centerline.csv를 읽는다.
    포맷: x_m, y_m, w_tr_right_m, w_tr_left_m (README.md 기준으로 확인함)."""
    import glob
    candidates = glob.glob(os.path.join(map_dir, "*centerline.csv"))
    if not candidates:
        raise FileNotFoundError(f"{map_dir} 안에 *_centerline.csv 파일이 없습니다.")
    data = np.loadtxt(candidates[0], delimiter=",", skiprows=1)
    return data  # columns: x, y, w_right, w_left


class F1TenthGymMultiMapEnv(_EnvBase):
    """실제 f1tenth_gym을 여러 맵에 걸쳐 도는 MultiTrackEnv 호환 래퍼.

    MultiTrackEnv와 똑같이:
    - 관측은 LiDAR + 속도 + 각속도만 (위치 X)
    - 리워드에만 각 맵의 centerline.csv(정답 진행경로)를 씀
    - 매 에피소드 racetracks_dir 아래 맵 폴더 중 하나를 무작위로 골라 로드
    """

    N_RAYS_DOWNSAMPLE = 20  # 실제 f1tenth LiDAR(270~1080빔)를 이 개수로 다운샘플

    def __init__(self, racetracks_dir: str, map_names: list[str] | None = None,
                 seed: int | None = None):
        try:
            import f1tenth_gym  # noqa: F401  (설치 여부만 확인)
            from f1tenth_gym.envs import F110Env
        except ImportError as e:
            raise SystemExit(
                "f1tenth_gym이 설치되어 있지 않습니다.\n"
                "  git clone https://github.com/f1tenth/f1tenth_gym && cd f1tenth_gym && pip install -e .\n"
                f"(원본 에러: {e})"
            )

        self.racetracks_dir = racetracks_dir
        self.map_names = map_names or [
            d for d in os.listdir(racetracks_dir)
            if os.path.isdir(os.path.join(racetracks_dir, d))
        ]
        if not self.map_names:
            raise FileNotFoundError(f"{racetracks_dir} 안에서 맵 폴더를 찾지 못했습니다.")

        self.rng = np.random.default_rng(seed)
        self._F110Env = F110Env
        self._inner_env = None
        self.current_map = None
        self.centerline = None

        if _HAS_GYM:
            obs_dim = self.N_RAYS_DOWNSAMPLE + 2
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]),
                                            dtype=np.float32)

    def _load_map(self, name: str):
        map_dir = os.path.join(self.racetracks_dir, name)
        map_yaml = os.path.join(map_dir, f"{name}_map.yaml")
        self.current_map = name
        self.centerline = load_centerline_csv(map_dir)
        # F110Env 생성자 인자는 f1tenth_gym 버전에 따라 다를 수 있어, 정확한 인자명은
        # 설치 후 `help(F110Env.__init__)`으로 재확인 필요 — 아래는 확인된 일반적 패턴.
        self._inner_env = self._F110Env(map=map_yaml, num_agents=1)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        name = str(self.rng.choice(self.map_names))
        self._load_map(name)
        raw_obs, info = self._inner_env.reset()
        obs = self._convert_obs(raw_obs)
        info = {**info, "map_name": name}
        return obs, info

    def _convert_obs(self, raw_obs) -> np.ndarray:
        """f1tenth_gym의 원본 관측(보통 dict: scans, poses_x, poses_y, linear_vels_x,
        ang_vels_z 등 포함)에서 'LiDAR + 속도 + 각속도'만 뽑고 위치는 버린다.
        실제 키 이름은 f1tenth_gym 버전에 따라 다를 수 있어 설치 후 raw_obs를
        print()해서 정확한 키를 확인하고 아래를 맞출 것.
        """
        scan = np.asarray(raw_obs["scans"][0])
        n = len(scan)
        idx = np.linspace(0, n - 1, self.N_RAYS_DOWNSAMPLE).astype(int)
        scan_ds = scan[idx]
        scan_norm = np.clip(scan_ds / 10.0, 0, 1)  # 대략적 정규화, 실제 max_range로 교체 권장

        speed = raw_obs.get("linear_vels_x", [0.0])[0]
        yaw_rate = raw_obs.get("ang_vels_z", [0.0])[0]
        return np.concatenate([scan_norm, [np.clip(speed / 8.0, -1, 1)],
                                [np.clip(yaw_rate / 3.0, -1, 1)]]).astype(np.float32)

    def step(self, action):
        steer, speed_cmd = float(action[0]) * 0.4, float(np.clip(action[1], -1, 1)) * 4.0 + 4.0
        raw_obs, reward_raw, done, truncated, info = self._inner_env.step(
            np.array([[steer, speed_cmd]])
        )
        obs = self._convert_obs(raw_obs)
        # f1tenth_gym 기본 reward 대신, MultiTrackEnv와 동일한 centerline 기반
        # progress reward를 쓰려면 raw_obs의 poses_x/y와 self.centerline으로
        # 직접 계산해 교체할 것 (여기서는 자리표시자로 f1tenth_gym 기본값을 사용).
        return obs, float(reward_raw), bool(done), bool(truncated), info


if __name__ == "__main__":
    print("[f1tenth_gym_wrapper] 이 모듈은 f1tenth_gym 설치 후 사용하세요:")
    print("  git clone https://github.com/f1tenth/f1tenth_gym && cd f1tenth_gym && pip install -e .")
    print("  git clone https://github.com/f1tenth/f1tenth_racetracks")
    print("  python -c \"from f1tenth_gym_wrapper import F1TenthGymMultiMapEnv; "
          "e = F1TenthGymMultiMapEnv('f1tenth_racetracks'); print(e.reset())\"")
