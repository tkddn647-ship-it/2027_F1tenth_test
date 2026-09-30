"""
mapless40.env
=============
40 Hz mapless 레이싱 env (Gymnasium Dict 관측).

타이밍 (README §11 과 동일):
  step(a_n) 호출 시점 = 스캔 d_{n-1} 이 막 끝난 시각 t_n.
  관측 o_n 은 d_{n-4..n-1} 과 그 구간들의 v̄, ω̄, Δt, 그리고 직전 명령 2개.
  a_n 은 compute_latency 뒤부터 [t_n, t_{n+1}] 구간에 적용되고, 구간 끝에 d_n 이 찍힌다.

관측:
  scan  (4, 1125) float16   LiDAR 0~1  (실차와 같은 preprocess_scan 을 거침)
  state (17,)     float32   속도·yaw rate·직전 명령·Δt  (obs_builder.ObsHistory 참고)
  priv  (24,)     float32   레이싱라인 privileged — critic 전용, actor 는 안 봄

센서: IMU 100 Hz, 속도 50 Hz 를 5 ms 서브스텝에서 샘플링해 구간 평균.
동역학: vehicle_dynamics.py Single-Track + actuators.py 서보/구동 모델.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from f1tenth_mapless_env import _load_occupancy, _resolve_map_yaml  # noqa: E402
from vehicle_dynamics import STParams, integrate_st_rk4  # noqa: E402

from .actuators import DriveModel, ServoModel  # noqa: E402
from .config import PRIV_DIM, STATE_DIM, EnvConfig  # noqa: E402
from .obs_builder import IntervalAverager, ObsHistory, action_to_command, preprocess_scan  # noqa: E402
from .raceline import Raceline, load_raceline  # noqa: E402
from .raycast import GridMap, LidarSim  # noqa: E402


@dataclass
class Track:
    name: str
    grid: GridMap
    line: Raceline
    line_file: str


def load_track(name: str, cfg: EnvConfig) -> Track:
    y = _resolve_map_yaml(name)
    occ, res, origin, meta = _load_occupancy(y)
    if str(meta.get("mode", "")).lower() == "trinary":
        # Cartographer 저장 맵: 254 = 빈칸, 205 = 미탐색, 0 = 벽.  미탐색을 빈칸으로 두면
        # 트랙 안쪽 섬·바깥이 뚫린 걸로 보여 라인이 벽을 가로지른다 → 빈칸만 주행 가능.
        from PIL import Image
        img = np.asarray(Image.open(y.parent / meta["image"]).convert("L"))
        occ = img < 250
    grid = GridMap(occ, res, origin)
    a = cfg.act
    if a.brake_enabled:
        decel = lambda v: 0.9 * a.brake_decel_max  # noqa: E731
    else:
        decel = lambda v: 0.9 * (a.coast_decel_c0 + a.coast_decel_c1 * v)  # noqa: E731
    line, f = load_raceline(y, cfg.raceline, grid, decel)
    grid.release_grad()
    return Track(name=name, grid=grid, line=line, line_file=f.name)


class MaplessRaceEnv40(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, maps=("Spielberg",), cfg: EnvConfig | None = None,
                 seed: int | None = None, sensor_noise: bool = True,
                 randomize: bool = True, tracks: list[Track] | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        if isinstance(maps, str):
            maps = [m for m in maps.split(",") if m]
        # "ifac:3" 처럼 가중치를 붙이면 그 비율로 트랙을 뽑는다 (기본 1)
        names, weights = [], []
        for m in maps:
            n, _, w = str(m).partition(":")
            names.append(n)
            weights.append(float(w) if w else 1.0)
        self.tracks = tracks if tracks is not None else [load_track(m, self.cfg) for m in names]
        w = np.asarray(weights[:len(self.tracks)] if tracks is None else [1.0] * len(self.tracks))
        self.track_p = w / w.sum()
        self.sensor_noise = sensor_noise
        self.randomize = randomize
        c = self.cfg
        self.n_beams = c.lidar.n_beams
        self.observation_space = spaces.Dict({
            "scan": spaces.Box(0.0, 1.0, (c.timing.hist, self.n_beams), np.float16),
            "state": spaces.Box(-5.0, 5.0, (STATE_DIM,), np.float32),
            "priv": spaces.Box(-10.0, 10.0, (PRIV_DIM,), np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.hist = ObsHistory(c.lidar, c.norm, c.act, c.action, c.timing.hist)
        self.base_params = STParams.roboracer()
        self.base_params.s_min, self.base_params.s_max = -c.act.steer_max, c.act.steer_max
        self.base_params.sv_min, self.base_params.sv_max = -c.act.servo_rate_max, c.act.servo_rate_max
        self.base_params.a_max = max(c.act.accel_max, c.act.brake_decel_max, 8.0)
        self.base_params.v_min = 0.0
        self.base_params.v_max = c.action.v_max * 1.3
        if c.mu is not None:
            self.base_params.mu = c.mu
        self.track: Track | None = None
        self.obstacles = np.zeros((0, 3))
        self.obs_s = np.zeros(0)
        self.obs_lat = np.zeros(0)

    # ------------------------------------------------------------------ utils
    def _footprint_hit(self, x: float, y: float, yaw: float) -> bool:
        c = self.cfg
        f, r, hw = c.front, c.rear, c.half_width
        lx = np.array([f, f, f, 0.0, 0.0, -r, -r, 0.5 * f, 0.5 * f])
        ly = np.array([hw, 0.0, -hw, hw, -hw, hw, -hw, hw, -hw])
        cy, sy = np.cos(yaw), np.sin(yaw)
        px = x + lx * cy - ly * sy
        py = y + lx * sy + ly * cy
        if self.track.grid.occupied(px, py).any():
            return True
        ob = getattr(self, "obstacles", None)
        if ob is not None and len(ob):
            d2 = (px[:, None] - ob[None, :, 0]) ** 2 + (py[:, None] - ob[None, :, 1]) ** 2
            if (d2 < ob[None, :, 2] ** 2).any():
                return True
        return False

    def _clearance(self) -> float:
        """차체 옆면 기준 가장 가까운 벽·장애물까지 여유 [m] (CG·앞·뒤 3점 중 최소)."""
        c, (x, y, yaw) = self.cfg, (self.state[0], self.state[1], self.state[4])
        lx = np.array([0.0, 0.7 * c.front, -0.7 * c.rear])
        px, py = x + lx * np.cos(yaw), y + lx * np.sin(yaw)
        d = float(self.track.grid.distance(px, py).min())
        ob = self.obstacles
        if len(ob):
            d = min(d, float((np.hypot(px[:, None] - ob[None, :, 0], py[:, None] - ob[None, :, 1])
                              - ob[None, :, 2]).min()))
        return d - c.half_width

    def _raw_scan(self) -> np.ndarray:
        raw = self.lidar.scan(self.state[0], self.state[1], self.state[4], noise=self.sensor_noise,
                              obstacles=self.obstacles)
        sp = self.cfg.lidar
        return preprocess_scan(raw, -sp.fov / 2.0, sp.angle_inc, sp)

    def _priv(self) -> np.ndarray:
        c = self.cfg
        line = self.track.line
        v = self.state[3]
        v_ref = float(line.at_s(self.s, line.v_ref))
        kap, vr = line.lookahead(self.s)
        e_psi = line.heading_error(self.line_idx, self.state[4])
        # 앞 10 m 안 가장 가까운 장애물: (거리/10, 라인 기준 옆 위치 − 내 e_y, 반지름/0.3). 없으면 (1, 0, 0)
        ob_feat = [1.0, 0.0, 0.0]
        if len(self.obstacles):
            ds = (self.obs_s - self.s) % line.length
            j = int(np.argmin(ds))
            if ds[j] < 10.0:
                ob_feat = [ds[j] / 10.0, float(np.clip(self.obs_lat[j] - self.e_y, -3, 3)),
                           self.obstacles[j, 2] / 0.3]
        p = np.concatenate([
            [np.clip(self.e_y, -3, 3), e_psi, (v - v_ref) / c.norm.v_scale, np.clip(self.state[6], -1, 1)],
            np.clip(kap, -3, 3),
            vr / c.norm.v_scale,
            ob_feat,
        ]).astype(np.float32)
        return p

    def _place_obstacles(self, n: int, s_spawn: float) -> None:
        """레이싱라인 근처에 원 장애물 n 개. 한쪽은 min_gap 이상 지나갈 틈을 남긴다."""
        from .raycast import cast_rays
        oc = self.cfg.obstacles
        line, grid = self.track.line, self.track.grid
        obs, ss, lats = [], [], []
        for _ in range(60 * max(n, 1)):
            if len(obs) >= n:
                break
            s_o = (s_spawn + self.rng.uniform(oc.min_ahead, line.length - 3.0)) % line.length
            if any(abs(line.ds_wrap(s_o, q)) < oc.min_sep for q in ss):
                continue
            r = float(self.rng.uniform(*oc.r_range))
            lat = float(self.rng.uniform(-oc.lat_range, oc.lat_range))
            i = int(round(s_o / line.ds)) % line.n
            psi = line.psi[i]
            cx, cy = line.x[i] - lat * np.sin(psi), line.y[i] + lat * np.cos(psi)
            if float(grid.distance(np.array([cx]), np.array([cy]))[0]) < r + 0.05:
                continue
            side = cast_rays(grid, cx, cy, psi, np.array([np.pi / 2, -np.pi / 2]), 5.0)
            if (side - r).max() < oc.min_gap:
                continue
            obs.append([cx, cy, r]); ss.append(s_o); lats.append(lat)
        self.obstacles = np.array(obs, dtype=np.float64).reshape(-1, 3)
        self.obs_s = np.array(ss); self.obs_lat = np.array(lats)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        c = self.cfg
        if "map" in options:
            self.track = next(t for t in self.tracks if t.name == options["map"])
        else:
            self.track = self.tracks[int(self.rng.choice(len(self.tracks), p=self.track_p))]
        self.lidar = LidarSim(self.track.grid, c.lidar, self.rng)
        line = self.track.line

        self.params = STParams(**vars(self.base_params))
        self.a_lat_cap = c.a_lat_cap
        if self.randomize and c.mu_rand > 0:
            self.params.mu *= float(self.rng.uniform(1 - c.mu_rand, 1 + c.mu_rand))
        if self.randomize and c.a_lat_cap_rand > 0:
            self.a_lat_cap *= float(self.rng.uniform(1 - c.a_lat_cap_rand, 1 + c.a_lat_cap_rand))

        self.obstacles = np.zeros((0, 3))
        for _ in range(100):
            s0 = float(options.get("s0", self.rng.uniform(0, line.length)))
            lat = float(options.get("lat", np.clip(self.rng.normal(0, c.spawn_lat_std), -0.4, 0.4)))
            i = int(round(s0 / line.ds)) % line.n
            psi = line.psi[i]
            x = line.x[i] - lat * np.sin(psi)
            y = line.y[i] + lat * np.cos(psi)
            yaw = psi + float(options.get("dyaw", self.rng.normal(0, c.spawn_heading_std)))
            self.track_hint = i
            if float(self.track.grid.distance(np.array([x]), np.array([y]))[0]) > 0.35 and \
                    not self._footprint_hit_pose(x, y, yaw):
                break
            options = {k: v for k, v in options.items() if k not in ("s0", "lat")}
        v0 = float(options.get("v0", self.rng.uniform(*c.spawn_v_range)))
        self.state = np.array([x, y, 0.0, v0, yaw, 0.0, 0.0], dtype=np.float64)

        oc = c.obstacles
        if "n_obstacles" in options:
            n_ob = int(options["n_obstacles"])
        else:
            n_ob = int(self.rng.integers(1, oc.max_n + 1)) if self.rng.random() < oc.prob else 0
        self.obstacles = np.zeros((0, 3)); self.obs_s = np.zeros(0); self.obs_lat = np.zeros(0)
        s_spawn = float(line.project(x, y, None)[1])
        if n_ob > 0:
            self._place_obstacles(n_ob, s_spawn)

        dt = c.timing.phys_dt
        self.servo = ServoModel(c.act, dt, 0.0)
        self.drive = DriveModel(c.act, dt, v0)
        self.imu = IntervalAverager(default=0.0)
        self.spd = IntervalAverager(default=v0)
        self.gyro_bias = float(self.rng.normal(0, c.imu.gyro_bias_std)) if self.sensor_noise else 0.0
        self.t = 0.0
        self.phys_step = 0
        self.t_last_scan = 0.0
        self.cmd = (0.0, v0)
        self.cmd_prev = self.cmd

        self.line_idx, self.s, self.e_y = line.project(x, y, None)
        self.s_travel = 0.0
        self.laps = 0
        self.lap_times: list[float] = []
        self.t_lap_start = 0.0
        self.reverse_t = 0.0
        self.max_abs_beta = 0.0

        self.hist.reset(self._raw_scan(), v0, 0.0, self.cmd)
        obs = self.hist.observation(self._priv())
        return obs, {"map": self.track.name, "line_file": self.track.line_file,
                     "obstacles": self.obstacles.copy()}

    def _footprint_hit_pose(self, x, y, yaw) -> bool:
        return self._footprint_hit(x, y, yaw)

    # ------------------------------------------------------------------ step
    def _run_substeps(self, n_sub: int, cmd_new, cmd_old) -> bool:
        c = self.cfg
        dt = c.timing.phys_dt
        lat_steps = int(round(c.timing.compute_latency / dt))
        for k in range(n_sub):
            steer_c, v_c = cmd_old if k < lat_steps else cmd_new
            sv = self.servo.steer_velocity(steer_c, self.state[2])
            acc = self.drive.accel(v_c, self.state[3])
            self.state = integrate_st_rk4(self.state, np.array([sv, acc]), self.params, dt)
            if self.state[3] < 0.0:
                self.state[3] = 0.0
            v = self.state[3]
            if v > 0.5 and abs(v * self.state[5]) > self.a_lat_cap:   # 마찰 한계: 요레이트 포화
                self.state[5] = np.sign(self.state[5]) * self.a_lat_cap / v
            self.t += dt
            self.phys_step += 1
            self.max_abs_beta = max(self.max_abs_beta, abs(self.state[6]))
            if self.phys_step % c.timing.imu_every == 0:
                noise = self.rng.normal(0, c.imu.gyro_noise_std) if self.sensor_noise else 0.0
                self.imu.push(self.t, self.state[5] + self.gyro_bias + noise)
            if self.phys_step % c.timing.speed_every == 0:
                noise = self.rng.normal(0, c.imu.speed_noise_std) if self.sensor_noise else 0.0
                self.spd.push(self.t, max(0.0, self.state[3] + noise))
            if self._footprint_hit(self.state[0], self.state[1], self.state[4]):
                return True
        return False

    def step(self, action):
        c = self.cfg
        r = c.reward
        steer_cmd, v_cmd = action_to_command(np.asarray(action, dtype=np.float64), c.act, c.action)
        self.cmd_prev, self.cmd = self.cmd, (steer_cmd, v_cmd)

        n_sub = c.timing.substeps
        if self.randomize:
            if self.rng.random() < c.timing.scan_drop_prob:
                n_sub *= 2                                   # 스캔 누락 → 명령 2틱 유지
            elif self.rng.random() < c.timing.scan_jitter_prob:
                n_sub += int(self.rng.choice([-1, 1]))       # ±5 ms
        t0 = self.t
        collided = self._run_substeps(n_sub, self.cmd, self.cmd_prev)
        dt_tick = self.t - t0

        # --- progress along raceline
        line = self.track.line
        s_old = self.s
        self.line_idx, self.s, self.e_y = line.project(self.state[0], self.state[1], self.line_idx)
        ds = line.ds_wrap(s_old, self.s)
        self.s_travel += ds
        lap_done = False
        if self.s_travel >= (self.laps + 1) * line.length:
            self.laps += 1
            self.lap_times.append(self.t - self.t_lap_start)
            self.t_lap_start = self.t
            lap_done = True
        self.reverse_t = self.reverse_t + dt_tick if ds < -1e-3 else 0.0

        # --- new scan d_n + sensor aggregation over this interval
        scan = self._raw_scan()
        v_bar = self.spd.mean(self.t_last_scan, self.t)
        w_bar = self.imu.mean(self.t_last_scan, self.t)
        self.t_last_scan = self.t
        self.hist.push(scan, v_bar, w_bar, self.imu.latest(), dt_tick)
        self.hist.push_cmd(steer_cmd, v_cmd)

        # --- reward
        v = self.state[3]
        v_ref = float(line.at_s(self.s, line.v_ref))
        dv = v - v_ref
        rew = r.w_progress * ds
        rew -= r.w_ey * self.e_y ** 2 * dt_tick
        rew -= r.w_v * dv * dv * (r.over_mult if dv > 0 else 1.0) * dt_tick
        rew -= r.w_dsteer * abs(self.cmd[0] - self.cmd_prev[0]) / c.act.steer_max
        rew -= r.w_slip * max(abs(self.state[6]) - 0.05, 0.0) * dt_tick
        if r.w_wall > 0:
            clear = self._clearance()
            if clear < r.wall_clear:
                rew -= r.w_wall * (r.wall_clear - max(clear, 0.0)) / r.wall_clear * dt_tick
        if lap_done:
            rew += r.lap_bonus
        reversed_ = self.reverse_t >= r.reverse_s
        if collided:
            rew = r.collision
        terminated = bool(collided or reversed_)
        truncated = bool(self.t >= c.max_episode_s)

        obs = self.hist.observation(self._priv())
        info = {
            "map": self.track.name, "collided": collided, "reversed": reversed_,
            "laps": self.laps, "lap_times": list(self.lap_times), "lap_completed": lap_done,
            "progress": self.s_travel / line.length, "speed": float(v), "v_ref": v_ref,
            "e_y": self.e_y, "slip": float(self.state[6]), "yaw_rate": float(self.state[5]),
            "steer": float(self.state[2]), "t": self.t, "dt_tick": dt_tick,
            "x": float(self.state[0]), "y": float(self.state[1]), "yaw": float(self.state[4]),
            "n_obstacles": int(len(self.obstacles)),
        }
        return obs, float(rew), terminated, truncated, info


def make_env(maps, seed: int = 0, cfg: EnvConfig | None = None, **kw):
    """SB3 make_vec_env / SubprocVecEnv 용 팩토리."""
    def _f():
        return MaplessRaceEnv40(maps=maps, cfg=cfg, seed=seed, **kw)
    return _f
