# CLAUDE.md — 이 레포에서 작업할 때 먼저 읽을 것

F1TENTH / Roboracer 2026 (아주대). **맵·위치추정 없이(mapless) 센서 → 강화학습 정책 → `/drive`** 로 달리는 것이 목표.
실차 컴퓨터 Jetson Orin Nano (6코어, 추론은 torch 없이 numpy). 학습은 Colab 무료 T4 (노트북 확보 후 로컬도).
사용자와는 **한국어**로, 헷갈리지 않게 짧고 정확하게. 추정값과 실측값을 섞어 말하지 말 것.

브랜치: **`feat/mapless40-asym-sac`** (모든 작업 여기). 원격: github.com/tkddn647-ship-it/2027_F1tenth_test

---

## 1. 지금 살아 있는 세 갈래

**카메라 작업은 전부 [`camera/`](camera/README.md) 폴더 안에서.** LiDAR 는 루트 `mapless40/`.

| 폴더 | 센서 | 정책 | 상태 |
|--|--|--|--|
| [`mapless40/`](mapless40/RESULTS.md) | LiDAR 1125빔 40 Hz + IMU + 속도 | 1D CNN → MLP (SAC, asymmetric) | **학습 완료(v3), 실차 시험 대기.** v4 (횡가속 4.5 제한) 코랩에서 이어 학습 중 |
| [`camera/camfly/`](camera/camfly/README.md) | Gemini 2L RGB(흑백) + depth 스캔 | 합성 초파리 커넥톰 (비교: 작은 CNN) | 구조만. 학습 안 함 |
| [`camera/depthfly/`](camera/depthfly/README.md) | **Gemini 2L depth 만** | **초파리 커넥톰만** (CNN·MLP 없음) | **현재 주력.** numpy 테스트 6/6, torch 테스트·학습 아직 |

레포 루트의 옛 파일들(`train_sac_*.py`, `f1tenth_mapless_env.py`, `connectome_*.py`, `watch_*.py` …)은 **레거시**. 새 작업에 쓰지 말 것.
`Roboracer-2026-main/` = 실차 ROS2 스택 (control_node, 센서, TF). `realcar/` = mapless40 실차 실행 스크립트. `sim_ros2/` = f1tenth_gym_ros 브리지.

## 2. 환경 / 실행

```bash
pip install torch "stable-baselines3>=2.3" "gymnasium>=0.29" numpy scipy pyyaml pillow matplotlib tensorboard
python -m camera.depthfly.tests          # numpy 6 + torch 2  (torch 있으면 8개 다 돌아야 정상)
python -m camera.camfly.tests
python -m mapless40.tests
```
- 모든 명령은 **레포 루트에서 `python -m 패키지.모듈`** 로 (패키지 간 import: camera.depthfly → camera.camfly → mapless40).
- Windows: `--subproc` (SubprocVecEnv) 은 `if __name__ == "__main__"` 가드가 있어 동작해야 하지만, 문제가 나면 `--subproc` 빼고 실행.
- 맵: `maps/` (팀 맵 `roboracer_0817`, `ifac_roboracer` 등, 전부 LiDAR SLAM 2D 지도) + `f1tenth_racetracks/` (Spielberg, Silverstone, Monza … 실제 서킷 축소, 센서로 만든 것 아님).
  맵 지정 `"이름:가중치"` 예: `ifac:3,roboracer_0817:3,Spielberg:1`.

## 3. 공통 설계 (세 갈래 모두)

- **차량 시뮬** `mapless40/env.py` `MaplessRaceEnv40`: 단일 트랙 모델 RK4, 서보·구동 지연, 능동 브레이크 없음, 횡가속 한계, μ ±10%.
  수치 출처는 `mapless40/RESULTS.md §3` (서보·구동·타력 감속은 **추정값**).
- **SAC asymmetric actor-critic** (`mapless40/policy.py` `AsymSACPolicy`): actor 는 센서·상태만, critic 은 추가로
  **privileged 27** (레이싱라인 오차·곡률·v_ref, 앞 장애물). 레이싱라인은 **학습 신호로만** 쓰이고 실차에는 없다.
- 센서별 환경 = `MaplessRaceEnv40` 상속 후 `_raw_scan()` 만 바꿈 → `self.hist.push(frame, v̄, ω̄, ω_latest, dt)` → `observation(priv)`.
- 행동: 조향 ±21.4°, 속도 2~5 m/s (실차 배터리 한계 6 m/s). 실차는 `max_speed` 파라미터로 더 낮춤 (처음 2.0).
- 배포: 학습 zip → `*/np_actor.py` (torch 없이, `mapless40.np_actor.load_sb3_zip`) → `*/ros_node.py`.
- 학습 저장: `--save-dir` 에 `last_model.zip`(원자적 저장), `best_model.zip`, `best.json`(평가 조건 태그). `--resume auto` 로 이어 학습
  (이어 할 때 워밍업은 랜덤이 아니라 **정책 행동**으로 — `_PolicyWarmupSAC`).

## 4. depthfly 요약 (자세히: [camera/depthfly/README.md](camera/depthfly/README.md))

```
depth (Gemini 2L, 30 fps) → 파리 눈 격자 16×64 '가까움' = 0.25 m / 수평거리 (바닥 제거, 7 m 밖·구멍 = 0, 구멍 2프레임 유지)
  × 최근 3프레임 → lamina ON/OFF → T4/T5 (4방향) → HS/VS 32, LPLC2 8, LC 32, L3 24  (= 96)
  → DN 48 (고정 희소 배선 fan-in 12, 부호 고정, 세기만 학습) → tanh ‖ 상태 14 → 선형 읽기 → (조향, 속도)
```
- 30 Hz 인 이유: Gemini 2L depth 최대 30 fps. γ = 0.99^(10/30).
- 시뮬 depth 는 **2D 맵의 벽을 33 cm 덕트로 세워** 레이캐스팅 + 추정 오차(σ = 0.005·d², 구멍 2% + 15%·(d/7)²). **실측 아님.**
- Jetson 부하: 칸 샘플링 ~2 ms + 회로 ~0.5 ms (클라우드 CPU 측정).

## 5. 결정된 것 (사용자와 합의)

- 카메라 갈래는 **depth 만, 커넥톰만, 과거 3프레임, 30 Hz**. RGB·LiDAR·CNN·MLP 안 씀.
- **맵은 LiDAR 로 만든 2D 맵 그대로 쓴다.** 맵 = 벽의 위치(기하)이고 LiDAR 가 더 정확. 실제 카메라 데이터로 맞춰야 하는 건
  **센서 모델**(노이즈·구멍·가짜 점·지연)이다. 카메라로 매핑한 공개 레이싱 맵은 없음 (3D 시뮬레이터 트랙은 있음: AutoDRIVE F1TENTH, Isaac Sim — 지금 입력 해상도엔 이득 작음).
- 카메라가 오면: LiDAR 와 같이 달고 수동 주행 rosbag → LiDAR 를 정답으로 센서 모델 보정 → 재학습.
- **git stash 의 domain randomization (`WIP domain randomization`) 은 커밋·푸시 금지** (실측 후 재검토).

## 6. 다음 할 일 (우선순위)

### depthfly — 카메라 오기 전 시뮬 검증
1. **torch 테스트**: `python -m camera.depthfly.tests` → 8/8 (회로 torch == numpy, SAC 루프, numpy actor == torch). 실패하면 여기부터.
2. **짧은 학습 확인**: `python -m camera.depthfly.train --timesteps 60000 --learning-starts 5000 --n-envs 4 --save-dir runs/df_smoke`
   → steps/s 기록, 에피소드 보상·진행률이 오르는지.
3. **본 학습**: 코랩 `camera/depthfly/colab_train.ipynb` 또는 로컬 `--timesteps 1000000 --resume auto`.
4. **평가**: `python -m camera.depthfly.evaluate --model runs/.../best_model.zip --maps ifac,roboracer_0817 --obstacles 0` 와 `--obstacles 2`.
   비교 기준: pure pursuit ifac 14.1 s (depthfly env), mapless40 v3 ifac 10.9 s / 팀 맵 9.63 s.
5. **센서 스트레스 평가 (구현 필요)**: evaluate 에 옵션 추가 — 노이즈 ×2·×3, 구멍 ×2, **섹터 통째 구멍**(은박·검정 덕트 반사 가정),
   pitch ±3°, 높이 ±2 cm, 지연 +30 ms. 어느 조건에서 무너지는지 표로.
6. **알려진 약점**: 덕트(33 cm)가 3 m 밖에서 16행 중 1~2행. 행을 지평선 근처로 몰기 (`CameraSpec.row_elevations` 를 열의 acute zone 처럼) → 재학습 비교.
7. **시뮬 보강**: 규정의 덕트 사이 틈, 가장자리 가짜 점(flying pixel), 실측 지연.
8. **카메라 도착 후**: rosbag(depth + camera_info + /scan + IMU + 속도) → 실제 가까움 영상 vs LiDAR 비교 스크립트 → σ·구멍 모델 맞춤.

### mapless40
- v4 이어 학습 결과 평가, 실차 `MAX_SPEED=2.0` 시험 (`realcar/README.md`).
- 팀 맵에 실제 트랙 경계 그려 넣고 재평가. 서보·타력 감속 실측 → config 반영.

## 7. 주의

- 학습된 모델 zip 은 대부분 레포에 없음 (사용자 구글 드라이브 `mapless40_runs/…`). `mapless40/last_model.zip` 만 있음.
- 옛 mapless40 모델은 zip 안의 `lidar_cfg`(range_max 등)를 자동 적용 — config 기본값 바꿔도 옛 모델 평가가 깨지지 않게 유지할 것.
- 커밋 전 해당 패키지 `tests` 를 돌릴 것.
