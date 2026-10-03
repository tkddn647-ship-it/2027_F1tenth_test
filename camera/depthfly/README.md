# depthfly — depth 카메라 하나 + 초파리 커넥톰만

**Orbbec Gemini 2L 의 depth 영상만** 쓴다. RGB·LiDAR·CNN·MLP·별도 depth 경로 없음.
depth 를 파리 눈 격자(16×64)의 **가까움 영상**으로 바꿔 광수용체 자리에 넣고, 합성 커넥톰 시각엽을 지난
하행 뉴런(DN) 48개를 **선형으로 읽어** 조향·속도를 낸다. 과거 3프레임, 30 Hz.
차량 동역학·보상·레이싱라인(critic 전용)·장애물·평가는 [`mapless40`](../../mapless40/RESULTS.md), 카메라 기하·회로 부품은 [`camfly`](../camfly/README.md) 를 쓴다.

![](results/sim_depth_eye_pp.gif)

*학습 전 (pure pursuit 주행, 초기 회로). 왼쪽: 트랙·91° 시야·장애물(빨강). 오른쪽 위→아래: 가까움 영상 16×64 (바닥 제거, 덕트만),
프레임 변화 n−(n−1) (T4/T5 입력), 시각엽 출력 96, DN 48.*

## 1. 구조

```
Gemini 2L depth (칩에서 계산, 30 fps)
   │  칸마다 주변 픽셀 → 차체 좌표 → 바닥(5 cm 아래) 제거 → 가장 가까운 점
   ▼
가까움 영상 16×64  (0.25 m / 수평거리, 7 m 밖·구멍 = 0, 구멍은 2프레임까지 직전 값 유지)  × 최근 3프레임 (n-2, n-1, n)
   │
   ├ lamina  ON/OFF = ±Δ가까움 (빠른 n−(n−1), 느린 (n−1)−(n−2))
   ├ T4/T5   Reichardt 상관기 4방향, 방향별 이득 학습
   ├ HS/VS   섹터 8 × 밴드 2 수평·수직 흐름            16 + 16
   ├ LPLC2   섹터별 바깥쪽 흐름 = 다가옴                 8
   ├ LC      섹터·밴드별 가장자리 (덕트·장애물 경계)     32
   └ L3      섹터·밴드별 평균 가까움 + 섹터별 최대       16 + 8      = 96
   ▼
DN 48   고정 희소 배선 (입력 12개, 부호 고정) × 학습되는 세기 → tanh
   ▼  ‖ 상태 14 (v̄×3, ω̄×3, ω, 직전 명령 2, Δt×3)
선형 읽기 → (조향 ±21.4°, 속도 2~5 m/s)
critic (학습 전용): 같은 회로 + 레이싱라인·장애물 privileged 27 → MLP 256·256
```

실제로 쓰이는 학습 값 ≈ 760개 (이득 13, DN 세기 48×12 = 576, DN 바이어스 48, 선형 읽기 62×2+2). 배선(누가 누구에게, 흥분/억제)은 고정.

**왜 30 Hz:** Gemini 2L depth 최대가 30 fps (40 fps 모드 없음). 과거 3프레임 = 0.1 s 의 움직임.
**Jetson 부하:** depth 는 카메라가 계산. Jetson 은 칸 샘플링 ~2 ms + 회로 ~0.5 ms (노트북 CPU 측정, Orin Nano 는 2~3배 예상) → 33 ms 예산의 10~20 %, CPU 1코어. GPU·torch 불필요.

## 2. 파일

| 파일 | 내용 |
|--|--|
| `config.py` | `DepthFlyConfig(fps=30, hist=3)` (camfly 설정 재사용: 91°×66°, 높이 0.18 m, 10° 숙임, depth 7 m, σ = 0.005·d², 구멍) |
| `sensor.py` | `DepthEyeSim` (시뮬: 레이캐스팅 → 덕트·장애물만, 잡음·구멍), `DepthEyeReal` (실차 depth 영상 → 같은 격자), `HoleHold` |
| `brain.py` | 커넥톰 회로 numpy 참조 + torch `DepthFlyBrain` (같은 수식) |
| `env.py` | `DepthFlyEnv` = mapless40 환경 + 관측 near(3,16,64)·state·priv |
| `policy.py` / `np_actor.py` | SB3 특징 추출기, 배포용 actor / 학습 zip → numpy actor (torch 없이) |
| `train.py` / `evaluate.py` / `viz.py` | 학습 · 평가 · 회로 활동 GIF |
| `ros_node.py` | Jetson ROS2: depth + camera_info + IMU + 속도 → `/drive`, depth 프레임마다 1회 |
| `tests.py` | numpy 6 (시뮬 센서 기하, 구멍 유지, 합성 depth 영상 → 가까움, 방향·다가옴 선택성, 속도, PP 완주) + torch 2 |
| `colab_train.ipynb` | 코랩: 테스트 → GIF → 학습 → 평가 → 학습된 정책 GIF (`mapless40_runs/depthfly`) |

## 3. 사용

```bash
python -m camera.depthfly.tests
python -m camera.depthfly.viz --map ifac --out eye.gif
python -m camera.depthfly.train --n-envs 4 --subproc --save-dir runs/depthfly --resume auto
python -m camera.depthfly.evaluate --model runs/depthfly/best_model.zip --maps ifac,roboracer_0817 --obstacles 2
```

실차 (Jetson):
```bash
ros2 launch orbbec_camera gemini2L.launch.py enable_color:=false depth_fps:=30     # 인자 이름은 드라이버 버전에 맞게
python3 -m camera.depthfly.ros_node --ros-args -p model:=best_model.zip -p max_speed:=2.0 -p cam_pitch_deg:=-10.0
```
- `cam_pitch_deg` 는 실제 장착 각도. 높이(0.18 m)를 바꾸면 `CameraSpec.height` 맞추고 재학습.
- 토픽 이름은 `ros2 topic list` 로 확인 후 `-p depth_topic:=... -p info_topic:=...`.

## 4. 검증된 것 / 아직 아닌 것

- 여기서 (numpy): 시뮬 센서 기하 (3 m 벽 → 덕트 높이 행만, 바닥 제거), 합성 Gemini depth 영상 → 4 m 벽 가까움 ±8 %,
  T4/T5 왼/오 부호, LPLC2 다가옴 선택, 정지 장면 → 움직임 0, ifac pure pursuit 완주 (14.1 s, env 3.3 ms/step).
- **torch 부분 (회로 torch == numpy, SAC 루프, numpy actor == torch) 은 코랩 ④ 에서 처음 돈다.**
- 학습 결과 아직 없음.

## 5. 시뮬 depth 는 무엇이고, 무엇이 실측이 아닌가

| 부분 | 지금 시뮬 | 출처 |
|--|--|--|
| 트랙 기하 | LiDAR SLAM 2D 맵 (`maps/`) · 서킷 축소 맵 (`f1tenth_racetracks/`) 의 벽을 **높이 33 cm 덕트**로 세움 | 맵 = 실측(LiDAR), 높이 = 규정 |
| 장애물 | 원통 r 0.12~0.30 m, 높이 `SceneSpec.obstacle_height` | 가정 |
| 카메라 | 91°×66°, 높이 0.18 m, 아래로 10°, 차 앞 0.139 m | 사양 + **가정 장착** |
| depth 오차 | σ = 0.005·d² (4 m 에서 8 cm) | **추정** (비슷한 카메라 사양) |
| depth 구멍 | 확률 2 % + 15 %·(d/7)² , 칸마다 독립 | **추정** |
| 지연 | 계산 30 ms | **추정** |
| 없음 | 덕트 사이 틈, 가장자리 가짜 점, 반사 재질(은박·검정)로 인한 큰 구멍, 햇빛, 진동 | — |

**맵은 LiDAR 맵이 맞다.** 맵은 벽이 어디 있는지(기하)이고 LiDAR 가 카메라보다 정확하다. 실제 카메라 데이터로 맞춰야 하는 건
**센서 모델**(위 표의 추정 칸)이다. 카메라로 매핑한 공개 레이싱 맵은 찾지 못했다. 카메라가 달린 3D 시뮬레이터
(AutoDRIVE F1TENTH, Isaac Sim) 는 있지만 입력이 16×64 가까움 영상이라 지금은 이득이 작다 (RGB 를 쓸 때 다시 검토).

## 6. 카메라 오기 전 시뮬 검증 계획

| # | 할 일 | 명령 / 방법 | 통과 기준 |
|--|--|--|--|
| 1 | torch 테스트 | `python -m camera.depthfly.tests` | 8/8 |
| 2 | 시각화 확인 | `python -m camera.depthfly.viz --map roboracer_0817 --obstacles 2 --out x.gif` | 덕트 띠·장애물이 보이고, 바닥·구멍 깜빡임 없음 |
| 3 | 짧은 학습 | `python -m camera.depthfly.train --timesteps 60000 --learning-starts 5000 --n-envs 4 --save-dir runs/df_smoke` | steps/s 기록, 진행률·보상 상승 |
| 4 | 본 학습 | 코랩 노트북 또는 `--timesteps 1000000 --resume auto` | `best.json` 갱신 |
| 5 | 평가 | `python -m camera.depthfly.evaluate --model …/best_model.zip --maps ifac,roboracer_0817 --obstacles 0` (그리고 `2`) | 비교: PP 14.1 s (ifac), mapless40 v3 10.9 s / 팀 맵 9.63 s |
| 6 | **센서 스트레스 평가 (구현 필요)** | evaluate 에 옵션: 노이즈 ×2·×3, 구멍 ×2, 섹터 통째 구멍, pitch ±3°, 높이 ±2 cm, 지연 +30 ms | 어느 조건에서 무너지는지 표 |
| 7 | 행 배치 개선 (구현 필요) | 행을 지평선 근처로 몰기 (열의 정면 ±15° 처럼) → 재학습 | 5·6 결과가 나아지는지 |
| 8 | 시뮬 보강 (구현 필요) | 덕트 사이 틈, 가장자리 가짜 점 | 학습이 여전히 되는지 |

## 7. 카메라 도착 후

1. **정지 측정**: 덕트를 1·2·3·5·7 m 에 두고 depth 기록 → 거리별 σ, 구멍 비율. 실제 덕트 재질로.
2. **주행 녹화**: 카메라 + LiDAR 같이 달고 수동으로 몇 바퀴, rosbag (`/camera/depth/*`, `/scan`, `/imu/data`, `/vehicle/speed_mps`).
3. **비교 스크립트 (구현 필요)**: bag → `DepthEyeReal` 가까움 영상 vs 같은 시각 LiDAR 로 만든 가까움 → 칸별 오차·구멍 지도 → `CameraSpec` 값 맞춤.
4. 실제 vs 보정된 시뮬 영상 GIF 나란히 → 재학습 → 실차 `max_speed:=2.0`.
