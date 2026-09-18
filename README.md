# robo_testr

초파리 **hemibrain 커넥톰** 배선을 prior로 쓴 RNN + **PPO**로  
F1TENTH / RoboRacer 스타일 **mapless LiDAR 주행**을 학습하는 레포입니다.

실차 기준 스택은 `Roboracer-2026-main/` (Ajou Jetson ROS2) 이고,  
여기서는 그 **입출력 규약·맵·조향/속도 한계**에 맞춰 PC에서 학습·시각화합니다.

---

## 한 줄 요약

| 질문 | 답 |
|------|----|
| 모방학습이냐? | **아니요. PPO 강화학습** (데모 궤적 없음) |
| mapless냐? | **관측은 mapless** (LiDAR+속도만). 맵/CSV는 에이전트 입력에 안 넣음 |
| 센터라인은? | **한 바퀴(랩) 학습·채점용 privileged 신호** (관측 X) |
| 커넥톰 vs 일반 RL? | 학습 알고리즘은 같고, **신경망 구조 prior**만 다름 |
| 실차 코드 역할? | 인터페이스·맵·Stanley 상한 레퍼런스. 공정 비교 본체는 **커넥톰 vs MLP** |

---

## 배경 / 왜 이렇게 짜였나

1. **Physical AI / 레이싱:** 랩타임·충돌 없는 주행이 목표. 바이럴 “초파리로 게임” 데모와 다르게 **보상 고정 ablation**이 필요함.
2. **커넥톰 prior:** hemibrain에서 속도/시야 관련 서브서킷 배선을 고정하고, 시냅스 세기·읽기 가중치를 PPO로 학습 (`ConnectomeRNN`).
3. **실차 정렬:** Ajou `Roboracer-2026-main`의 `/scan` → `/drive`(Ackermann) 드롭인 자리를 목표로 함. Stanley CSV 추종을 **정책으로 교체**하는 그림.
4. **공정 비교:** 같은 맵·같은 보상·같은 timesteps에서 커넥톰 vs MLP. Stanley는 별도 상한 레퍼런스.

---

## 전체 파이프라인

```
[데이터]
  neuPrint hemibrain → download_hemibrain.py → hemibrain_cache/
  Roboracer Cartographer 맵 (yaml+png) + centerline CSV

[학습 PC — Windows OK]
  lidar_race_env (occupancy + 레이캐스트 LiDAR)
       ↓ obs: LiDAR20 + v_norm + yaw_rate
  ConnectomeRNN 또는 MLP
       ↓ action: [steer_norm, speed_norm] → 조향각, 2~7 m/s
  PPO (stable-baselines3)
       ↓
  connectome_ppo_lidar_ajou.zip

[실차 Jetson — Ubuntu/ROS2]
  /scan + speed → roboracer_connectome_node.py → /drive
  (control_node → VESC 조향은 기존 스택 유지)
```

### 실차 고전 스택 vs 우리

```
[Roboracer-2026]                         [본 레포]
/scan → FGM/회피 → local path
      → Stanley(CSV 웨이포인트) → /drive      /scan → 정책 → /drive
      → control_node → VESC                   (동일 control 가정)
```

자세한 실차 문서:
- `Roboracer-2026-main/src/race_pkg/ARCHITECTURE.md`
- `Roboracer-2026-main/src/path_following/README.md`

---

## 현재 상태 (체크리스트)

| 항목 | 상태 | 비고 |
|------|------|------|
| hemibrain 캐시 | ✅ | `hemibrain_cache/` (~2995 뉴런 원본 → 서브서킷 ~800) |
| 커넥톰 RNN + PPO | ✅ | `connectome_rnn.py`, `train_connectome.py` |
| 실차 맵 `ajou` | ✅ | `Roboracer-2026-main/maps/cartographer_map_20260817_003202.*` |
| LiDAR occupancy 시뮬 | ✅ | **본선** `--env lidar` |
| 속도 2~7 m/s | ✅ | 최저 2, 최고 7 (정지 없음) |
| 조향 ±0.3735 rad | ✅ | 실측 전륜 ±21.4° |
| bicycle 역학 | ✅ | `R = L/tan(δ)` (예전 식은 코너 불가 버그) |
| 학습 resume | ✅ | `--resume` / 기존 zip 자동 이어학습 |
| watch GIF (초 단위) | ✅ | `watch_lidar_map.py` |
| ROS2 드롭인 노드 | 🟡 | `roboracer_connectome_node.py` 스케치 (Jetson에서) |
| 한 바퀴 안정 완주 | 🟡 | 학습 진행 중 / 추가 step 필요 |
| 커넥톰 vs MLP 벤치 | ⬜ | 동일 설정으로 `--baseline` 돌릴 것 |

---

## 설치

### PC 학습 (Windows / Ubuntu 공통)

```powershell
cd c:\Users\user\Downloads\robo_testr
pip install -r requirements.txt
```

주요 패키지: `torch`, `gymnasium`, `stable-baselines3`, `numpy`, `pandas`, `pyyaml`, `pillow`, `matplotlib`, `neuprint-python`

### 초파리 데이터 (최초 1회)

[neuPrint](https://neuprint.janelia.org) 토큰 발급 후:

```powershell
$env:NEUPRINT_APPLICATION_CREDENTIALS = "여기에_토큰"
python download_hemibrain.py
```

이후에는 `--use-cache`로 API 없이 학습.

> 토큰을 채팅/깃에 올리지 말 것.

### 실차 (Ubuntu + ROS2 + Jetson)

- `Roboracer-2026-main/` 워크스페이스 빌드 (기존 팀 절차)
- 학습된 `*.zip`을 Jetson으로 복사
- `roboracer_connectome_node.py`를 Stanley 노드 대신 실행 (패키지화는 추가 작업)

**학습은 Windows로 충분. ROS2 실연동만 Ubuntu가 필수.**

---

## 빠른 시작 (명령)

### 학습 — 커넥톰 (본선)

```powershell
# 처음부터
python train_connectome.py --use-cache --env lidar --map ajou --timesteps 200000

# 이어서 (기존 zip 있으면 자동 resume, 또는 명시)
python train_connectome.py --use-cache --env lidar --map ajou --timesteps 300000 --resume connectome_ppo_lidar_ajou.zip
```

저장 파일: `connectome_ppo_lidar_ajou.zip`

### 학습 — MLP 베이스라인 (공정 비교)

```powershell
python train_connectome.py --baseline --env lidar --map ajou --timesteps 200000
```

저장 파일: `baseline_mlp_lidar_ajou.zip`

### 시각화

```powershell
# 휴리스틱 wall-follow (모델 없음)
python watch_lidar_map.py --map ajou --max-sec 30

# 학습 모델
python watch_lidar_map.py --map ajou --model connectome_ppo_lidar_ajou.zip --use-cache --max-sec 40
```

출력: `watch_out_race/lidar_ajou_model.gif`  
제목의 `t=` 는 **초(s)**, `v=` 는 m/s.

### 맵 선택

| `--map` | 경로 |
|---------|------|
| `ajou` (기본) | `Roboracer-2026-main/maps/cartographer_map_20260817_003202.yaml` |
| `ajou_prev` | `...002900.yaml` |
| `Spielberg` 등 | `f1tenth_racetracks/<Name>/` |
| `vegas` | `maps/vegas.yaml` |
| `경로/foo.yaml` | 직접 지정 |

센터라인(채점/랩 보상): 맵 옆 `{stem}_centerline.csv` 우선, 없으면  
`Roboracer-2026-main/src/path_following/config/centerline.csv`.

센터라인 재추출 (실차 스크립트):

```text
Roboracer-2026-main/src/path_following/scripts/extract_centerline_from_map.py
```

---

## 환경 명세 (`lidar_race_env.py`)

### 관측 (mapless) — 차원 22

| 성분 | 설명 |
|------|------|
| LiDAR 20빔 | FOV ≈ 270°, 최대 8 m, occupancy 레이캐스트, `[0,1]` 정규화 |
| `v_norm` | `(v - 2) / (7 - 2)` ∈ `[0,1]` |
| `yaw_rate` | `clip(yaw_rate/3, -1, 1)` |

**에이전트는 맵 좌표·센터라인·웨이포인트를 관측으로 받지 않음.**

### 행동 → 실차 `/drive`

| action | 의미 | 물리량 |
|--------|------|--------|
| `[0]` ∈ `[-1,1]` | 조향 | `steer = action[0] * 0.3735` rad |
| `[1]` ∈ `[0,1]` | 속도 명령 | `speed = 2 + action[1] * 5` → **2~7 m/s** |

### 동역학

- 샘플 시간 `DT = 0.05 s` (20 Hz)
- 휠베이스 `L = 0.33 m`
- bicycle:  
  `x += v cosθ Δt`, `y += v sinθ Δt`,  
  `θ += (v/L) tan(δ) Δt`  
  → 최대 조향에서 최소 회전반경 **R ≈ L/tan(δ) ≈ 0.84 m**  
  (예전에 쓰던 `0.5*tan` 모델은 R≈1.72 m라 좁은 코너 불가였음 → 수정됨)
- 속도는 1차 추종 후 항상 `[2, 7]` 클램프 (정지 없음)

### 충돌

- occupancy에서 차체 inflate ≈ 0.14 m
- 맵: ROS trinary, 어두운 픽셀·unknown ≈ 벽, 밝은 픽셀 = free

### 보상 (현재: 한 바퀴 목표)

**관측은 mapless.** 보상만 랩 진행을 위해 센터라인을 privileged로 사용.

| 항 | 대략 | 역할 |
|----|------|------|
| `+2.5 * max(Δcenterline, 0)` | 진행 | 한 바퀴의 핵심 |
| `+1.0 * v_norm` | 속도 | 2 m/s 고착 완화 |
| `+0.6 * mid + 0.4 * front` | LiDAR 여유 | 벽 회피 |
| 앞 뚫림 + 거의 최저속 | −0.4 | 저속 고착 페널티 |
| 충돌 | −30 | 종료 |
| 랩 완주 | `+ (200 - lap_time)` | 빠른 완주 보너스 |

- 에피소드 종료: **충돌 또는 랩 완주** (또는 `MAX_STEPS=8000`)
- `info["progress"]`, `info["lap_time"]` 로 모니터링
- `reward_mode`: `mapless_obs_lap_reward`

> 보상 스케일을 바꾸면 `ep_rew_mean` 숫자는 이전 run과 **비교 불가**.  
> 볼 것: `ep_len`, `progress`, 완주 횟수(`LapTimeLogger`).

### 모방학습이 아닌 이유

- 데모/Stanley 궤적을 따라 학습하지 않음
- 시행착오 + 보상으로 정책 갱신 (PPO)
- 센터라인은 “이 조향을 따라라” 라벨이 아니라 **진행도 줄자**

---

## 커넥톰 모델

1. `hemibrain_cache` 로드 (또는 합성 T4/T5→CX→DN)
2. `select_speed_relevant_subcircuit` 으로 운동/시야 관련 ~800 뉴런
3. 인접행렬 고정 → `ConnectomeRNN` (토폴로지 고정, 학습 가능 가중치)
4. SB3 `MlpPolicy`의 **features extractor**로 장착, `net_arch=[]`
5. PPO: `n_steps=512`, `batch_size=256`, `lr=3e-4`, `gamma=0.99`, 기본 `n_envs=4`

MLP 베이스라인(`--baseline`): 동일 env, `net_arch=[64,64]`, 커넥톰 없음.

---

## 학습 로그 읽는 법

| 지표 | 의미 |
|------|------|
| `total_timesteps` | 누적 환경 step (`reset_num_timesteps=False`면 resume 시 이어서 증가) |
| `ep_len_mean` | 평균 에피소드 길이 (×0.05 ≈ 초). 충돌 전 생존 |
| `ep_rew_mean` | 평균 에피소드 보상 (**보상식 바뀌면 절대값 비교 금지**) |
| `explained_variance` | value 망 적합도 (0→1로 오르면 학습 안정 신호) |
| `std` | 정책 탐험 폭 (학습되며 보통 감소) |
| `[LapTimeLogger] 누적 완주` | 한 바퀴 성공 횟수 |

시각화에서 “순간이동”처럼 보이던 이유 (수정됨):
- 예전 GIF가 step을 많이 건너뛰고 재생이 빨랐음
- 카메라 `pad=±20 m`가 맵(≈23×12 m)보다 커서 흰 여백만 큼 → **전체 맵 고정 뷰**로 변경
- `t` 표시를 **초**로 변경

---

## 파일 지도

### 학습 / 시뮬 (루트)

| 경로 | 역할 |
|------|------|
| `lidar_race_env.py` | **본선 Gym 환경** |
| `train_connectome.py` | 커넥톰/MLP 학습, resume |
| `watch_lidar_map.py` | LiDAR+궤적 GIF |
| `roboracer_connectome_node.py` | 실차 `/scan`→`/drive` 스케치 |
| `connectome_loader.py` | hemibrain 로드·서브서킷·인접행렬 |
| `connectome_rnn.py` | 커넥톰 제약 RNN |
| `download_hemibrain.py` | neuPrint 다운로드 |
| `watch_agent.py` | watch용 extractor 헬퍼 |
| `hemibrain_cache/` | 뇌 캐시 |
| `maps/` | vegas 등 보조 맵 |
| `f1tenth_racetracks/` | 공개 레이스 트랙 |
| `multitrack_env.py` | 절차적 트랙 (디버그, 본선 아님) |
| `requirements.txt` | pip 의존성 |
| `connectome_ppo_lidar_ajou.zip` | 학습된 커넥톰 정책 |
| `watch_out_race/` | GIF 출력 |

### 실차 (`Roboracer-2026-main/`)

| 경로 | 역할 |
|------|------|
| `maps/` | Cartographer 맵 + (복사된) centerline CSV |
| `src/path_following/` | Stanley, control, FGM, AEB 등 |
| `src/path_following/config/centerline.csv` | 실차 경로 CSV |
| `src/race_pkg/` | 레이스 아키텍처·기획 |
| `src/localization_layer/` | Cartographer 등 |
| `src/sllidar_ros2/` | LiDAR 드라이버 |

---

## 권장 실험 순서

1. **스모크:** `python -c "from lidar_race_env import LidarRaceEnv; ..."` / `watch_lidar_map.py --map ajou`
2. **커넥톰 학습** `ajou` 20만+ (필요 시 resume으로 추가)
3. **동일 설정 MLP** `--baseline`
4. **지표 비교:** 완주율, 랩타임 분포, (가능하면) LiDAR 노이즈 강건성
5. **Jetson:** zip 배포 → connectome 노드로 `/drive` 교체 (안전 모드·저속부터)

---

## 한계 / 알려진 이슈

- `ajou` Cartographer 맵은 **실내 소형 루프** → 파이프라인·랩 학습용. 고속 레이스 벤치는 Spielberg 등 병행 권장.
- 트랙 폭 ≈ 1 m대, 최저 2 m/s + 조향 한계면 헤어핀은 여전히 어려움.
- mapless 관측만으로 “한 바퀴”는 보상 설계가 까다로움 → 현재는 **진행 privileged 보상**으로 완주를 유도.
- CPU 학습 시 20만 step ≈ 수십 분~1시간+ 수준 (환경·코어에 따라 다름).
- 예전에 학습한 zip은 **속도 상한·역학·보상이 달랐으면** 새 설정과 호환되지 않음. resume은 **같은 관측/행동 차원**일 때만.
- ROS2 노드는 프로덕션 패키지화·안전 interlock 미완.

---

## FAQ

**Q. 왜 속도가 2 m/s에 붙어 있나?**  
A. `action[1]=0` → 2 m/s. 초반 정책이 생존만 익히면 최저속에 고착하기 쉬움. 속도·진행 보상과 추가 학습으로 완화하는 중.

**Q. `ep_rew`가 갑자기 달라졌는데 망한 건가?**  
A. 보상 식을 바꾸면 스케일이 달라짐. `ep_len` / 완주 로그를 볼 것.

**Q. 우분투에서만 되나?**  
A. **학습·watch는 Windows OK.** 실차 ROS2만 Ubuntu/Jetson.

**Q. Stanley보다 빠른가?**  
A. 아직 보장 없음. Stanley는 레퍼런스. 주 비교는 **같은 보상 커넥톰 vs MLP**.

---

## 참고 링크

- [neuPrint hemibrain](https://neuprint.janelia.org)
- [f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks)
- 실차: `Roboracer-2026-main/`
