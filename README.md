# robo_testr

초파리 **hemibrain 커넥톰**을 **시간축 temporal memory**로 학습하는  
**SAC** 기반 F1TENTH **mapless** 레이싱 레포.  
학습·추론은 **GPU(CUDA / Jetson)** 우선, CPU는 폴백.

| 구분 | 본선 | Ablation / 레거시 |
|------|------|-------------------|
| 알고리즘 | **SAC** + ConnectomeRNN (학습) | plain MLP SAC / PPO |
| Env | `f1tenth_mapless_env.py` | `lidar_race_env.py` (ajou PPO) |
| 관측 | LiDAR 135×5 + yaw×5 → **680** | dim 82 (레거시) |
| LiDAR 시뮬 | **40 m / 40 Hz** (실차), 제어 frame_skip=4 → **~10 Hz** | — |
| 보상 | privileged CL progress (관측은 mapless) | mapless LiDAR+odom |
| 디바이스 | `--device auto\|cuda` (Jetson GPU) | CPU 가능 |
| 체크포인트 | `EvalCallback`(deterministic) + `best_model` + periodic ckpt | — |

```bash
# GPU 있으면 자동 cuda (Jetson / PC)
python train_sac_connectome.py --use-cache --map Spielberg --timesteps 200000 --fresh \
  --device cuda --max-neurons 256 --batch-size 512

python watch_sac_f1tenth.py --model connectome_sac_f1tenth_Spielberg.zip --map Spielberg --use-cache
```

실차: `Roboracer-2026-main/` (Jetson ROS2).

---

## 파이프라인 한눈에 (그림)

<p align="center">
  <img src="docs/figures/arch_pipeline.png" alt="전체 파이프라인" width="900"/>
</p>

**흐름:** LiDAR·IMU → 인코더 → ConnectomeRNN(시간 메모리) ‖ skip → Fuse → SAC → 조향·속도  
배포: `/scan`(+yaw) → 같은 정책 → `/drive` (맵/센터라인 불필요)

---

## 1. 한 줄 요약

| 질문 | 답 |
|------|----|
| 무엇을 풀나? | mapless LiDAR로 F1TENTH 트랙 주행 |
| 왜 SAC? | 연속 행동 + off-policy (GPU 배치 학습과 잘 맞음) |
| 커넥톰? | LSTM 역할의 **고정 배선 RNN 메모리** (scale/\(W_{in}\) 학습) |
| 인코더? | 원시 센서를 짧은 벡터로 압축 (아래 개념 절) |
| Jetson? | `device=cuda`, dense matmul, 배치 인코딩으로 **GPU 비중↑** |

### 타이밍 (실차 정렬 · 학습 안정)

| | 값 | 이유 |
|--|-----|------|
| LiDAR 주기 | **40 Hz** (`DT=0.025`) | 실차 측정 주기 |
| 거리 상한 | **40 m** | 실차 range |
| 레이 샘플 | 맵 `resolution`(~5 cm) | 얇은 벽 관통 방지 |
| `frame_skip` | **4** → 제어 **~10 Hz** | 50→10 Hz로 행동 차이·히스토리 정보량 확보 |
| 히스토리 5프레임 | span ≈ **0.4 s** | skip 간격으로 스택 (예전 0.08 s) |
| 스폰 | CL **전체** + 헤딩/횡 노이즈 | 직선만 버퍼에 쌓이는 것 방지 |
| 랩 | `truncate` + 보너스 30 (terminate 아님) | Q 절벽 완화 |
| 보상 v4 | alive +0.2 / crash −10 / progress×4 | 짧은 충돌이 “best”로 잡히던 붕괴 방지 |
| 학습 저장 | `runs/*/best`(길이 우선), `checkpoints` | EvalCallback length-aware |

A/B: **먼저** `train_sac_plain.py` (Eval deterministic). plain도 무너지면 env/SAC, plain만 안정이면 커넥톰.

---

## 학습 결과 (Spielberg · 실차 정렬 A/B)

실차 스펙 env (`40 m` LiDAR / `40 Hz` sim / 제어 `~10 Hz`)에서  
**plain MLP SAC 150k** 완료 + **connectome SAC ~80k** 저장 후 deterministic watch.

| 모델 | steps | 대표 seed0 랩 | 비고 |
|------|------:|---------------|------|
| plain MLP SAC | 150k | **101.7 s** 완주 | `plain_sac_f1tenth_Spielberg.zip` |
| connectome SAC | ~80k | **98.4 s** 완주 | `connectome_sac_f1tenth_Spielberg_80k.zip` (best도 랩 가능) |

<p align="center">
  <img src="docs/figures/sac_ab_eval_curves.png" alt="SAC A/B eval curves" width="900"/>
</p>

Deterministic eval: 두 모델 모두 ~10k 이후 **랩 완주 구간**(ep_len ≈ 1000–1500 ≈ 100–150 s)에 진입.  
connectome은 초반 상승이 더 가파름(10k에서 이미 랩). plain은 150k까지 안정적으로 유지.

| | plain last (seed0) | connectome 80k (seed0) |
|--|--------------------|------------------------|
| GIF | ![](docs/figures/sac_plain_Spielberg.gif) | ![](docs/figures/sac_connectome_Spielberg.gif) |
| traj | ![](docs/figures/sac_plain_Spielberg_traj.png) | ![](docs/figures/sac_connectome_Spielberg_traj.png) |

```powershell
# 결과 재현 (GIF + traj PNG)
python watch_sac_f1tenth.py --model plain_sac_f1tenth_Spielberg.zip --map Spielberg --plain --seed 0
python watch_sac_f1tenth.py --model connectome_sac_f1tenth_Spielberg_80k.zip --map Spielberg --use-cache --seed 0
python plot_sac_results.py   # → docs/figures/sac_ab_eval_curves.png
```

체크포인트: `runs/plain_Spielberg/`, `runs/connectome_Spielberg/` (gitignore). zip은 로컬 보관.

### 왜 7 m/s까지 안 올라가냐?

속도는 정책이 “마음대로” 올리는 게 아니라 **환경 상한**에 클립됩니다.

```text
v_cmd = min_speed + action_speed * (max_speed - min_speed)
```

| 단계 | 물리 | 속도 밴드 | 상태 |
|------|------|-----------|------|
| **1 (현재 데모 zip)** | kinematic | `[2, 3.5]` | `*_80k` / `*_v35` — 랩·GIF·레이어 viz |
| **2 (진행 중)** | **Single-Track** (슬립/μ) | `[0.5, 7.0]` | plain ST 재학습 → 이후 connectome |

지금까지 1단계는 kinematic + `--max-speed 3.5`로 랩을 확보했습니다.  
**현재 코드 기본은 Single-Track 동역학(타이어 슬립/μ, f1tenth_gym 계열) + 최고속도 7.0 m/s**입니다.  
`min_speed`는 코너 감속을 위해 **0.5**로 내렸습니다.

천장만 7로 연다고 바로 7 m/s가 나오지는 않습니다.

1. **액션 스케일:** `speed_u=1`이어도 env `max_speed`가 3.5면 **물리적으로 3.5가 끝**.  
2. **보상:** ST 모드(`privileged_progress_st_v1`)는 진행 \(\Delta s\) 위주 + 조향 급변·슬립각·횡가속 초과·CTE 패널티.  
   고속에서 미끄러지면 패널티 → 정책이 풀스로틀을 늦게 배움.  
3. **역주행 terminate:** 헤딩이 CL과 크게 어긋나면 즉시 종료 → 초반 `ep_len`이 짧을 수 있음.  
4. **커리큘럼:** 3.5에서 안정 랩 → zip 백업 → `[0.5,7]` ST로 fresh/이어학습이 안전.

```powershell
python train_sac_plain.py --map Spielberg --timesteps 150000 --fresh `
  --min-speed 0.5 --max-speed 7.0 --max-steer 0.30

# kinematic 전용 zip은 *_v35.bak.zip 등으로 남겨 두세요.
```

### 제로샷 전이 · 실차 맵 `ajou`

`--map ajou` → `Roboracer-2026-main/maps/cartographer_map_20260817_003202`.  
Spielberg 학습 zip을 **재학습 없이** ajou에 올리면 (seed 0–5) plain/connectome 모두 **랩 완주·무충돌** (랩 ~15–23 s, 코스가 짧음).

```powershell
python watch_sac_f1tenth.py --model plain_sac_f1tenth_Spielberg.zip --map ajou --plain --seed 0
python watch_sac_f1tenth.py --model connectome_sac_f1tenth_Spielberg_80k.zip --map ajou --use-cache --seed 0
```

---

## 2. 모듈 개념 설명 (초심자용)

### 2.1 왜 “인코더”가 필요한가?

LiDAR 한 프레임은 **135개 거리값**. 매 스텝 이걸 그대로 거대한 MLP에 넣으면:

- 파라미터·노이즈가 많고  
- “왼쪽 벽 / 전방 갭” 같은 **국소 패턴**을 매번 처음부터 배워야 함  

**인코더** = 센서 → **의미 있는 짧은 특징 벡터** \(z\).  
CNN/MLP가 “가까운 빔끼리의 패턴”을 먼저 뽑아주면, 뒤의 커넥톰·SAC가 주행만 배우기 쉬워진다.

### 2.2 LiDAR Encoder (1D Conv)

<p align="center">
  <img src="docs/figures/arch_lidar_encoder.png" alt="LiDAR 1D Encoder" width="900"/>
</p>

| 개념 | 설명 |
|------|------|
| 입력 | 빔 배열 \(d\in\mathbb{R}^{135}\) (거리/10으로 정규화) |
| **축의 의미** | 이미지가 아니라 **각도 방향의 1D 신호**. 옆 빔 = 비슷한 시야 |
| Conv1d | 작은 커널이 빔을 따라 미끄러지며 **국소 벽·갭 패턴** 추출 |
| Pool + Linear | 길이를 줄여 \(z_{\mathrm{lidar}}\in\mathbb{R}^{48}\) |
| 구현 | `policy_sac_connectome.py` → `LidarEncoder` |
| GPU | **5프레임을 한 번에** `(B·T, 1, 135)` 배치 Conv → GPU 점유↑ |

직관: “사진용 2D CNN”의 1D 버전. 전방 중앙이 막히면 중앙 채널 활성, 한쪽 벽이면 좌/우 패턴.

### 2.3 IMU Encoder (MLP)

| 개념 | 설명 |
|------|------|
| 입력 | yaw rate \(\tilde\omega=\mathrm{clip}(\dot\theta/3,-1,1)\) (히스토리 5칸) |
| 역할 | “지금 얼마나 회전 중인지”를 짧은 벡터로 |
| 출력 | \(z_{\mathrm{imu}}\in\mathbb{R}^{16}\) |
| 합치기 | \(z_t = [z_{\mathrm{lidar}}; z_{\mathrm{imu}}]\in\mathbb{R}^{64}\) |

LiDAR만으로도 회전은 추정 가능하지만, yaw를 직접 주면 **고속 코너**에서 신호가 더 또렷하다.

### 2.4 ConnectomeRNN = 시간 메모리 (LSTM 자리)

<p align="center">
  <img src="docs/figures/arch_connectome_memory.png" alt="Connectome temporal memory" width="900"/>
</p>

| 개념 | 설명 |
|------|------|
| 왜 메모리? | 한 프레임만 보면 POMDP (속도·곡률 변화 모호) |
| 히스토리 스택 | obs에 이미 5프레임 포함 |
| **RNN unroll** | \(t=0..4\) 순서로 \(h\)를 넘기며 갱신 (LSTM 셀 대신 **초파리 배선**) |
| 고정 | \(A_{\mathrm{signed}}\) — 누가 누구와 연결되는지 |
| 학습 | \(\mathrm{scale}\) (세기), \(W_{\mathrm{in}}\) (입력 주입) |
| 출력 | DN 활성 \(y_4\) → “행동 관련 요약” |

수식 (leaky rate RNN):

\[
W_{\mathrm{eff}}=A_{\mathrm{signed}}\odot\mathrm{scale}\odot M_{\mathrm{topo}}
\]

\[
h \leftarrow h + \frac{\Delta t}{\tau}\Big(-h + \tanh(W_{\mathrm{eff}}^\top h + \mathrm{drive}(z))\Big)
\]

**LSTM과 비교**

| | LSTM | ConnectomeRNN (본 레포) |
|--|------|-------------------------|
| 구조 | 학습된 게이트 | **생물 배선 prior** |
| 새 연결 | 자유롭게 생김 | **금지** (토폴로지 고정) |
| 역할 | 시계열 압축 | 동일 + 해석 가능한 prior |

### 2.5 skip MLP · Fuse · SAC

| 모듈 | 하는 일 |
|------|---------|
| **skip** | \(z\) 평균을 MLP로 바로 보냄 — 커넥톰이 흔들려도 주행 학습 경로 유지 |
| **fuse** | `[skip; DN]` → 256차원 공통 feature |
| **SAC Actor** | feature → 조향·속도 분포 |
| **SAC Critic** | feature+action → soft Q |

SAC 목표 (최대 엔트로피):

\[
J(\pi)=\mathbb{E}\Big[\sum_t \gamma^t\big(r_t+\alpha\mathcal{H}(\pi(\cdot|s_t))\big)\Big]
\]

### 2.6 레이어·커넥톰 활성화 시각화 (무엇을 보나)

정책이 “핸들만 돌리는지”, 아니면 **인코더 → 커넥톰 메모리 → fuse**가 실제로 장면에 반응하는지  
확인하려면 중간 활성화를 찍어 보는 것이 가장 직관적입니다.

| 산출물 | 설명 |
|--------|------|
| 정적 스냅샷 | `docs/figures/policy_layers_activation.png` — 한 제어 스텝의 forward |
| **연속 GIF** | `docs/figures/policy_layers_activation.gif` — 주행 중 패널이 시간에 따라 변하는 모습 |

<p align="center">
  <img src="docs/figures/policy_layers_activation.png" alt="한 스텝 레이어 활성화" width="900"/>
</p>

<p align="center">
  <img src="docs/figures/policy_layers_activation.gif" alt="연속 레이어·커넥톰 활성화" width="900"/>
</p>

사용 모델(예시): kinematic 학습된 `connectome_sac_f1tenth_Spielberg_80k.zip`,  
속도 밴드 \(v\in[2, 3.5]\) (아래 “7 m/s” 절 참고). GIF는 센터라인에 정렬된 출발점에서  
수백 제어 스텝을 돌며 매 stride마다 패널을 렌더합니다.

#### 데이터 흐름 (한 제어 스텝 = GIF 한 프레임의 내부)

```text
Obs 680
  ├─ LiDAR 135 × 5  ──► LidarEnc (Conv1d) ──► z_L (48)
  └─ yaw × 5        ──► IMUEnc (MLP)       ──► z_I (16)
                              │
                         z_t = [z_L; z_I] ∈ R^64
                              │
              t = 0..4  ConnectomeRNN unroll (고정 A_signed, 학습 scale/W_in)
                              │
                    h_t (뉴런 상태) ,  y_DN (출구 요약)
                              │
              skip(mean_t z)  ‖  y_DN  ──► Fuse(256) ──► SAC Actor → (steer, speed)
```

`policy_sac_connectome.py`의 `forward_intermediates()`가 위 중간 텐서를 이름으로 반환하고,  
`viz_policy_layers*.py`가 그걸 그립니다.

#### GIF / PNG 패널 읽는 법

| 패널 | 보는 것 | 해석 포인트 |
|------|---------|-------------|
| **Track** | 맵 위 차·궤적 | “지금 직선인지 코너인지”의 시간축 기준 |
| **LiDAR (m)** | 최신 스캔 (각도→거리) | 전방이 열리면 중앙 peak↑, 코너·벽 접근 시 한쪽이 깎임 |
| **LiDAR hist** | 5프레임 거리 히트맵 | 장면이 시간으로 밀려오는 패턴; 직진이면 세로로 안정, 진입 시 기울어짐 |
| **Encoder \(z_t\)** | \(t=0..4\) 잠재벡터 | LiDAR+yaw 압축. **줄무늬/색이 바뀌면 장면 표현이 바뀐 것** |
| **Connectome \(\|h\|\) + DN** | 언롤 중 \(\|h_t\|\)와 DN 활성 | 커넥톰 **메모리 세기**. 코너·급변 구간에서 \(\|h\|\)·DN이 출렁이면 시간 통합이 살아 있는 증거 |
| **\(h\) top40 / fuse** | \(\|h\|\) 큰 뉴런 + \(\|\mathrm{fuse}\|\) | SAC에 들어가는 요약. fuse 노름이 장면과 같이 변하면 “중간 feature가 행동을 받치는” 상태 |

제목줄 텔레메트리: `steer`, `speed_u`(정규화 속도 명령), 실제 `v`, `slip`, `prog`.

#### 이 그림이 “뜻하는” 것 / 뜻하지 않는 것

- **뜻함:** 커넥톰은 조향 각도를 직접 뱉는 모듈이 아니라, **짧은 시간 맥락을 \(h\)에 담아 fuse로 넘기는 LSTM 자리**다.  
  LiDAR가 기울고 → \(z\)가 바뀌고 → \(\|h\|\)/DN이 변하면, 그 경로가 실제로 쓰이고 있다는 **정성 증거**다.
- **뜻하지 않음:** 특정 뉴런 = “왼쪽 벽” 같은 단일 개념 라벨은 아직 없다. top40은 **활성 크기 순위**일 뿐 해부학 이름 매핑이 아니다.
- plain MLP ablation에는 Connectome 패널이 없다 (obs→MLP→행동). A/B는 eval 곡선·랩타임으로 비교하고, 이 GIF는 **connectome 경로 해석용**이다.

#### 재생 커맨드

```powershell
# 한 스텝 스냅샷 (warmup 후 PNG)
python viz_policy_layers.py --model connectome_sac_f1tenth_Spielberg_80k.zip `
  --map Spielberg --use-cache --seed 0 --warmup 40 `
  --min-speed 2 --max-speed 3.5 --out docs/figures/policy_layers_activation.png

# 연속 활성화 GIF (길게: 제어 ~400스텝, stride 2, 프레임간격 100ms)
python viz_policy_layers_gif.py --model connectome_sac_f1tenth_Spielberg_80k.zip `
  --map Spielberg --use-cache --seed 0 `
  --max-steps 400 --stride 2 --duration-ms 100 --max-frames 250 `
  --min-speed 2 --max-speed 3.5 --physics kinematic --start-idx 10 `
  --out docs/figures/policy_layers_activation.gif
```

참고: 랜덤 스폰이 센터라인과 **헤딩이 반대**이면 `reversed` terminate가 바로 걸릴 수 있다.  
GIF 스크립트는 `--start-idx`로 센터라인에 정렬된 pose를 넣어 긴 구간을 뽑는다.  
80k zip은 **kinematic** 학습본이므로 `--physics kinematic`을 맞춘다 (기본 env는 ST).

---

## 3. GPU / Jetson 설계

나중에 **Jetson에 올려 GPU를 쓰게** 맞춘 포인트:

| 항목 | CPU | GPU (CUDA / Jetson) |
|------|-----|---------------------|
| Connectome matmul | **sparse** (희소 연결) | **dense** \(h W^\top\) — Tensor 코어에 유리 |
| LiDAR 인코딩 | 프레임 루프도 가능 | **B×T 배치 Conv1d** 한 방 |
| SAC batch | 256 | 기본 **512** (`--batch-size`) |
| cuDNN | — | `cudnn.benchmark=True` |
| 디바이스 | `--device cpu` | `--device cuda` 또는 `auto` |

```bash
# Jetson / CUDA PC
python train_sac_connectome.py --use-cache --map Spielberg --device cuda \
  --max-neurons 256 --batch-size 512 --n-envs 4 --timesteps 300000 --fresh

# 추론도 동일 zip, device=cuda
```

**주의:** 시뮬 env(레이캐스트)는 아직 CPU. **정책 forward/backward만 GPU**.  
Jetson에서는 env step이 짧고 배치 추론이 잦을수록 GPU 비율이 커진다.  
추후: TensorRT / `torch.jit` / ONNX로 추론만 더 가속 가능.

---

## 4. 이론 요약 (MDP · 보상)

### MDP

| 기호 | 내용 |
|------|------|
| \(s\) | LiDAR hist + yaw hist (맵 없음) |
| \(a\) | \([a^{\mathrm{steer}}, a^{\mathrm{speed}}]\) |
| \(r\) | 시뮬: CL progress (privileged) |
| \(\gamma\) | 0.99 |

### 관측 레이아웃 (680)

```text
[ LiDAR t-4 … t ][ yaw × 5 ]
     675              5
```

### 보상 (privileged · v4)

\[
\begin{aligned}
r &= 4\max(\Delta s,0) + 0.2 + 0.15\max(\cos\phi,0)
    + 0.25\,v_{\mathrm{norm}}\max(\cos\phi,0)\\
    &\quad - 0.1\max(\mathrm{CTE}-0.7,0) - 0.4\max(-\Delta s,0)\\
r &\leftarrow \mathrm{clip}(r,-2,10)
\end{aligned}
\]

충돌 \(-10\), 랩 \(+30\) (truncate). alive 항으로 **긴 주행 > 짧은 충돌**.  
**실차 추론에는 센터라인 불필요** (관측 mapless).

### 제어 · 동역학

\[
\delta=a_0\delta_{\max},\quad
v^{\mathrm{cmd}}=v_{\min}+a_1(v_{\max}-v_{\min})
\]

기본 커리큘럼 \(v\in[2,3.5]\), \(\delta_{\max}=0.30\), 물리 \(\Delta t=0.025\) (40 Hz),  
제어 `frame_skip=4` → \(\Delta t_{\mathrm{ctrl}}=0.1\) (~10 Hz), \(L=0.33\).

---

## 5. 정책 의사코드

```text
# GPU: LidarEnc/IMUEnc on all frames batched
Z = EncodeAllFrames(lidar[B,T,135], yaw[B,T])   # → z[B,T,64]

h = 0
for t in 0..4:
    y, h = ConnectomeRNN(z[:,t], h)             # CUDA: dense W

feat = Fuse( Skip(mean_t z) , y_DN )
a ~ Actor(feat)                                 # SAC
```

| 모듈 | 학습 |
|------|------|
| LidarEnc, IMUEnc | ✅ |
| \(A_{\mathrm{signed}}\) | ❌ 고정 |
| scale, \(W_{\mathrm{in}}\) | ✅ |
| skip, fuse, π, Q | ✅ |

`--freeze-connectome`: 커넥톰 미학습 ablation.

---

## 6. 학습 커맨드 · 하이퍼

| 항목 | 기본 |
|------|------|
| `max_neurons` | 256 |
| `device` | `auto` → cuda 우선 |
| `batch_size` | CUDA 512 / CPU 256 |
| `train_freq` / `grad_steps` | 4 / 4 |
| `learning_starts` | 3000 |
| `target_entropy` | **-2.0** (행동 dim=2) |

```powershell
python train_sac_connectome.py --use-cache --map Spielberg --timesteps 150000 --fresh `
  --device auto --max-neurons 256 --min-speed 2 --max-speed 3.5 --max-steer 0.30

python train_sac_plain.py --map Spielberg --timesteps 150000 --fresh `
  --min-speed 2 --max-speed 3.5 --max-steer 0.30
```

| 로그 | 의미 |
|------|------|
| `ep_len_mean` | 생존 (×0.1≈초 @10 Hz) **1순위** |
| `eval/mean_ep_length` | deterministic 길이 (best 저장 기준) |
| `ep_rew_mean` | 보상 합 (식 바꾸면 비교 금지) |
| `fps` | CPU env 병목 시 낮을 수 있음 |
| `[lap]` | 완주 |

---

## 7. 파일 지도

| 경로 | 역할 |
|------|------|
| `docs/figures/*.png` / `*.gif` | 구조·인코더·메모리·**레이어 활성화** 그림 |
| `f1tenth_mapless_env.py` | Gym env (ST / kinematic, 40 m·40 Hz) |
| `vehicle_dynamics.py` | Single-Track RK4 · 슬립/μ |
| `policy_sac_connectome.py` | 인코더 + temporal connectome + fuse (+ `forward_intermediates`) |
| `connectome_rnn.py` | leaky RNN (**CUDA dense / CPU sparse**) |
| `connectome_loader.py` | hemibrain 서브서킷 |
| `train_sac_connectome.py` | SAC (`--device`) |
| `train_sac_plain.py` | MLP ablation |
| `watch_sac_f1tenth.py` | 주행 GIF + traj PNG |
| `viz_policy_layers.py` | 한 스텝 레이어/커넥톰 활성화 PNG |
| `viz_policy_layers_gif.py` | 연속 활성화 GIF |
| `plot_sac_results.py` | plain/connectome eval 곡선 |
| `train_callbacks.py` | length-aware Eval + ckpt |
| `roboracer_connectome_node.py` | Jetson `/drive` 스케치 |
| `maps/ifac_roboracer*` | IFAC 트랙 맵·센터라인 (전이 실험) |

---

## 8. 설치 · Jetson 메모

```powershell
pip install -r requirements.txt
# CUDA wheel: 환경에 맞는 torch 설치 (Jetson은 NVIDIA 문서의 PyTorch wheel)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

hemibrain: `download_hemibrain.py` 후 `--use-cache`. 토큰 깃 금지.

실차:

```text
Roboracer: /scan → Stanley → /drive
본 레포:   /scan(+yaw) → 정책(cuda) → /drive
```

---

## 9. 권장 순서

1. Env 스모크 `python f1tenth_mapless_env.py`  
2. plain SAC로 `ep_len` 상승 확인  
3. connectome SAC `--device cuda` (가능하면)  
4. watch GIF + **레이어 활성화 GIF** (`viz_policy_layers_gif.py`)  
5. 속도 커리큘럼 3.5→7 (ST)  
6. Jetson zip + ROS2 저속  

---

## 10. 한계 · FAQ

- 5프레임 메모리 ≠ 에피소드 전체 LSTM 상태  
- env 물리/레이캐스트는 CPU — 정책만 GPU  
- `fly-brain`(전뇌 LIF)은 참고용; 본선 RL 루프와는 별개  
- zip / cache / watch_out gitignore  
- 레이어 GIF의 top-k 뉴런 ≠ 해부학 라벨; **정성 해석용**  

**Q. 인코더 없이 안 되나요?**  
A. 가능하지만(plain MLP) 빔 패턴을 정책이 통째로 배워야 해서 샘플이 더 필요합니다.

**Q. Jetson에서 CPU만 쓰이면?**  
A. `torch.cuda.is_available()` 확인, `--device cuda`, Jetson용 PyTorch 휠 설치.

**Q. 커넥톰이 LSTM인가요?**  
A. 역할은 같고, 구조는 **고정 초파리 배선 + 학습 scale**입니다.

**Q. 활성화 GIF에서 속도가 3.x에서 멈추는데?**  
A. 해당 zip/커맨드가 `--max-speed 3.5`로 묶여 있기 때문입니다. 7 m/s는 ST 재학습 zip이 필요합니다 (위 절).

---

## 참고

- https://github.com/tkddn647-ship-it/2027_F1tenth_test  
- [neuPrint hemibrain](https://neuprint.janelia.org)  
- [f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks)  
- Soft Actor-Critic (Haarnoja et al.)  
- FLYNN-style connectome-constrained RNN  
