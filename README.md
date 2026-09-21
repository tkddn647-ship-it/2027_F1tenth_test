# robo_testr

초파리 **hemibrain 커넥톰**을 **시간축 temporal memory**로 학습하는  
**SAC** 기반 F1TENTH **mapless 관측** 레이싱 레포.  
학습·추론은 **GPU(CUDA / Jetson)** 우선, CPU는 폴백.

> **Mapless란?** 정책 입력에는 **맵·센터라인·(x,y)가 없다**.  
> 관측 = LiDAR hist + yaw만. 센터라인은 **학습 보상/랩 채점용 privileged 신호**일 뿐이며  
> 실차 추론(` /scan` → 정책 → `/drive`)에는 불필요하다.

| 구분 | 본선 | Ablation / 레거시 |
|------|------|-------------------|
| 알고리즘 | **SAC** + ConnectomeRNN | plain MLP SAC / PPO |
| Env | `f1tenth_mapless_env.py` | `lidar_race_env.py` (ajou PPO) |
| 관측 | LiDAR 135×5 + yaw×5 → **680** (**mapless**) | dim 82 (레거시) |
| 보상 | privileged CL progress (**관측과 분리**) | — |
| 물리 | **Single-Track** (슬립/μ, Roboracer 제원) | kinematic (1단계 데모) |
| LiDAR 시뮬 | **40 m / 40 Hz**, 제어 ~**10 Hz** | — |
| 속도 | 기본 **`[2, 7]` m/s** | kinematic 데모 `[2, 3.5]` |
| 디바이스 | `--device auto\|cuda` | CPU 가능 |

```bash
# Physical AI plain (Roboracer ST, v∈[2,7])
python train_sac_plain.py --map Spielberg --timesteps 200000 --fresh `
  --min-speed 2 --max-speed 7 --max-steer 0.3735 --physics st --device auto

# Connectome (같은 env 기본값)
python train_sac_connectome.py --use-cache --map Spielberg --timesteps 200000 --fresh `
  --device cuda --max-neurons 256 --min-speed 2 --max-speed 7 --max-steer 0.3735 --physics st
```

실차 스택·제원: `Roboracer-2026-main/` (Jetson ROS2, `vehicle_geometry.py` 등).

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
| 무엇을 풀나? | **맵 없이** LiDAR(+yaw)만으로 F1TENTH 트랙 주행 |
| 센터라인? | **관측 ❌ / 보상·랩 채점 ✅** (privileged). 실차 불필요 |
| 왜 SAC? | 연속 행동 + off-policy |
| 커넥톰? | LSTM 자리의 **고정 배선 RNN 메모리** |
| Physical AI? | ST 슬립/μ + Roboracer 조향·\(a_{lat}\)·가속 제원 |
| Jetson? | `device=cuda`, dense matmul, 배치 인코딩 |

### 타이밍 (실차 정렬 · 학습 안정)

| | 값 | 이유 |
|--|-----|------|
| LiDAR 주기 | **40 Hz** (`DT=0.025`) | 실차 측정 주기 |
| 거리 상한 | **40 m** | 실차 range |
| 레이 샘플 | 맵 `resolution`(~5 cm) | 얇은 벽 관통 방지 |
| `frame_skip` | **4** → 제어 **~10 Hz** | 50→10 Hz로 행동 차이·히스토리 정보량 확보 |
| 히스토리 5프레임 | span ≈ **0.4 s** | skip 간격으로 스택 (예전 0.08 s) |
| 스폰 | CL 접선 헤딩 정렬 + 소량 횡노이즈 | 역헤딩 스폰·가짜 reverse 방지 |
| 랩 | `truncate` + 보너스 30 | Q 절벽 완화 |
| 보상 | ST progress + 슬립/\(a_y\)/조향급변 | Physical AI |
| 학습 저장 | `runs/*/best`(길이 우선), ckpt | EvalCallback length-aware |

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

---

## Physical AI · Roboracer 제원 · 슬립

### Mapless 관측 vs privileged 센터라인 (헷갈리면 여기)

| | 정책(관측) | 시뮬 내부 |
|--|------------|-----------|
| LiDAR 135×5 + yaw×5 | ✅ | ✅ |
| 맵 occupancy | ❌ | 레이캐스트·충돌용 |
| **센터라인** | ❌ | 진행 \(\Delta s\), 랩, 스폰 정렬, CTE |
| 슬립각 β, \(a_y\) | ❌ (직접 안 줌) | 동역학 상태 + **보상 패널티** |

실차: `/scan`(+yaw)만 있으면 된다. CL CSV는 **학습 서버에서만** 쓴다.

### Roboracer-2026-main 에서 가져온 제원

| 항목 | 값 | 출처 |
|------|-----|------|
| 휠베이스 \(L\) | **0.33 m** | `vehicle_geometry.WHEELBASE_M` |
| 차체 반폭 | **0.15 m** (충돌 inflate) | `HALF_WIDTH_M` |
| 실측 전륜각 | **±0.3735 rad (±21.4°)** | `max_steering_angle_real_rad` |
| 횡가속 한계 \(a_{lat}\) | **6.0 m/s²** | `speed_profile.VEHICLE` |
| 가속 한계 | **7.0 m/s²** | 동상 `a_accel` |
| 감속(참고) | 4.0 m/s² | 제원에만; ST에 비대칭 브레이크는 아직 약함 |
| 학습 속도 밴드 | **`[2.0, 7.0]` m/s** | 커리큘럼 |

구현: `vehicle_dynamics.STParams.roboracer()` → \(\mu \approx a_{lat}/g \approx 0.61\),  
\(C_{Sf},C_{Sr}\)는 f1tenth_gym 식별값 유지.

### 동역학에서 **고려하는** 것 (ST)

- 타이어 **슬립각 β**, 요레이트, 횡가속 \(a_y \approx v\dot\psi\)
- 노면 마찰 **μ**, 코너링 강성, CG 높이(하중 이동 근사)
- 조향각·조향각속도 한계, 종가속 한계·고속 가속 감쇠
- \(v<0.5\)면 kinematic bicycle으로 전환
- RK4 + 5 ms 서브스텝, 명령은 PID → `(accl, steer_vel)`
- 보상: 진행 + **슬립/ \(a_y\) 초과 / 조향급변 / CTE** 패널티

### 아직 **약한/없는** 것

개별 휠·서스펜션, 종방향 슬립(록), 브레이크≠가속 비대칭, 서보 지연,  
μ 맵, 모터/VESC 전류, 차체 스윕 충돌(현재 점+반폭). → Isaac Sim 후속 후보.

### 속도 커리큘럼 · 현재 학습

```text
v_cmd = min_speed + action_speed * (max_speed - min_speed)
```

| 단계 | 물리 | 속도 | 상태 |
|------|------|------|------|
| 1 | kinematic | `[2, 3.5]` | 데모 zip (`*_80k`, `*_v35.bak`) — 안정 랩 |
| **2 (현재)** | **ST + Roboracer 제원** | **`[2, 7]`** | plain `…_st27` / `runs/plain_Spielberg/` |

```powershell
python train_sac_plain.py --map Spielberg --timesteps 200000 --fresh `
  --min-speed 2 --max-speed 7 --max-steer 0.3735 --physics st `
  --save-path plain_sac_f1tenth_Spielberg_st27.zip
```

스폰은 CL **접선 방향으로 헤딩 정렬**. 역주행 terminate는 **유예+연속 프레임**으로  
가짜 reverse 학살을 줄였다.

### 전이 맵 (본인 Cartographer / IFAC)

| 맵 | alias | kinematic 3.5 zero-shot | ST 7 zero-shot |
|----|-------|-------------------------|----------------|
| ajou | `ajou` | 예전에 랩 가능 | 재평가/파인튜닝 권장 |
| IFAC | `ifac` | zero-shot 실패(벽) | 파인튜닝 필요 |

```powershell
python watch_sac_f1tenth.py --model connectome_sac_f1tenth_Spielberg_80k.zip `
  --map ajou --use-cache --seed 0 --min-speed 2 --max-speed 3.5
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

### 2.6 레이어 활성화 시각화

중간 활성화를 찍어 **정책이 장면에 반응하는지** 본다.  
모델 종류에 따라 패널이 다르다.

#### A) Connectome (kinematic 데모 zip)

| 산출물 | 파일 |
|--------|------|
| PNG | `docs/figures/policy_layers_activation.png` |
| GIF (길게) | `docs/figures/policy_layers_activation.gif` |

<p align="center">
  <img src="docs/figures/policy_layers_activation.png" alt="connectome 한 스텝" width="900"/>
</p>

<p align="center">
  <img src="docs/figures/policy_layers_activation.gif" alt="connectome 연속 활성화" width="900"/>
</p>

```text
Obs 680 → LidarEnc/IMUEnc → z_t
       → ConnectomeRNN unroll (A_signed 고정, scale/W_in 학습)
       → h_t, y_DN ‖ skip(z) → Fuse → Actor (steer, speed)
```

| 패널 | 의미 |
|------|------|
| Track / LiDAR / hist | 장면 (맵은 **표시용**; 관측은 LiDAR만) |
| Encoder \(z_t\) | 장면 압축이 시간에 따라 바뀌는지 |
| \(\|h\|\) + DN | 커넥톰 메모리·출구 활성 |
| \(h\) top40 / fuse | SAC에 들어가는 요약 |

**뜻함:** LiDAR→\(z\)→\(h\)가 장면과 같이 변하면 커넥톰 경로가 쓰이는 **정성 증거**.  
**뜻하지 않음:** top-k 뉴런 = 해부학 라벨이 아님.

#### B) Plain ST Physical AI (현재 학습 체크포인트)

커넥톰이 없는 MLP라 **Actor h1/h2 ReLU** + **slip / \(a_y\) / \(v\)** 를 본다.

| 산출물 | 파일 |
|--------|------|
| GIF | `docs/figures/policy_layers_activation_st_plain.gif` |

<p align="center">
  <img src="docs/figures/policy_layers_activation_st_plain.gif" alt="ST plain 레이어·슬립 활성화" width="900"/>
</p>

모델 예: `runs/plain_Spielberg/best/best_model.zip` (ST `[2,7]`, Roboracer 제원).  
제목줄에 `slip`, `ay`가 보이면 **슬립 동역학이 시뮬에 살아 있는 상태**다.

#### 재생 커맨드

```powershell
# Connectome kinematic 데모
python viz_policy_layers.py --model connectome_sac_f1tenth_Spielberg_80k.zip `
  --map Spielberg --use-cache --min-speed 2 --max-speed 3.5
python viz_policy_layers_gif.py --model connectome_sac_f1tenth_Spielberg_80k.zip `
  --map Spielberg --use-cache --max-steps 400 --stride 2 --physics kinematic `
  --min-speed 2 --max-speed 3.5 --out docs/figures/policy_layers_activation.gif

# Plain ST Physical AI
python viz_policy_layers_plain_gif.py `
  --model runs/plain_Spielberg/best/best_model.zip `
  --physics st --min-speed 2 --max-speed 7 --max-steer 0.3735 `
  --max-steps 400 --stride 2 --seed 7 `
  --out docs/figures/policy_layers_activation_st_plain.gif
```

GIF는 `--start-idx`로 CL에 **헤딩 정렬**된 출발을 쓴다 (관측에 CL을 넣는 것이 아님).

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

### 보상 (privileged · ST Physical AI)

관측에는 CL이 없고, 보상만 CL 진행을 쓴다 (`reward_mode=privileged_progress_st_roboracer_v2`).

대략:

\[
\begin{aligned}
r &\approx 5\max(\Delta s,0) - 1\max(-\Delta s,0)
  + 0.15\max(\Delta s,0)\,v_{\mathrm{norm}}\\
  &\quad - 0.6\max(|\beta|-0.04,0) - 0.05\max(|a_y|-0.85 a_{\mathrm{lat}},0)\\
  &\quad - 0.04\,\dot\delta_{\mathrm{cmd}} - 0.12\max(\mathrm{CTE}-0.55,0)
\end{aligned}
\]

충돌 \(-10\), 랩 \(+30\) (truncate).  
역주행은 스폰 유예 후 **연속 프레임**일 때만 terminate.

**실차 추론에는 센터라인 불필요.**

### 제어 · 동역학

\[
\delta=a_0\delta_{\max},\quad
v^{\mathrm{cmd}}=v_{\min}+a_1(v_{\max}-v_{\min})
\]

기본: \(v\in[2,7]\), \(\delta_{\max}=0.3735\) (실측), ST RK4,  
\(\Delta t=0.025\) (40 Hz), `frame_skip=4` → 제어 ~10 Hz, \(L=0.33\).

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
# Physical AI plain (권장 베이스라인)
python train_sac_plain.py --map Spielberg --timesteps 200000 --fresh `
  --device auto --min-speed 2 --max-speed 7 --max-steer 0.3735 --physics st `
  --save-path plain_sac_f1tenth_Spielberg_st27.zip

# Connectome (plain 안정 후)
python train_sac_connectome.py --use-cache --map Spielberg --timesteps 200000 --fresh `
  --device auto --max-neurons 256 --min-speed 2 --max-speed 7 --max-steer 0.3735 --physics st
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
| `docs/figures/*.png` / `*.gif` | 구조·A/B·**레이어 활성화**(connectome / ST plain) |
| `f1tenth_mapless_env.py` | Gym: mapless obs + privileged reward + ST/kinematic |
| `vehicle_dynamics.py` | ST RK4, `STParams.roboracer()` |
| `policy_sac_connectome.py` | 인코더 + connectome + `forward_intermediates` |
| `connectome_rnn.py` | leaky RNN (CUDA dense / CPU sparse) |
| `train_sac_plain.py` / `train_sac_connectome.py` | 학습 (`--physics`, `[2,7]`, steer 0.3735) |
| `watch_sac_f1tenth.py` | 주행 GIF + traj |
| `viz_policy_layers.py` / `_gif.py` | connectome 활성화 |
| `viz_policy_layers_plain_gif.py` | **ST plain** h1/h2 + slip/ay GIF |
| `Roboracer-2026-main/` | 실차 ROS2·제원·Cartographer 맵 |
| `maps/ifac_roboracer*` | IFAC 맵·센터라인 |

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
2. **plain ST** `[2,7]` — `ep_len`↑ · `[lap]` 확인  
3. ST plain 레이어 GIF (`viz_policy_layers_plain_gif.py`)  
4. connectome ST 같은 제원으로 A/B  
5. ajou / IFAC fine-tune  
6. (후속) Isaac Sim 검증 · Jetson `/drive`  

---

## 10. 한계 · FAQ

- **Mapless = 관측**. 센터라인은 보상용 privileged일 뿐.  
- 슬립은 ST β + 보상 패널티로 반영. 종슬립·서스펜션은 없음.  
- 5프레임 메모리 ≠ 에피소드 전체 LSTM  
- env 물리/레이캐스트 CPU — 정책만 GPU  
- zip / cache / `runs/` / `watch_out*` gitignore  
- 레이어 GIF top-k ≠ 해부학 라벨  

**Q. 센터라인 넣으면 mapless 아니잖아?**  
A. 관측에는 안 넣습니다. 학습 서버만 CL로 \(\Delta s\)/랩을 계산합니다. 실차는 LiDAR만.

**Q. 슬립 고려했어?**  
A. 네. ST 상태 β와 \(a_y\), 보상 패널티. GIF 제목줄 `slip`/`ay`로 확인 가능.

**Q. 커넥톰이 LSTM인가요?**  
A. 역할은 같고, 구조는 고정 초파리 배선 + 학습 scale입니다.

**Q. Isaac Sim?**  
A. 다음 스테이지로 적합. 지금은 Gym ST로 랩 안정화가 우선.

---

## 참고

- https://github.com/tkddn647-ship-it/2027_F1tenth_test  
- [neuPrint hemibrain](https://neuprint.janelia.org)  
- [f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks)  
- Soft Actor-Critic (Haarnoja et al.)  
- FLYNN-style connectome-constrained RNN  
