# lidarfly — LiDAR 1080빔 × 3프레임 + 초파리 커넥톰

맵·위치추정 없이 LiDAR 스캔만으로 조향·속도를 낸다. CNN·MLP actor 없음.

```
LiDAR 1080빔, FOV 270°, 40 Hz
  × 최근 3프레임 (현재 + 과거 2)
  → 15빔씩 72열 풀링 (중앙값·최근접) → 가까움 0.25 m / 거리
  → lamina ON/OFF → T4/T5 → HS/VS, LPLC2, LC, L3
  → DN 48 (좌우 대칭 배선, 세기만 학습) ‖ 상태 14
  → 선형 읽기 → (조향 ±21.4°, 속도 2~7 m/s)
critic (학습 전용): 같은 회로 + 레이싱라인 privileged 27 → MLP 256·256
```

차량 동역학·레이싱라인은 `mapless40`. `lidarfly_config()` 가 덮어쓰지 않은 값(μ, 횡가속 한계, `range_max` 10 m 등)은 `mapless40` `EnvConfig` 기본값이다. 그 주석은 `mapless40/config.py`.

학습은 코랩 [`colab_train.ipynb`](colab_train.ipynb). 세션이 끊기면 ①과 ④만 다시 실행한다. `--resume auto` 가 `/content/drive/MyDrive/lidarfly_runs/v1` 의 `last_model.zip`(또는 더 최신 체크포인트)에서 이어 간다. 노트북 `N_ENVS` 는 `min(os.cpu_count() or 1, 8)` (Colab CPU 코어 수, 상한 8). 맵 랜덤화는 CPU 병렬 env에서 돌고 GPU(T4)는 SAC 업데이트에만 쓰임. **이 저장소 CI에서는 학습을 돌리지 않는다. 학습 필요(미실행).**

```bash
python -m lidarfly.tests
python -m lidarfly.train --maps "ifac:3,roboracer_0817:3,random:2" --v-max 7 \
    --timesteps 1000000 --n-envs 4 --subproc --save-dir runs/lidarfly --resume auto
python -m lidarfly.evaluate --model runs/lidarfly/best_model.zip \
    --maps ifac,roboracer_0817 --v-max 7
```

`random` 맵은 에피소드 `reset` 마다 폐곡선을 다시 만든다. 병렬 env `i` 는 `--seed + i` 를 쓴다. `--n-envs N --seed S` 이면 시드는 `S .. S+N-1` 이고, env 마다 다른 랜덤 맵이 이어진다. `--subproc` 는 `SubprocVecEnv`, 없으면 `DummyVecEnv`. Windows 는 `train.py` 의 `if __name__ == "__main__"` 가드 아래에서 `--subproc` 가 동작하고, 문제가 나면 `--subproc` 없이 실행한다. 학습 시작 로그: `병렬 env N개, 시드 a..b, random 맵 에피소드마다 재생성`.

torch 없는 numpy 배포(`np_actor`)는 아직 없다.

## 설정 값 출처

`config.py` 가 이 패키지에서 바꾼 값만. 추정과 실측을 같은 칸에 넣지 않는다.

| 값 | 숫자 | 출처 |
|--|--|--|
| 빔 수, FOV | 1080, 270° | **사용자 지정.** 데이터시트 각도 분해능으로 역산한 값이 아니다 |
| 주기, 프레임 | 40 Hz (`dt` 0.025 s), 히스토리 3 | **사용자 지정** (이슈: 1080빔, 40 Hz, 현재+과거 2) |
| 속도 | 2~7 m/s | **사용자 지정.** 실차 최고 7 m/s 는 사용자 확인(2026-10-03)이지, 이 레포에서 새로 잰 시계열은 아니다 |
| 최대 조향 | 0.3735 rad (약 21.4°) | **실측.** `control_node.py` `max_steering_angle_rad`, 20260816 요레이트 실측 |
| 서보 시정수 | τ = −0.020 / ln(1−0.20) ≈ 90 ms | **펌웨어.** `esp_smooth_factor=0.20`, `esp_loop_period_sec=0.020`. 스텝 응답 실측은 아니다 |
| 서보 데드타임, 각속도 상한 | 0.015 s, 4.0 rad/s | **추정.** 순수 데드타임은 실차 코드에 없음. Stanley `steering_rate_limit` 은 서보각 단위라 전륜 기구 실측이 아니다 |
| 구동 | 가속 5.0 m/s², 제동 4.0 m/s², 타력 `0.6+0.15·v`, 지연 0.030 s, `speed_kp` 4.0 1/s, jerk 40 | **추정.** `control_node` 의 `speed_kp=0.15` 는 duty PI 라 단위가 다르다. `speed_profile.py` 의 a_accel 7 / a_brake 4 는 파일 주석이 TODO 실측이라 쓰지 않는다 |
| 능동 브레이크 끔 | `brake_enabled=False` | **코드 사실.** AUTO 속도 PI duty 하한 0. 감속 크기 실측은 아니다 |
| 열 72 × 15빔, 가까움 `0.25/거리` | | **설계값.** 센서 분해능·라이다 교정으로 역산한 값이 아니다. 가까움 정의는 depthfly 회로 스케일과 같다 |

## 보상·극성은 임의값

**보상 가중치는 임의 초기값이다.** 데이터시트도 실측도 아니다. 튜닝 전 숫자이며 `config.py` 상수 한곳만 바꾼다.

| 상수 | 임의값 | 역할 |
|--|--|--|
| `W_PROGRESS` | 1.0 | 레이싱라인 전진 Δs |
| `W_EY` | 0.5 | 횡오차 e_y²·Δt |
| `W_V_TRACK` | 0.08 | v_ref 대비 속도 편차 |
| `OVER_MULT` | 3.0 | v > v_ref 일 때 위 항의 배율 |
| `W_ALAT` | 1.0 | 횡가속 한계 초과 |
| `W_WALL` | 2.0 | 벽 근접 |
| `COLLISION` | −20.0 | 충돌 시 그 스텝 보상 |

**DN 극성(흥분/억제 부호)도 임의값이다.** 좌우 20쌍은 같은 쪽 흥분·반대쪽 억제, 가운데 8개는 그 규칙의 고정 배선이다. 초파리 연결 실측이 아니고, 학습되는 것은 세기뿐이다. 부호 버퍼는 기울기가 없다 (`make_dn_wiring_bilateral`).
