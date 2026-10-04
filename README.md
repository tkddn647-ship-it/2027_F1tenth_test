# 지도 없이 달리는 강화학습 자율주행 — F1TENTH / Roboracer 2026

1/10 크기 레이싱카가 **SLAM·위치추정 없이**, 센서 입력에서 바로 조향과 속도를 낸다. 아주대 팀.

**포트폴리오 웹페이지 → https://tkddn647-ship-it.github.io/2027_F1tenth_test/**

| | 01. LiDAR 강화학습 | 02. depth 카메라 + 초파리 커넥톰 |
|--|--|--|
| 센서 | LiDAR + IMU + 속도 (40 Hz) | depth 카메라 한 대 (30 Hz) |
| 정책 | 1D CNN + MLP, SAC (asymmetric actor-critic) | 초파리 시각계를 본뜬 회로 (학습 값 약 760개), 모방학습 → SAC |
| 결과 (시뮬) | 팀 트랙 3바퀴 무사고 **9.63 초**, ifac 10.9 초 | 모방학습만으로 ifac **12.8 초**, 팀 트랙 11.5 초 (진행 중) |
| 자세히 | [portfolio/01_라이다_강화학습.md](portfolio/01_라이다_강화학습.md) | [portfolio/02_카메라_초파리커넥톰.md](portfolio/02_카메라_초파리커넥톰.md) |

### 01. LiDAR 강화학습

| ifac | 팀 트랙 |
|--|--|
| ![](docs/assets/v3_318k_ifac.gif) | ![](docs/assets/v3_318k_team.gif) |

### 02. depth 카메라 + 초파리 시각 회로 (회로가 직접 운전)

![](docs/assets/circuit_lap_ifac.gif)

*오른쪽 위 → 아래: 파리 눈 '가까움' 영상, 움직임, 시각엽 출력, 하행 뉴런 48개. 결과는 모두 시뮬레이션이고, 회로는 실제 초파리 연결 데이터가 아닌 원칙을 따른 합성 회로.*

---

| 폴더 | 내용 |
|--|--|
| [`mapless40/`](mapless40/) | LiDAR 정책 + 공통 차량 시뮬레이터·SAC ([실험 기록](mapless40/RESULTS.md)) |
| [`camera/depthfly/`](camera/depthfly/) | depth 카메라 + 초파리 회로 (극성 검사 `polarity.py`, 모방학습 `bc.py`) |
| [`realcar/`](realcar/) · [`Roboracer-2026-main/`](Roboracer-2026-main/) | 실차(Jetson ROS2) 실행 |
| [`portfolio/`](portfolio/) · [`docs/index.html`](docs/index.html) | 포트폴리오 문서 · 웹페이지 |
| [`CLAUDE.md`](CLAUDE.md) · [`REPORT.md`](REPORT.md) | 작업 현황 · 상세 진행 보고서 |
| [`docs/LEGACY_README.md`](docs/LEGACY_README.md) | 초기 실험(hemibrain 커넥톰 RNN 등) 설명 |
