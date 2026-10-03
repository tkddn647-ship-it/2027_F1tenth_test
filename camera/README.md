# camera — 카메라(Orbbec Gemini 2L) 전용 작업 폴더

LiDAR 없이 카메라만으로 달리는 코드는 **전부 이 폴더 안**에 있다. LiDAR 정책은 루트의 `mapless40/`.

| 폴더 | 입력 | 정책 | 상태 |
|--|--|--|--|
| **[`depthfly/`](depthfly/README.md)** | **depth 만** (16×64 가까움 영상 × 3프레임, 30 Hz) | **초파리 커넥톰만** → 선형 읽기 | **주력.** numpy 테스트 6/6, torch 테스트·학습 아직 |
| [`camfly/`](camfly/README.md) | RGB(흑백) + depth 스캔 | 커넥톰 (비교용 CNN 포함) | 첫 구조. depthfly 가 카메라 기하·회로 부품을 여기서 가져다 씀 — **지우지 말 것** |

**왜 depth 인가:** 덕트·장애물 색이 랜덤이어도 depth 는 거리만 잰다. RGB 는 색·조명·반사를 전부 랜덤화해야 하고
"이게 벽이고 얼마나 멀다"를 정책이 배워야 해서 작은 커넥톰 회로엔 어렵다. 위험은 덕트 재질(은박·검정)에서 depth 구멍 → 카메라 오면 먼저 실측.

## 실행 (레포 루트에서)

```bash
python -m camera.depthfly.tests                       # numpy 6 + torch 2
python -m camera.depthfly.viz --map ifac --out eye.gif
python -m camera.depthfly.train --n-envs 4 --subproc --save-dir runs/depthfly --resume auto
python -m camera.depthfly.evaluate --model runs/depthfly/best_model.zip --maps ifac,roboracer_0817 --obstacles 2
```
코랩: `camera/depthfly/colab_train.ipynb`. 실차: `python3 -m camera.depthfly.ros_node --ros-args -p model:=best_model.zip -p max_speed:=2.0`.

## 무엇을 밖에서 가져다 쓰나

- `mapless40/` (루트): 차량 시뮬 `MaplessRaceEnv40`, 보상, 레이싱라인(critic 전용), 장애물, SAC `AsymSACPolicy`, 학습 콜백, zip 로더.
  카메라 환경은 이걸 상속해서 센서(`_raw_scan`)만 바꾼다. 그래서 `mapless40` 은 루트에 그대로 둔다.
- `maps/`, `f1tenth_racetracks/` (루트): 트랙. LiDAR 2D 맵의 벽을 33 cm 덕트로 세워 depth 를 계산한다.

다음 할 일은 [`depthfly/README.md` §6~7](depthfly/README.md) 과 루트 [`CLAUDE.md`](../CLAUDE.md).
