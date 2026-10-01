# mapless40 실차 주행

위치추정·SLAM·AEB 없이 모델 하나로 달린다. 조종기로 수동 ↔ 모델을 바꾼다.

## 준비 (Jetson, 한 번만)
```bash
cd ~ && git clone -b feat/mapless40-asym-sac https://github.com/tkddn647-ship-it/2027_F1tenth_test
# 모델 zip 이 레포에 없으면 PC 에서 복사:
#   scp mapless40/last_model.zip nvidia@<jetson-ip>:~/2027_F1tenth_test/mapless40/
```
실차 워크스페이스 `/home/nvidia/f1tenth_ajou` 가 빌드돼 있어야 한다 (sensor_layer, path_following). torch 불필요 (numpy).

## 실행
```bash
~/2027_F1tenth_test/realcar/run_realcar.sh
```
1. **CH5 를 수동에 두고**, 차를 세워 둔 채 실행 (시작 검사가 정지 스캔으로 라이다 방향을 확인).
2. `[run] 3/3 control_node 시작` 이 뜨면 준비 끝. 수동으로 트랙에 올린다.
3. **CH5 자율 → 모델 주행** (최대 3.5 m/s). 이상하면 CH5 수동 / CH6 ESTOP / 터미널 Space.
4. Ctrl-C → 전부 종료. 로그·rosbag: `realcar/logs/<시각>/`

| 환경변수 | 기본 | |
|--|--|--|
| `MAX_SPEED` | 3.5 | 모델 속도 상한 m/s. 처음엔 `MAX_SPEED=2.0` 권장 |
| `MOUNT_YAW` | (TF) | 시작 검사 실패 시 로그에 나오는 `mount_yaw:=` 값 |
| `WS` | /home/nvidia/f1tenth_ajou | |
| `BAG` | 1 | rosbag 기록 (0=끔) |

## 시작 검사에서 멈추면
- `라이다 방향이 틀림` → `MOUNT_YAW=<로그의 값> ./realcar/run_realcar.sh`
- `/scan 이 안 나옴` → LiDAR UDP(192.168.11.2) 연결, `realcar/logs/*/sensor.log`
- `/drive 를 이미 누가 발행 중` → Stanley/FGM 등 끄고 다시

## 주의
- 정책 출력은 2~5 m/s 라 최소 2 m/s 로 달린다. 정지는 CH5 수동 / ESTOP 으로만.
- 능동 브레이크 없음 (타력 감속). ESTOP 도 duty 0 → 타력 정지.
- control_node 는 AUTO 중 RC 신호가 끊겨도 AUTO 를 유지한다 (`_is_autonomous_mode`). 수신기 failsafe 로 CH5 가 수동값이 되는지 확인할 것.
