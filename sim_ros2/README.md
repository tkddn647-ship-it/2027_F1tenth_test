# mapless40 정책 → f1tenth_gym_ros (ROS2 Humble) 시뮬

**모델을 학습한 브랜치와 같은 코드로 돌릴 것.** Colab 은 `feat/mapless40-asym-sac` 로 학습한다
(LiDAR range_max 10 m). main(15 m)으로 돌리면 입력 정규화가 달라 첫 코너에서 박는다.

```
f1tenth_gym_ros  /scan (~250 Hz) ─▶ sim_adapter ─▶ /mapless/scan (40 Hz) ─┐
                 /ego_racecar/odom ─▶ sim_adapter ─▶ /imu/data, /vehicle/speed_mps ─┤─▶ mapless40.ros_node
                 /drive ◀──────────────────────────────────────────────────────────┘
```

맵 선택 (학습 맵과 같은 점유 격자 + 레이싱라인 출발 자세로 sim.yaml 갱신, 재빌드 불필요):
```bash
python3 ~/2027_F1tenth_test/sim_ros2/set_map.py ifac             # 또는 roboracer_0817
python3 ~/2027_F1tenth_test/sim_ros2/set_map.py ifac --s0 20     # 라인 위 20 m 지점 출발
```

터미널 1 — 시뮬 (RViz 포함):
```bash
source /opt/ros/humble/setup.bash && source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

터미널 2 — 정책 (torch 불필요, numpy 추론):
```bash
~/2027_F1tenth_test/sim_ros2/run_policy.sh ~/2027_F1tenth_test/mapless40/last_model.zip   # [max_speed]
```
`dummy` 를 모델 자리에 주면 랜덤 가중치로 파이프라인만 점검.

시뮬 전용 설정: `mount_yaw=0` (시뮬 라이다 0° = 차 정면), `front_check=false` (시뮬엔 차체 가림 없음).
충돌하면 어댑터가 `/initialpose` 로 출발점에 자동 리셋 (`-p auto_reset:=false` 로 끔).

주의: `sim.yaml` 의 `scan_fov` / `scan_beams` 는 f1tenth_gym 실제 값(4.7 rad / 1080)이어야 한다.
브리지는 이 값을 gym 에 넘기지 않고 LaserScan 각도 표기에만 쓰므로, 다르게 쓰면 RViz·정책의 라이다 각도가 틀어진다.
