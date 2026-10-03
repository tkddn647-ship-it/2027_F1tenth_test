#!/usr/bin/env bash
# mapless40 모델 실차 주행 (위치추정·SLAM·AEB 없음)
#
#   ./realcar/run_realcar.sh
#
# 띄우는 것:  sensor_layer (LiDAR 40 Hz + IMU + 고정 TF)
#             mapless40.ros_node (모델, /drive 발행)
#             control_node (전경 — Space=ESTOP, r=해제). 이 레포의 control_node.py 를 직접 실행
#             (RC 신호 끊기면 AUTO 해제 수정 포함 — Jetson 워크스페이스 빌드 불필요)
# 조종기:     CH5 수동 = 사람 조종 / CH5 자율 = 모델 / CH6 = ESTOP
#
# 환경변수로 바꿀 수 있는 값 (기본값):
#   MAX_SPEED=3.5           모델 속도 상한 [m/s] (정책 출력 2~5 m/s 를 여기서 자름)
#   MODEL=<repo>/mapless40/last_model.zip
#   WS=/home/nvidia/f1tenth_ajou   실차 ROS2 워크스페이스
#   MOUNT_YAW=              비우면 TF base_link→laser 사용. 시작 검사가 실패하면 로그의 추정값을 넣을 것
#   BAG=1                   rosbag 기록 (나중에 서보·감속 실측용). 0 이면 끔
set -u

MAX_SPEED=${MAX_SPEED:-3.5}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=${MODEL:-$REPO/mapless40/last_model.zip}
WS=${WS:-/home/nvidia/f1tenth_ajou}
MOUNT_YAW=${MOUNT_YAW:-}
BAG=${BAG:-1}
# 테스트용 대체 명령 (보통 건드리지 않음)
SENSOR_CMD=${SENSOR_CMD:-ros2 launch sensor_layer sensor_layer_launch.py}
CONTROL_CMD=${CONTROL_CMD:-python3 $REPO/Roboracer-2026-main/src/path_following/path_following/control_node.py}

LOG="$REPO/realcar/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 사전 확인
[ -f "$MODEL" ] || { red "모델 없음: $MODEL"; exit 1; }
set +u   # ROS setup.bash 는 정의 안 된 변수를 씀
source /opt/ros/humble/setup.bash
if [ -f "$WS/install/setup.bash" ]; then source "$WS/install/setup.bash"
else red "워크스페이스 없음: $WS/install/setup.bash  (WS=... 로 지정)"; exit 1; fi
set -u
python3 -c "import numpy, serial" 2>/dev/null || { red "numpy / pyserial 없음 (pip3 install pyserial)"; exit 1; }

PIDS=()
cleanup() {
  echo; echo "[run] 종료 중..."
  # setsid 로 띄워서 PID = 프로세스 그룹 → ros2 launch/run 아래 노드까지 한 번에 종료
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done
  echo "[run] 로그: $LOG"
}
trap cleanup EXIT

if ros2 topic info /drive 2>/dev/null | grep -q "Publisher count: [1-9]"; then
  red "/drive 를 이미 누가 발행 중 (Stanley/FGM 등). 끄고 다시 실행:"; ros2 topic info /drive -v | grep "Node name"; exit 1
fi

# ---------------------------------------------------------------- 1) 센서
echo "[run] 1/3 sensor_layer 시작 → $LOG/sensor.log"
setsid $SENSOR_CMD > "$LOG/sensor.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 20); do
  ros2 topic list 2>/dev/null | grep -qx /scan && break; sleep 1
done
ros2 topic list | grep -qx /scan || { red "/scan 이 안 나옴 — LiDAR(UDP 192.168.11.2) 연결 확인. $LOG/sensor.log"; exit 1; }
green "[run]   /scan OK"

# ---------------------------------------------------------------- 2) 모델
echo "[run] 2/3 모델 시작 (max_speed $MAX_SPEED m/s) — 차를 세워 두고, 주변 0.35 m 안에 물건 두지 말 것"
ARGS=(--ros-args -p model:="$MODEL" -p meta:=none -p max_speed:="$MAX_SPEED")
[ -n "$MOUNT_YAW" ] && ARGS+=(-p mount_yaw:="$MOUNT_YAW")
cd "$REPO"
setsid python3 -u -m mapless40.ros_node "${ARGS[@]}" > "$LOG/model.log" 2>&1 &
PIDS+=($!)
ok=0
for i in $(seq 1 30); do
  if grep -q "시작 검사 통과" "$LOG/model.log"; then ok=1; break; fi
  if grep -q "라이다 방향이 틀림" "$LOG/model.log"; then ok=2; break; fi
  kill -0 "${PIDS[-1]}" 2>/dev/null || { ok=3; break; }
  sleep 0.5
done
grep -E "mount_yaw|시작 검사|가린 구간|WARN" "$LOG/model.log" | sed 's/^/    /'
case $ok in
  1) green "[run]   모델 시작 검사 통과" ;;
  2) red "라이다 방향이 틀림 → 위 로그의 'mount_yaw:=' 값으로 다시:  MOUNT_YAW=<값> $0"; exit 1 ;;
  3) red "모델 노드가 죽음:"; tail -20 "$LOG/model.log"; exit 1 ;;
  *) red "15 s 안에 시작 검사가 안 끝남 (스캔 수신?)"; tail -10 "$LOG/model.log"; exit 1 ;;
esac

# ---------------------------------------------------------------- rosbag
if [ "$BAG" = "1" ]; then
  setsid ros2 bag record -o "$LOG/bag" /scan /imu/data /vehicle/speed_mps /drive /vehicle/telemetry \
    > "$LOG/bag.log" 2>&1 &
  PIDS+=($!)
  echo "[run]   rosbag → $LOG/bag"
fi

# ---------------------------------------------------------------- 3) 하드웨어
echo
green "[run] 3/3 control_node 시작 (이 터미널: Space=ESTOP, r=해제, Ctrl-C=전체 종료)"
echo "      CH5 수동 → 사람 조종 / CH5 자율 → 모델 ($MAX_SPEED m/s 상한) / CH6 → ESTOP"
echo "      모델 상태 보기: tail -f $LOG/model.log"
echo
$CONTROL_CMD
