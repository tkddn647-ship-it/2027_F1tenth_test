#!/usr/bin/env bash
# 사용법:  sim_ros2/run_policy.sh /path/to/best_model.zip [max_speed]
#  (먼저 다른 터미널에서 f1tenth_gym_ros 시뮬을 띄워 둘 것)
#  model 자리에 dummy 를 주면 랜덤 가중치로 파이프라인만 점검.
set -e
MODEL=${1:?"모델 경로 (best_model.zip / actor.ts.pt) 또는 dummy"}
MAXV=${2:-99.0}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/humble/setup.bash
[ -f ~/f1tenth_ws/install/setup.bash ] && source ~/f1tenth_ws/install/setup.bash
cd "$REPO"
POSE=$(python3 -c "import yaml,os;p=yaml.safe_load(open(os.path.expanduser('~/f1tenth_ws/src/f1tenth_gym_ros/config/sim.yaml')))['bridge']['ros__parameters'];print(p['sx'],p['sy'],p['stheta'])")
read SX SY ST <<< "$POSE"
python3 -m mapless40.sim_adapter --ros-args -p sx:=$SX -p sy:=$SY -p stheta:=$ST -p collision_dist:=${COLL_DIST:-0.10} &
ADP=$!
trap 'kill $ADP 2>/dev/null' EXIT
META="$(dirname "$MODEL")/actor_meta.json"
ARGS=(--ros-args -p scan_topic:=/mapless/scan -p mount_yaw:=0.0 -p front_check:=false
      -p imu_use_header_stamp:=false -p max_speed:=$MAXV -p meta:="$META")
if [ "$MODEL" = dummy ]; then
  python3 sim_ros2/dummy_policy_node.py "${ARGS[@]}" -p model:=dummy.zip
else
  python3 -m mapless40.ros_node "${ARGS[@]}" -p model:="$MODEL"
fi
