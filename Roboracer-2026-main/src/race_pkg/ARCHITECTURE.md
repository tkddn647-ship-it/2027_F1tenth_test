# CSV → 제어 노드 파이프라인 정리

CSV 발행부터 로컬 플래너가 제어 노드(waypoint_follow)에 경로를 보내는 과정까지 전체 파이프라인을 정리한 문서입니다.

---

## 1. 전체 데이터 흐름

```
[CSV 파일]  (centerline.csv / raceline.csv)
      │
      ▼  centerline_path 노드 (CSV 로드)
      │  · TF로 차량 위치 조회
      │  · 차량 기준 앞쪽 50개 waypoint → /recommended_path (2 Hz)
      │
      ├──────────────────────────────────────────────────────────┐
      │                                                          │
      ▼                                                          ▼
[static_obstacle]  /scan → 클러스터링·필터 → /static_obstacles   [FGM]  /scan + /static_obstacles
      │            (매 스캔)                                              → Safety Bubble → Max Gap
      │                                                                   → /fgm_target (매 스캔)
      │                                                          │
      └──────────────────────┬───────────────────────────────────┘
                             ▼
                    [로컬 플래너]
                    · /recommended_path, /static_obstacles, /fgm_target 구독
                    · 장애물 멀면: 추천경로 그대로
                    · 장애물 가까우면: [현재→FGM타겟] + 추천경로 앞구간 (최대 200개)
                    · /local_path 발행 (20 Hz)
                             │
                             ▼
                    [waypoint_follow]  /local_path 구독
                    · Pure Pursuit → /drive (50 ms 주기)
                             │
                             ▼
                    [차량 제어]  AckermannDriveStamped
```

---

## 2. 노드별 상세

### 2.1 centerline_path_node (센터라인패스)

| 항목 | 내용 |
|------|------|
| **역할** | CSV 기반 추천경로를 /recommended_path로 발행 |
| **입력** | CSV 파일 (x, y), TF (map ↔ base_link) |
| **출력** | `/recommended_path` (nav_msgs/Path), `/centerline_marker` (Marker) |
| **발행 주기** | 2 Hz |

**동작:**
1. 시작 시 CSV에서 전체 waypoint 로드
2. 2 Hz 타이머로 publish_path 호출
3. TF로 차량 현재 위치 조회
4. 전체 경로에서 차량에 가장 가까운 waypoint 인덱스 탐색
5. 그 인덱스부터 앞쪽 **50개** waypoint를 Path 메시지로 발행 (슬라이딩 윈도우)

**주요 파라미터:**
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| csv_path | Spielberg_raceline.csv | 추천경로 CSV 경로 |
| path_topic | /recommended_path | 발행 토픽 |
| publish_hz | 2.0 | 발행 주기 (Hz) |
| path_window_size | 50 | 차량 기준 앞쪽 waypoint 개수 |

---

### 2.2 static_obstacle_node (정적 장애물)

| 항목 | 내용 |
|------|------|
| **역할** | LiDAR 스캔에서 정적 장애물 검출 |
| **입력** | `/scan` (sensor_msgs/LaserScan) |
| **출력** | `/static_obstacles` (Float32MultiArray), `/visualization_marker_array` (MarkerArray) |
| **발행** | /scan 수신 시마다 (매 스캔) |

**동작:**
1. /scan 수신 → 유효 range 포인트만 추출
2. 인접 포인트 거리 > 0.2 m 기준으로 클러스터 분리
3. 포인트 5개 이상인 클러스터만 유효 장애물로 판단
4. 각 장애물: laser 프레임 기준 중심 (x, y), 반경(r) 계산
5. `[id, x, y, radius, id, x, y, radius, ...]` 형식으로 Float32MultiArray 발행

**/static_obstacles 포맷:**
- 4개씩 한 장애물: `id`, `x`, `y`, `radius`
- laser 프레임 기준 (x=전방, y=좌측)

---

### 2.3 fgm_node (FGM)

| 항목 | 내용 |
|------|------|
| **역할** | 막히지 않은 가장 큰 갭 방향으로 회피 목표점 발행 |
| **입력** | `/scan`, `/static_obstacles` |
| **출력** | `/fgm_target` (PointStamped), `/fgm_gap_marker` (Marker), `/fgm_debug_scan` (LaserScan) |
| **발행** | /scan 수신 시마다 (매 스캔) |

**동작:**
1. ranges 5 m 초과 구간은 5 m로 클리핑 (장애물 인식 거리)
2. Safety Bubble: 가장 가까운 점 + /static_obstacles 장애물 주변을 ranges에서 0으로 마스킹
3. Max Gap: `ranges > 1.2` (없으면 0.5)인 연속 구간 중 가장 긴 갭 선택
4. 갭 중심 방향으로 **3 m 앞**을 회피 목표점으로 /fgm_target 발행
5. 갭 시작/끝을 빨간 선으로 /fgm_gap_marker 발행 (시각화용, 차량 기준 1.5 m 오프셋)

**주요 파라미터 (코드 상수):**
- preprocess_dist: 5.0 m (장애물 인식 거리)
- gap_threshold: 1.2 m (갭 판단, 실패 시 0.5 m)
- target_dist: 3.0 m (FGM 목표점 거리)

---

### 2.4 local_planner_node (로컬 플래너)

| 항목 | 내용 |
|------|------|
| **역할** | 추천경로 또는 회피 경로를 /local_path로 발행 |
| **입력** | `/recommended_path`, `/static_obstacles`, `/fgm_target` |
| **출력** | `/local_path` (nav_msgs/Path) |
| **발행 주기** | 20 Hz |

**동작:**
1. 20 Hz 타이머로 timer_publish 호출
2. /recommended_path가 없거나 2개 미만이면 종료
3. **FGM 미사용** (use_fgm=False): /recommended_path를 그대로 /local_path에 발행
4. **장애물이 멂** (가장 가까운 전방 장애물 거리 > avoid_threshold): /recommended_path를 그대로 발행
5. **장애물이 가까움** (거리 ≤ avoid_threshold):
   - 현재 pose + FGM 타겟을 map 프레임으로 변환
   - FGM 타겟에 가장 가까운 추천경로 인덱스 탐색
   - Path 구성: [현재 pose, FGM 타겟] + [그 인덱스부터 추천경로 앞구간] (최대 200개)
   - /local_path 발행

**주요 파라미터:**
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| avoid_threshold | 0.7 | 회피 시작 거리 (m) |
| use_fgm | true | FGM 회피 사용 여부 |
| forward_cone_deg | 80 | 전방 장애물 판단 콘 각도 (deg) |
| publish_hz | 20.0 | 발행 주기 (Hz) |

---

### 2.5 waypoint_follow_node (제어 노드) 신경안써도됨 이건 어차피 현재랑 얘기해야함

| 항목 | 내용 |
|------|------|
| **역할** | /local_path 따라 Pure Pursuit으로 주행 |
| **입력** | `/local_path` (Path), TF (map ↔ base_link) |
| **출력** | `/drive` (ackermann_msgs/AckermannDriveStamped) |
| **콜백 주기** | 50 ms (20 Hz) |

**동작:**
1. /local_path 구독 → waypoint 리스트로 저장
2. 50 ms 타이머로 timer_callback 호출
3. TF로 차량 pose (x, y, yaw) 조회
4. 가장 가까운 waypoint 인덱스 탐색
5. lookahead_distance 이상 떨어진 앞쪽 waypoint를 목표점으로 선택
6. Pure Pursuit: 조향각 = atan2(목표) - yaw, 클리핑
7. /drive 발행 (steering_angle, speed)

**주요 파라미터:**
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| local_path_topic | /local_path | 구독 토픽 |
| lookahead_distance | 1.0 | Pure Pursuit lookahead (m) |
| speed | 2.0 | 주행 속도 |
| loop | true | 경로 루프 여부 |

---

## 3. 좌표 개수 요약

| 구간 | 개수 | 비고 |
|------|------|------|
| 센터라인패스 → 로컬플래너 | 50개 | path_window_size=50 |
| 로컬플래너 → waypoint_follow (일반) | 50개 | 추천경로 그대로 |
| 로컬플래너 → waypoint_follow (회피) | 2 ~ 200개 | [현재+FGM] + 추천경로 앞구간, 최대 200 |

---


## 4. 토픽 정리

| 토픽 | 타입 | 발행 노드 | 구독 노드 |
|------|------|-----------|-----------|
| /recommended_path | nav_msgs/Path | centerline_path | local_planner |
| /static_obstacles | Float32MultiArray | static_obstacle | local_planner, FGM |
| /fgm_target | PointStamped | FGM | local_planner |
| /local_path | nav_msgs/Path | local_planner | waypoint_follow |
| /drive | AckermannDriveStamped | waypoint_follow | 시뮬/차량 |
| /scan | LaserScan | 시뮬/센서 | static_obstacle, FGM |

---

## 5. CSV 생성 파이프라인 (참고)

주행에 사용하는 CSV는 다음 스크립트로 생성할 수 있습니다.

| 스크립트 | 입력 | 출력 |
|----------|------|------|
| extract_centerline_from_map.py | map (yaml+png) | centerline.csv |
| generate_raceline_from_centerline.py | centerline.csv + map.yaml | raceline.csv |


