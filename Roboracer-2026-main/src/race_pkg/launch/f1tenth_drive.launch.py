import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'race_pkg'
    pkg_dir = get_package_share_directory(pkg_name)
    
    # 맵/경로 파일 설정 (config 폴더에 있는 파일 이름과 똑같아야 합니다!)
    # 만약 Spielberg_raceline.csv가 아직 없다면 생성 스크립트를 먼저 돌려야 합니다.
    csv_file_name = 'Spielberg_raceline.csv' 
    csv_path = os.path.join(pkg_dir, 'config', csv_file_name)

    return LaunchDescription([
        # 1. 전역 경로(Centerline/Raceline) 발행 노드
        Node(
            package=pkg_name,
            executable='centerline_publisher',
            name='centerline_publisher',
            parameters=[{'csv_path': csv_path}],
            output='screen'
        ),
        # 2. 정적 장애물 인식 노드
        Node(
            package=pkg_name,
            executable='static_obstacle_detector',
            name='static_obstacle_detector',
            output='screen'
        ),
        # 3. FGM (장애물 회피 타겟 계산) 노드
        Node(
            package=pkg_name,
            executable='fgm_node',
            name='fgm_node',
            output='screen'
        ),
        # 4. 로컬 플래너 (경로 중재) 노드
        Node(
            package=pkg_name,
            executable='local_planner',
            name='local_planner',
            parameters=[
                {'use_fgm': True},        # FGM 회피 기능 켜기
                {'avoid_threshold': 0.8}  # 0.8m 앞에 장애물 있으면 회피 시작
            ],
            output='screen'
        ),
    ])
