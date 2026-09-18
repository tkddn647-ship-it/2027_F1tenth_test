cmake_minimum_required(VERSION 3.8)
project(race_layer)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  launch/path_following.py
  launch/static_node.py
  launch/FGM_node.py
  launch/local.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
