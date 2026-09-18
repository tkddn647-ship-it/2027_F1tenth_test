cmake_minimum_required(VERSION 3.8)
project(tf_manager_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

add_executable(sensor_static_tf src/sensor_static_tf.cpp)
ament_target_dependencies(sensor_static_tf
  rclcpp tf2_ros geometry_msgs)

add_executable(wheel_odom_tf src/wheel_odom_tf.cpp)
ament_target_dependencies(wheel_odom_tf
  rclcpp tf2_ros geometry_msgs nav_msgs)

install(TARGETS
  sensor_static_tf
  wheel_odom_tf
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME})
endif()

ament_package()
