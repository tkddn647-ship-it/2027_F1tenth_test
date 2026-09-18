#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    csv_path = LaunchConfiguration("csv_path")
    path_window_size = LaunchConfiguration("path_window_size")
    avoid_threshold = LaunchConfiguration("avoid_threshold")
    use_fgm = LaunchConfiguration("use_fgm")
    forward_cone_deg = LaunchConfiguration("forward_cone_deg")
    publish_hz = LaunchConfiguration("publish_hz")
    map_frame = LaunchConfiguration("map_frame")
    base_frame = LaunchConfiguration("base_frame")
    laser_frame = LaunchConfiguration("laser_frame")
    waypoint_offset_x = LaunchConfiguration("waypoint_offset_x")
    waypoint_offset_y = LaunchConfiguration("waypoint_offset_y")
    waypoint_flip_y = LaunchConfiguration("waypoint_flip_y")

    return LaunchDescription([
        DeclareLaunchArgument(
            "csv_path",
            default_value="/home/tkddn647/test/maps/Austin_centerline.csv",
            description="Path to centerline/raceline CSV used for /recommended_path",
        ),
        DeclareLaunchArgument(
            "path_window_size",
            default_value="80",
            description="Number of forward waypoints published on /recommended_path",
        ),
        DeclareLaunchArgument(
            "avoid_threshold",
            default_value="0.7",
            description="Obstacle distance threshold (m) for switching to FGM-assisted local path",
        ),
        DeclareLaunchArgument(
            "use_fgm",
            default_value="true",
            description="If false, local planner publishes recommended path without FGM avoidance",
        ),
        DeclareLaunchArgument(
            "forward_cone_deg",
            default_value="80.0",
            description="Forward cone angle (deg) for obstacle filtering in local planner",
        ),
        DeclareLaunchArgument(
            "publish_hz",
            default_value="20.0",
            description="Publish rate for local planner output path",
        ),
        DeclareLaunchArgument(
            "map_frame",
            default_value="map",
            description="Global map frame",
        ),
        DeclareLaunchArgument(
            "base_frame",
            default_value="base_link",
            description="Vehicle base frame",
        ),
        DeclareLaunchArgument(
            "laser_frame",
            default_value="laser",
            description="Laser frame",
        ),
        DeclareLaunchArgument(
            "waypoint_offset_x",
            default_value="0.0",
            description="X offset applied to CSV waypoints",
        ),
        DeclareLaunchArgument(
            "waypoint_offset_y",
            default_value="0.0",
            description="Y offset applied to CSV waypoints",
        ),
        DeclareLaunchArgument(
            "waypoint_flip_y",
            default_value="false",
            description="Flip waypoint Y axis (true/false)",
        ),
        Node(
            package="race_layer",
            executable="path_following.py",
            name="centerline_path",
            output="screen",
            parameters=[{
                "csv_path": csv_path,
                "path_window_size": ParameterValue(path_window_size, value_type=int),
                "frame_id": map_frame,
                "map_frame": map_frame,
                "base_frame": base_frame,
                "waypoint_offset_x": ParameterValue(waypoint_offset_x, value_type=float),
                "waypoint_offset_y": ParameterValue(waypoint_offset_y, value_type=float),
                "waypoint_flip_y": ParameterValue(waypoint_flip_y, value_type=bool),
            }],
        ),
        Node(
            package="race_layer",
            executable="static_node.py",
            name="static_obstacle_node",
            output="screen",
        ),
        Node(
            package="race_layer",
            executable="FGM_node.py",
            name="fgm_node",
            output="screen",
        ),
        Node(
            package="race_layer",
            executable="local.py",
            name="local_planner",
            output="screen",
            parameters=[{
                "avoid_threshold": ParameterValue(avoid_threshold, value_type=float),
                "use_fgm": ParameterValue(use_fgm, value_type=bool),
                "forward_cone_deg": ParameterValue(forward_cone_deg, value_type=float),
                "publish_hz": ParameterValue(publish_hz, value_type=float),
                "map_frame": map_frame,
                "base_frame": base_frame,
                "laser_frame": laser_frame,
            }],
        ),
    ])
