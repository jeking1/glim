import launch
import launch_ros.actions
import launch.substitutions
import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("cuda_slam_ros")

    config_file = os.path.join(pkg_dir, "config", "cuda_slam_ros.yaml")

    cuda_slam_node = launch_ros.actions.Node(
        package="cuda_slam_ros",
        executable="cuda_slam_ros_node",
        name="cuda_slam_ros",
        output="screen",
        parameters=[config_file],
        remappings=[
            ("imu", "/os_cloud_node/imu"),
            ("points", "/os_cloud_node/points"),
        ],
    )

    return launch.LaunchDescription([cuda_slam_node])