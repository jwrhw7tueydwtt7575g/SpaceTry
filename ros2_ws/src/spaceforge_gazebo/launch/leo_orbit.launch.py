from launch import LaunchDescription
import os

from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share = get_package_share_directory("spaceforge_gazebo")
    # The CW system plugin installs to <prefix>/lib and the chaser model to
    # share/models; Gazebo only finds them through these search paths.
    prefix = os.path.dirname(os.path.dirname(share.rstrip("/")))
    scenario = LaunchConfiguration("scenario")
    world = PythonExpression(["\x27", share, "/worlds/\x27 + \x27", scenario, "\x27 + \x27.sdf\x27"])
    ros_namespace = LaunchConfiguration("namespace")
    ros_prefix = LaunchConfiguration("ros_prefix")
    gz_prefix = LaunchConfiguration("gz_prefix")
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="leo_orbit"),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("ros_prefix", default_value="spaceforge"),
        DeclareLaunchArgument("gz_prefix", default_value="/world/leo_orbit"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("clock_rate_hz", default_value="100.0"),
        DeclareLaunchArgument("odometry_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("imu_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("camera_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("lidar_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("headless", default_value="false",
                              description="Run the Gazebo server only (no GUI), e.g. on CI"),
        AppendEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", os.path.join(prefix, "lib")),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(share, "models")),
        ExecuteProcess(cmd=["gz", "sim", "-r", "-v", "3", world], output="screen",
                       condition=UnlessCondition(LaunchConfiguration("headless"))),
        ExecuteProcess(cmd=["gz", "sim", "-s", "-r", "-v", "3", world], output="screen",
                       condition=IfCondition(LaunchConfiguration("headless"))),
        Node(package="spaceforge_gazebo", executable="spaceforge_sim_bridge", name="sim_bridge", namespace=ros_namespace, output="screen", parameters=[{
            "scenario": scenario, "ros_prefix": ros_prefix, "gz_prefix": gz_prefix, "use_sim_time": use_sim_time,
            "clock_rate_hz": LaunchConfiguration("clock_rate_hz"), "odometry_rate_hz": LaunchConfiguration("odometry_rate_hz"),
            "imu_rate_hz": LaunchConfiguration("imu_rate_hz"), "camera_rate_hz": LaunchConfiguration("camera_rate_hz"), "lidar_rate_hz": LaunchConfiguration("lidar_rate_hz")
        }])
    ])
