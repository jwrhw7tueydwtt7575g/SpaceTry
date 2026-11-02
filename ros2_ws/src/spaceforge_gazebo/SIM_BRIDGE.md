# Sim bridge contract

The executable spaceforge_sim_bridge is the only ROS 2/Gazebo transport translator. It contains no policy, safety filter, perception, planning, or LLM code.

| ROS 2 topic | Type | Direction | Frame / units | Rate | QoS |
|---|---|---|---|---|---|
| spaceforge/clock | rosgraph_msgs/msg/Clock | Gazebo -> ROS | simulation time, sec+nsec | 100 Hz | reliable, depth 10 |
| spaceforge/chaser/odometry | nav_msgs/msg/Odometry | Gazebo -> ROS | target_lvlh -> chaser_base; m, m/s, quaternion | 50 Hz | SensorDataQoS |
| spaceforge/chaser/imu | sensor_msgs/msg/Imu | Gazebo -> ROS | chaser_imu; m/s2, rad/s | 50 Hz | SensorDataQoS |
| spaceforge/chaser/camera/image_raw | sensor_msgs/msg/Image | Gazebo -> ROS | chaser_camera; 8-bit RGB image | 10 Hz | SensorDataQoS |
| spaceforge/chaser/lidar/scan | sensor_msgs/msg/LaserScan | Gazebo -> ROS | chaser_lidar; metres, radians | 10 Hz | SensorDataQoS |
| spaceforge/chaser/cmd_wrench | geometry_msgs/msg/WrenchStamped | ROS -> Gazebo | target_lvlh; force N, torque N m | command rate | reliable, depth 10 |

Gazebo topics are derived from the scenario/world prefix: /world/leo_orbit/clock, /world/leo_orbit/model/chaser/pose, /world/leo_orbit/model/chaser/velocity, and sensor topics below the chaser body. The bridge publishes acceleration Vector3d to /world/leo_orbit/model/chaser/command/accel after dividing wrench force by chaser_mass_kg. The CW point-mass plugin has no rotational actuator, so wrench torque is ignored.

All forwarded messages preserve the Gazebo simulation timestamp. Header frame IDs are configurable parameters but default to target_lvlh, chaser_base, chaser_imu, chaser_camera, and chaser_lidar. Rates are bridge publication caps, not simulator clock rates.

Parameters: scenario, gz_prefix, ros_prefix, chaser_mass_kg, clock_rate_hz, odometry_rate_hz, imu_rate_hz, camera_rate_hz, lidar_rate_hz, odom_frame, base_frame, imu_frame, camera_frame, lidar_frame. Launch exposes scenario, namespace, ros_prefix, gz_prefix, use_sim_time, and all sensor rates.
