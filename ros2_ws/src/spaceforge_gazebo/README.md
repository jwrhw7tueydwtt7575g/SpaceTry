# SpaceForge ROS 2/Gazebo Phase 1

This package is separate from the Python MVP. It defines a minimal Gazebo Harmonic world (leo_orbit.sdf) with target and chaser models, sensor placeholders, and a C++17 CW system plugin. The Python simulator remains the reference oracle.

Host inspection result: this host has no ros2, gz, or gazebo executable. No launch or build claim is made here. On ROS 2 Humble plus Gazebo Harmonic, source /opt/ros/humble/setup.bash, run rosdep install, then colcon build --symlink-install --packages-select spaceforge_gazebo, source the workspace, and run ros2 launch spaceforge_gazebo leo_orbit.launch.py.

Validate only on that host: gz sim -v 3 should start the world and gz topic -e -t /model/chaser/state should expose state. See INTERFACES.md.
