# spaceforge_gazebo

A Gazebo Harmonic world (`worlds/leo_orbit.sdf`) with target and chaser models, a C++17 system
plugin that propagates the chaser with the exact CW state transition matrix, and
`spaceforge_sim_bridge`, which connects the plugin to ROS 2 topics.

```bash
ros2 launch spaceforge_gazebo leo_orbit.launch.py headless:=true
```

The `ros` CI job builds the package and runs a live test: it starts Gazebo headless with the
bridge, sends a wrench command over ROS and checks that the chaser's velocity responds. Topic and
frame details are in `INTERFACES.md` (Gazebo side) and `SIM_BRIDGE.md` (ROS side).
