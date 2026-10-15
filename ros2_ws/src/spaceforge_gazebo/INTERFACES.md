# Phase 1 interfaces

Coordinate frame: relative LVLH/Hill frame centered on target. x is radial outward, y is along-track, z is orbit-normal. Gazebo world coordinates use metres and seconds; target is the origin.

State: the chaser state is [x,y,z,vx,vy,vz] in metres and metres/sec. The plugin publishes ground truth as Gazebo components on /model/chaser/pose and /model/chaser/velocity; exact topic names are middleware-version dependent and must be confirmed with gz topic -l.

Dynamics: n is mean motion in rad/sec. The continuous CW model is xddot=3*n*n*x+2*n*vy+ax, yddot=-2*n*vx+ay, zddot=-n*n*z+az. The Python cw_dynamics module is the oracle; this plugin uses explicit fixed-step integration only as a Phase 1 bridge.

Command: publish gz.msgs.Vector3d to /model/chaser/command/accel. The vector is commanded acceleration in m/s^2 in LVLH axes. Commands are clamped by max_accel from SDF. No ROS message bridge is assumed.

Reset: reset the Gazebo world. Initial relative position, velocity, n, dt, and max_accel are SDF plugin parameters. Timestep: Gazebo simulation time is authoritative; the plugin integrates once per PreUpdate using UpdateInfo.dt and rejects non-positive dt.

Safety and docking remain out of scope for Phase 1. The plugin is a dynamics/observability bridge; safety authority remains in the Python reference oracle stack.
