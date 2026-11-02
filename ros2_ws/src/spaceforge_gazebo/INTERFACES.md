# Phase 1 interfaces

Coordinate frame: relative LVLH/Hill frame centered on target. x is radial outward, y is along-track, z is orbit-normal. Gazebo world coordinates use metres and seconds; target is the origin.

State: the chaser state is [x,y,z,vx,vy,vz] in metres and metres/sec. The plugin owns this state and publishes ground truth every update on <topic_prefix>/model/chaser/pose (gz.msgs.Pose, stamped with simulation time) and <topic_prefix>/model/chaser/velocity (gz.msgs.Vector3d); topic_prefix defaults to /world/leo_orbit, matching the bridge's gz_prefix.

Dynamics: n is mean motion in rad/sec. The continuous CW model is xddot=3*n*n*x+2*n*vy+ax, yddot=-2*n*vx+ay, zddot=-n*n*z+az. The Python cw_dynamics module is the reference oracle; this plugin propagates with the same exact closed-form state transition matrix (no numerical integrator), then places the model with a world-pose command.

Command: publish gz.msgs.Vector3d to <topic_prefix>/model/chaser/command/accel. The vector is commanded acceleration in m/s^2 in LVLH axes. Commands are clamped by max_accel from SDF (non-finite components are zeroed) and read under a mutex. No ROS message bridge is assumed.

Reset: reset the Gazebo world. The initial position is the chaser's spawn pose; initial_velocity, n, max_accel, model_name and topic_prefix are SDF plugin parameters. A world reset (simulation time running backwards) re-initialises the state. Timestep: Gazebo simulation time is authoritative; the plugin integrates once per PreUpdate using UpdateInfo.dt and rejects non-positive dt.

Safety and docking remain out of scope for Phase 1. The plugin is a dynamics/observability bridge; safety authority remains in the Python reference oracle stack.
