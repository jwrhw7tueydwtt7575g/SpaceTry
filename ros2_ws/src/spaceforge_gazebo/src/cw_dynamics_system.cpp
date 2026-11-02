// Clohessy-Wiltshire point-mass dynamics for the chaser model.
//
// The plugin owns the chaser's relative state and propagates it with the exact
// CW state transition matrix -- the same closed form as the Python oracle in
// spaceforge_mvp/core/cw_dynamics.py -- once per Gazebo update. It then places
// the model with a world-pose command (so Gazebo's own physics never fights
// it) and publishes the ground-truth pose and velocity on the topics the ROS
// bridge subscribes to:
//
//   <topic_prefix>/model/<model_name>/command/accel   (gz.msgs.Vector3d, in)
//   <topic_prefix>/model/<model_name>/pose            (gz.msgs.Pose, out)
//   <topic_prefix>/model/<model_name>/velocity        (gz.msgs.Vector3d, out)
//
// The command arrives on a gz-transport thread; it is guarded by a mutex.

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/vector3d.pb.h>
#include <gz/msgs/Utility.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace spaceforge_gazebo {

// Exact CW propagation over dt under constant acceleration a (Hill frame:
// x radial, y along-track, z orbit-normal). Mirrors cw_stm_matrices().
void CwStep(gz::math::Vector3d &p, gz::math::Vector3d &v, const gz::math::Vector3d &a,
            double dt, double n) {
  const double nt = n * dt;
  const double s = std::sin(nt);
  const double c = std::cos(nt);
  const double n2 = n * n;

  // Gamma_r (integral of Phi_rv) and Gamma_v (== Phi_rv).
  const double gr00 = (1.0 - c) / n2;
  const double gr01 = 2.0 / n2 * (nt - s);
  const double gr10 = 2.0 / n2 * (s - nt);
  const double gr11 = (4.0 * (1.0 - c) - 1.5 * nt * nt) / n2;
  const double gr22 = (1.0 - c) / n2;
  const double gv00 = s / n;
  const double gv01 = 2.0 / n * (1.0 - c);
  const double gv10 = 2.0 / n * (c - 1.0);
  const double gv11 = (4.0 * s - 3.0 * nt) / n;
  const double gv22 = s / n;

  const double x = p.X(), y = p.Y(), z = p.Z();
  const double vx = v.X(), vy = v.Y(), vz = v.Z();

  const double x1 = (4.0 - 3.0 * c) * x + gv00 * vx + gv01 * vy + gr00 * a.X() + gr01 * a.Y();
  const double y1 = 6.0 * (s - nt) * x + y + gv10 * vx + gv11 * vy + gr10 * a.X() + gr11 * a.Y();
  const double z1 = c * z + gv22 * vz + gr22 * a.Z();
  const double vx1 = 3.0 * n * s * x + c * vx + 2.0 * s * vy + gv00 * a.X() + gv01 * a.Y();
  const double vy1 = 6.0 * n * (c - 1.0) * x - 2.0 * s * vx + (4.0 * c - 3.0) * vy + gv10 * a.X() + gv11 * a.Y();
  const double vz1 = -n * s * z + c * vz + gv22 * a.Z();

  p.Set(x1, y1, z1);
  v.Set(vx1, vy1, vz1);
}

class CwDynamicsSystem final : public gz::sim::System,
                               public gz::sim::ISystemConfigure,
                               public gz::sim::ISystemPreUpdate {
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    if (sdf) {
      this->n = sdf->Get<double>("n", this->n).first;
      this->maxAccel = sdf->Get<double>("max_accel", this->maxAccel).first;
      this->modelName = sdf->Get<std::string>("model_name", this->modelName).first;
      this->topicPrefix = sdf->Get<std::string>("topic_prefix", this->topicPrefix).first;
      this->initialVelocity = sdf->Get<gz::math::Vector3d>("initial_velocity", this->initialVelocity).first;
    }
    const std::string base = this->topicPrefix + "/model/" + this->modelName;
    this->node.Subscribe(base + "/command/accel", &CwDynamicsSystem::OnCommand, this);
    this->posePub = this->node.Advertise<gz::msgs::Pose>(base + "/pose");
    this->velocityPub = this->node.Advertise<gz::msgs::Vector3d>(base + "/velocity");
  }

  void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override {
    if (!this->model.Valid(ecm)) {
      this->model = gz::sim::Model(ecm.EntityByComponents(
          gz::sim::components::Model(), gz::sim::components::Name(this->modelName)));
      if (!this->model.Valid(ecm)) return;
    }
    // (Re)initialise from the model's spawn pose on the first update and after
    // a world reset (simulation time running backwards).
    if (!this->initialized || info.simTime < this->lastSimTime) {
      const auto pose = gz::sim::worldPose(this->model.Entity(), ecm);
      this->position = pose.Pos();
      this->rotation = pose.Rot();
      this->velocity = this->initialVelocity;
      this->initialized = true;
    }
    this->lastSimTime = info.simTime;
    if (info.paused) {
      this->Publish(info.simTime);
      return;
    }
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt <= 0.0) return;

    gz::math::Vector3d accel;
    {
      std::lock_guard<std::mutex> lock(this->commandMutex);
      accel = this->command;
    }
    CwStep(this->position, this->velocity, accel, dt, this->n);
    this->model.SetWorldPoseCmd(ecm, gz::math::Pose3d(this->position, this->rotation));
    this->Publish(info.simTime + info.dt);
  }

 private:
  void OnCommand(const gz::msgs::Vector3d &msg) {
    auto clamp = [this](double value) {
      if (!std::isfinite(value)) return 0.0;  // never forward NaN/inf to the dynamics
      return std::max(-this->maxAccel, std::min(this->maxAccel, value));
    };
    std::lock_guard<std::mutex> lock(this->commandMutex);
    this->command.Set(clamp(msg.x()), clamp(msg.y()), clamp(msg.z()));
  }

  void Publish(const std::chrono::steady_clock::duration &stamp) {
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(stamp);
    const auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(stamp - secs);
    gz::msgs::Pose pose;
    pose.mutable_header()->mutable_stamp()->set_sec(secs.count());
    pose.mutable_header()->mutable_stamp()->set_nsec(static_cast<int32_t>(nsecs.count()));
    pose.set_name(this->modelName);
    gz::msgs::Set(pose.mutable_position(), this->position);
    gz::msgs::Set(pose.mutable_orientation(), this->rotation);
    this->posePub.Publish(pose);

    gz::msgs::Vector3d velocityMsg;
    gz::msgs::Set(&velocityMsg, this->velocity);
    this->velocityPub.Publish(velocityMsg);
  }

  gz::sim::Model model{gz::sim::kNullEntity};
  gz::transport::Node node;
  gz::transport::Node::Publisher posePub;
  gz::transport::Node::Publisher velocityPub;
  std::mutex commandMutex;
  gz::math::Vector3d command{0, 0, 0};
  gz::math::Vector3d position{0, 0, 0};
  gz::math::Vector3d velocity{0, 0, 0};
  gz::math::Vector3d initialVelocity{0, 0, 0};
  gz::math::Quaterniond rotation{gz::math::Quaterniond::Identity};
  std::chrono::steady_clock::duration lastSimTime{0};
  bool initialized{false};
  double n{0.00113};
  double maxAccel{0.1};
  std::string modelName{"chaser"};
  std::string topicPrefix{"/world/leo_orbit"};
};

}  // namespace spaceforge_gazebo

GZ_ADD_PLUGIN(spaceforge_gazebo::CwDynamicsSystem, gz::sim::System,
              gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(spaceforge_gazebo::CwDynamicsSystem, "spaceforge_gazebo::CwDynamicsSystem")
