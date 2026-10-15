#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/msgs/vector3d.pb.h>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/LinearVelocity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace spaceforge_gazebo {
class CwDynamicsSystem final : public gz::sim::System,
  public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate {
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    if (sdf && sdf->HasElement("n")) this->n = sdf->Get<double>("n");
    if (sdf && sdf->HasElement("max_accel")) this->maxAccel = sdf->Get<double>("max_accel");
    if (sdf && sdf->HasElement("model_name")) this->modelName = sdf->Get<std::string>("model_name");
    this->node.Subscribe("/model/" + this->modelName + "/command/accel", &CwDynamicsSystem::OnCommand, this);
  }

  void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override {
    if (info.paused || !this->model.Valid(ecm)) {
      if (!this->model.Valid(ecm)) this->model = gz::sim::Model(ecm.EntityByComponents(gz::sim::components::Model(), gz::sim::components::Name(this->modelName)));
      return;
    }
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt <= 0.0) return;
    auto pose = ecm.Component<gz::sim::components::Pose>(this->model.Entity());
    auto vel = ecm.Component<gz::sim::components::LinearVelocity>(this->model.Entity());
    if (!pose || !vel) return;
    auto p = pose->Data().Pos();
    auto v = vel->Data();
    const auto a = this->command;
    const double ax = 3.0*n*n*p.X() + 2.0*n*v.Y() + a.X();
    const double ay = -2.0*n*v.X() + a.Y();
    const double az = -n*n*p.Z() + a.Z();
    v += gz::math::Vector3d(ax, ay, az) * dt;
    p += v * dt;
    pose->Data().Set(gz::math::Pose3d(pose->Data().Rot(), p));
    vel->Data() = v;
  }

 private:
  void OnCommand(const gz::msgs::Vector3d &msg) {
    this->command.Set(std::clamp(msg.x(), -maxAccel, maxAccel), std::clamp(msg.y(), -maxAccel, maxAccel), std::clamp(msg.z(), -maxAccel, maxAccel));
  }
  gz::sim::Model model;
  gz::transport::Node node;
  gz::math::Vector3d command{0, 0, 0};
  double n{0.00113};
  double maxAccel{1.0};
  std::string modelName{"chaser"};
};
}

GZ_ADD_PLUGIN(spaceforge_gazebo::CwDynamicsSystem, gz::sim::System, spaceforge_gazebo::CwDynamicsSystem::ISystemConfigure, spaceforge_gazebo::CwDynamicsSystem::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(spaceforge_gazebo::CwDynamicsSystem, "spaceforge_gazebo::CwDynamicsSystem")
