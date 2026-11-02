#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/imu.pb.h>
#include <gz/msgs/laserscan.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/vector3d.pb.h>
#include <gz/transport/Node.hh>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

namespace spaceforge_gazebo {
class SimBridge final : public rclcpp::Node {
 public:
  SimBridge() : Node("sim_bridge") {
    scenario_ = declare_parameter<std::string>("scenario", "leo_orbit");
    // use_sim_time is declared by rclcpp itself; re-declaring it throws
    // ParameterAlreadyDeclaredException at startup. Launch sets it.
    gz_prefix_ = declare_parameter<std::string>("gz_prefix", "/world/leo_orbit");
    ros_prefix_ = declare_parameter<std::string>("ros_prefix", "spaceforge");
    mass_kg_ = declare_parameter<double>("chaser_mass_kg", 1.0);
    odom_frame_ = declare_parameter<std::string>("odom_frame", "target_lvlh");
    base_frame_ = declare_parameter<std::string>("base_frame", "chaser_base");
    imu_frame_ = declare_parameter<std::string>("imu_frame", "chaser_imu");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "chaser_camera");
    lidar_frame_ = declare_parameter<std::string>("lidar_frame", "chaser_lidar");
    clock_rate_ = declare_parameter<double>("clock_rate_hz", 100.0);
    odom_rate_ = declare_parameter<double>("odometry_rate_hz", 50.0);
    imu_rate_ = declare_parameter<double>("imu_rate_hz", 50.0);
    camera_rate_ = declare_parameter<double>("camera_rate_hz", 10.0);
    lidar_rate_ = declare_parameter<double>("lidar_rate_hz", 10.0);
    auto sensor_qos = rclcpp::SensorDataQoS();
    clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>(ros_prefix_ + "/clock", rclcpp::QoS(10).reliable());
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(ros_prefix_ + "/chaser/odometry", sensor_qos);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(ros_prefix_ + "/chaser/imu", sensor_qos);
    camera_pub_ = create_publisher<sensor_msgs::msg::Image>(ros_prefix_ + "/chaser/camera/image_raw", sensor_qos);
    lidar_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(ros_prefix_ + "/chaser/lidar/scan", sensor_qos);
    command_sub_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      ros_prefix_ + "/chaser/cmd_wrench", rclcpp::QoS(10).reliable(),
      [this](geometry_msgs::msg::WrenchStamped::ConstSharedPtr msg) { this->onWrench(*msg); });
    command_pub_ = gz_node_.Advertise<gz::msgs::Vector3d>(gz_prefix_ + "/model/chaser/command/accel");
    gz_node_.Subscribe(gz_prefix_ + "/clock", &SimBridge::onClock, this);
    gz_node_.Subscribe(gz_prefix_ + "/model/chaser/pose", &SimBridge::onPose, this);
    gz_node_.Subscribe(gz_prefix_ + "/model/chaser/velocity", &SimBridge::onVelocity, this);
    gz_node_.Subscribe(gz_prefix_ + "/model/chaser/link/body/sensor/imu_placeholder/imu", &SimBridge::onImu, this);
    gz_node_.Subscribe(gz_prefix_ + "/model/chaser/link/body/sensor/camera_placeholder/image", &SimBridge::onImage, this);
    gz_node_.Subscribe(gz_prefix_ + "/model/chaser/link/body/sensor/lidar_placeholder/scan", &SimBridge::onScan, this);
  }

 private:
  static builtin_interfaces::msg::Time rosTime(const gz::msgs::Time &t) {
    builtin_interfaces::msg::Time out;
    out.sec = static_cast<int32_t>(t.sec());
    out.nanosec = static_cast<uint32_t>(t.nsec());
    return out;
  }
  bool due(int64_t stamp, int64_t &last, double hz) {
    if (hz <= 0.0 || last < 0 || stamp - last >= static_cast<int64_t>(1e9 / hz)) { last = stamp; return true; }
    return false;
  }
  void onClock(const gz::msgs::Clock &msg) {
    const auto stamp = rosTime(msg.sim());
    const int64_t ns = static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
    if (!due(ns, last_clock_ns_, clock_rate_)) return;
    rosgraph_msgs::msg::Clock out; out.clock = stamp; clock_pub_->publish(out);
  }
  void onPose(const gz::msgs::Pose &msg) {
    const auto stamp = rosTime(msg.header().stamp());
    const int64_t ns = static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
    if (!due(ns, last_odom_ns_, odom_rate_)) return;
    nav_msgs::msg::Odometry out; out.header.stamp = stamp; out.header.frame_id = odom_frame_; out.child_frame_id = base_frame_;
    out.pose.pose.position.x = msg.position().x(); out.pose.pose.position.y = msg.position().y(); out.pose.pose.position.z = msg.position().z();
    out.pose.pose.orientation.x = msg.orientation().x(); out.pose.pose.orientation.y = msg.orientation().y(); out.pose.pose.orientation.z = msg.orientation().z(); out.pose.pose.orientation.w = msg.orientation().w(); out.twist.twist.linear.x = last_velocity_.x(); out.twist.twist.linear.y = last_velocity_.y(); out.twist.twist.linear.z = last_velocity_.z();
    odom_pub_->publish(out);
  }
  void onVelocity(const gz::msgs::Vector3d &msg) { last_velocity_ = msg; }
  void onImu(const gz::msgs::IMU &msg) {
    const auto stamp = rosTime(msg.header().stamp()); const int64_t ns = static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec; if (!due(ns, last_imu_ns_, imu_rate_)) return;
    sensor_msgs::msg::Imu out; out.header.stamp = stamp; out.header.frame_id = imu_frame_; out.linear_acceleration.x = msg.linear_acceleration().x(); out.linear_acceleration.y = msg.linear_acceleration().y(); out.linear_acceleration.z = msg.linear_acceleration().z(); out.angular_velocity.x = msg.angular_velocity().x(); out.angular_velocity.y = msg.angular_velocity().y(); out.angular_velocity.z = msg.angular_velocity().z(); imu_pub_->publish(out);
  }
  void onImage(const gz::msgs::Image &msg) {
    const auto stamp = rosTime(msg.header().stamp()); const int64_t ns = static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec; if (!due(ns, last_camera_ns_, camera_rate_)) return;
    sensor_msgs::msg::Image out; out.header.stamp = stamp; out.header.frame_id = camera_frame_; out.width = msg.width(); out.height = msg.height(); out.encoding = "rgb8"; out.step = msg.width() * 3; out.data.assign(msg.data().begin(), msg.data().end()); camera_pub_->publish(out);
  }
  void onScan(const gz::msgs::LaserScan &msg) {
    const auto stamp = rosTime(msg.header().stamp()); const int64_t ns = static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec; if (!due(ns, last_lidar_ns_, lidar_rate_)) return;
    sensor_msgs::msg::LaserScan out; out.header.stamp = stamp; out.header.frame_id = lidar_frame_; out.angle_min = msg.angle_min(); out.angle_max = msg.angle_max(); out.angle_increment = msg.angle_step(); out.range_min = msg.range_min(); out.range_max = msg.range_max(); for (int i = 0; i < msg.ranges_size(); ++i) out.ranges.push_back(msg.ranges(i)); lidar_pub_->publish(out);
  }
  void onWrench(const geometry_msgs::msg::WrenchStamped &msg) {
    gz::msgs::Vector3d command; command.set_x(msg.wrench.force.x / mass_kg_); command.set_y(msg.wrench.force.y / mass_kg_); command.set_z(msg.wrench.force.z / mass_kg_); command_pub_.Publish(command);
  }
  gz::transport::Node gz_node_;
  gz::transport::Node::Publisher command_pub_;
  std::string scenario_, gz_prefix_, ros_prefix_, odom_frame_, base_frame_, imu_frame_, camera_frame_, lidar_frame_;
  double mass_kg_, clock_rate_, odom_rate_, imu_rate_, camera_rate_, lidar_rate_;
  int64_t last_clock_ns_{-1}, last_odom_ns_{-1}, last_imu_ns_{-1}, last_camera_ns_{-1}, last_lidar_ns_{-1};
  gz::msgs::Vector3d last_velocity_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr lidar_pub_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr command_sub_;
};
}

int main(int argc, char **argv) { rclcpp::init(argc, argv); rclcpp::spin(std::make_shared<spaceforge_gazebo::SimBridge>()); rclcpp::shutdown(); return 0; }
