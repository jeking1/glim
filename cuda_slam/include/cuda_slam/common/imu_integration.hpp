#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <memory>
#include <vector>

namespace cuda_slam {

struct ImuMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp = 0.0;
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

struct ImuBias {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
};

struct NavState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double stamp = 0.0;
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

  [[nodiscard]] Eigen::Isometry3d pose() const {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = orientation.toRotationMatrix();
    T.translation() = position;
    return T;
  }
};

class IMUIntegration {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUIntegration() = default;

  explicit IMUIntegration(std::size_t max_queue_size)
      : max_queue_size_(max_queue_size) {}

  void insert_imu(double stamp, const Eigen::Vector3d& linear_acc,
                  const Eigen::Vector3d& angular_vel);

  [[nodiscard]] NavState integrate_imu(double t0, double t1,
                                       const ImuBias& bias) const;

  [[nodiscard]] std::vector<NavState> predict_states(
      double t0, double t1, const NavState& initial_navstate,
      const ImuBias& bias) const;

  void clear();

  [[nodiscard]] std::size_t size() const { return imu_queue_.size(); }
  [[nodiscard]] bool empty() const { return imu_queue_.empty(); }

  void set_gravity(const Eigen::Vector3d& g) { gravity_ = g; }
  [[nodiscard]] const Eigen::Vector3d& gravity() const { return gravity_; }

  void set_max_queue_size(std::size_t n) { max_queue_size_ = n; }
  [[nodiscard]] std::size_t max_queue_size() const { return max_queue_size_; }

  [[nodiscard]] const std::deque<ImuMeasurement>& queue() const {
    return imu_queue_;
  }

private:
  static NavState integrate_single_measurement(
      const NavState& state, const ImuMeasurement& imu,
      const ImuBias& bias, const Eigen::Vector3d& gravity,
      double dt);

  std::deque<ImuMeasurement> imu_queue_;
  std::size_t max_queue_size_ = 1000;
  Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, -9.81);
};

} // namespace cuda_slam