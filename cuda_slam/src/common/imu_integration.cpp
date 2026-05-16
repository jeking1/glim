#include "cuda_slam/common/imu_integration.hpp"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace cuda_slam {

void IMUIntegration::insert_imu(double stamp,
                                const Eigen::Vector3d& linear_acc,
                                const Eigen::Vector3d& angular_vel) {
  ImuMeasurement meas;
  meas.stamp = stamp;
  meas.linear_acceleration = linear_acc;
  meas.angular_velocity = angular_vel;

  if (!imu_queue_.empty() && stamp <= imu_queue_.back().stamp) {
    auto it = std::lower_bound(
        imu_queue_.begin(), imu_queue_.end(), stamp,
        [](const ImuMeasurement& m, double t) { return m.stamp < t; });
    imu_queue_.insert(it, meas);
  } else {
    imu_queue_.push_back(meas);
  }

  while (imu_queue_.size() > max_queue_size_) {
    imu_queue_.pop_front();
  }
}

void IMUIntegration::clear() { imu_queue_.clear(); }

NavState IMUIntegration::integrate_single_measurement(
    const NavState& state, const ImuMeasurement& imu,
    const ImuBias& bias, const Eigen::Vector3d& gravity, double dt) {
  if (dt <= 0.0) {
    return state;
  }

  const Eigen::Vector3d accel_unbiased =
      imu.linear_acceleration - bias.accel_bias;
  const Eigen::Vector3d gyro_unbiased =
      imu.angular_velocity - bias.gyro_bias;

  NavState next;
  next.stamp = state.stamp + dt;

  const Eigen::Quaterniond dq =
      Eigen::Quaterniond(1.0, 0.5 * gyro_unbiased.x() * dt,
                         0.5 * gyro_unbiased.y() * dt,
                         0.5 * gyro_unbiased.z() * dt)
          .normalized();
  next.orientation = (state.orientation * dq).normalized();

  const Eigen::Vector3d accel_world =
      state.orientation * accel_unbiased + gravity;

  next.velocity = state.velocity + accel_world * dt;

  next.position = state.position + state.velocity * dt +
                  0.5 * accel_world * dt * dt;

  return next;
}

NavState IMUIntegration::integrate_imu(double t0, double t1,
                                       const ImuBias& bias) const {
  NavState state;
  state.stamp = t0;

  if (imu_queue_.empty() || t1 <= t0) {
    return state;
  }

  auto it_begin = std::lower_bound(
      imu_queue_.begin(), imu_queue_.end(), t0,
      [](const ImuMeasurement& m, double t) { return m.stamp < t; });

  auto it_end = std::upper_bound(
      imu_queue_.begin(), imu_queue_.end(), t1,
      [](double t, const ImuMeasurement& m) { return t < m.stamp; });

  if (it_begin == imu_queue_.end() || it_begin == it_end) {
    return state;
  }

  if (it_begin != imu_queue_.begin()) {
    --it_begin;
  }

  if (it_end != imu_queue_.end()) {
    ++it_end;
  }

  for (auto it = it_begin; it != it_end; ++it) {
    auto next_it = std::next(it);
    if (next_it == it_end) {
      break;
    }

    double seg_t0 = std::max(it->stamp, t0);
    double seg_t1 = std::min(next_it->stamp, t1);

    if (seg_t1 <= seg_t0) {
      continue;
    }

    double dt = seg_t1 - seg_t0;

    ImuMeasurement interp;
    interp.stamp = seg_t0;

    double alpha = 0.0;
    double denom = next_it->stamp - it->stamp;
    if (denom > 1e-12) {
      alpha = (seg_t0 - it->stamp) / denom;
    }
    alpha = std::clamp(alpha, 0.0, 1.0);

    interp.linear_acceleration =
        it->linear_acceleration +
        alpha * (next_it->linear_acceleration - it->linear_acceleration);
    interp.angular_velocity =
        it->angular_velocity +
        alpha * (next_it->angular_velocity - it->angular_velocity);

    state = integrate_single_measurement(state, interp, bias, gravity_, dt);
  }

  state.stamp = t1;
  return state;
}

std::vector<NavState> IMUIntegration::predict_states(
    double t0, double t1, const NavState& initial_navstate,
    const ImuBias& bias) const {
  std::vector<NavState> states;
  states.reserve(imu_queue_.size() + 2);
  states.push_back(initial_navstate);

  if (imu_queue_.empty() || t1 <= t0) {
    return states;
  }

  auto it_begin = std::lower_bound(
      imu_queue_.begin(), imu_queue_.end(), t0,
      [](const ImuMeasurement& m, double t) { return m.stamp < t; });

  auto it_end = std::upper_bound(
      imu_queue_.begin(), imu_queue_.end(), t1,
      [](double t, const ImuMeasurement& m) { return t < m.stamp; });

  if (it_begin == imu_queue_.end()) {
    return states;
  }

  if (it_begin != imu_queue_.begin()) {
    --it_begin;
  }

  if (it_end != imu_queue_.end()) {
    ++it_end;
  }

  NavState current = initial_navstate;

  for (auto it = it_begin; it != it_end; ++it) {
    auto next_it = std::next(it);
    if (next_it == it_end) {
      break;
    }

    double seg_t0 = std::max(it->stamp, t0);
    double seg_t1 = std::min(next_it->stamp, t1);

    if (seg_t1 <= seg_t0) {
      continue;
    }

    double dt = seg_t1 - seg_t0;

    ImuMeasurement interp;
    interp.stamp = seg_t0;

    double alpha = 0.0;
    double denom = next_it->stamp - it->stamp;
    if (denom > 1e-12) {
      alpha = (seg_t0 - it->stamp) / denom;
    }
    alpha = std::clamp(alpha, 0.0, 1.0);

    interp.linear_acceleration =
        it->linear_acceleration +
        alpha * (next_it->linear_acceleration - it->linear_acceleration);
    interp.angular_velocity =
        it->angular_velocity +
        alpha * (next_it->angular_velocity - it->angular_velocity);

    current = integrate_single_measurement(current, interp, bias, gravity_, dt);
    states.push_back(current);
  }

  return states;
}

} // namespace cuda_slam