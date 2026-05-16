#include "cuda_slam/common/cloud_deskewing.hpp"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace cuda_slam {

Eigen::Isometry3d CloudDeskewing::interpolate_pose(
    double target_time,
    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
    const std::vector<double>& times) {
  if (poses.empty() || times.empty() || poses.size() != times.size()) {
    return Eigen::Isometry3d::Identity();
  }

  if (poses.size() == 1) {
    return poses[0];
  }

  if (target_time <= times.front()) {
    return poses.front();
  }

  if (target_time >= times.back()) {
    return poses.back();
  }

  auto it = std::upper_bound(times.begin(), times.end(), target_time);
  if (it == times.begin()) {
    return poses.front();
  }
  if (it == times.end()) {
    return poses.back();
  }

  std::size_t idx_after = static_cast<std::size_t>(it - times.begin());
  std::size_t idx_before = idx_after - 1;

  double t_before = times[idx_before];
  double t_after = times[idx_after];
  double dt = t_after - t_before;

  if (dt < 1e-12) {
    return poses[idx_before];
  }

  double alpha = (target_time - t_before) / dt;
  alpha = std::clamp(alpha, 0.0, 1.0);

  const Eigen::Isometry3d& T0 = poses[idx_before];
  const Eigen::Isometry3d& T1 = poses[idx_after];

  Eigen::Isometry3d T_interp = Eigen::Isometry3d::Identity();

  const Eigen::Quaterniond q0(T0.linear());
  const Eigen::Quaterniond q1(T1.linear());
  const Eigen::Quaterniond q_interp = q0.slerp(alpha, q1);

  T_interp.linear() = q_interp.toRotationMatrix();
  T_interp.translation() =
      T0.translation() + alpha * (T1.translation() - T0.translation());

  return T_interp;
}

void CloudDeskewing::deskew(
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    const std::vector<double>& times,
    double scan_stamp,
    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& imu_poses,
    const std::vector<double>& imu_times,
    const Eigen::Isometry3d& T_imu_lidar,
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& deskewed_points) const {
  deskewed_points.clear();

  if (points.empty()) {
    spdlog::warn("CloudDeskewing: no points to deskew");
    return;
  }

  if (imu_poses.size() < 2 || imu_times.size() < 2) {
    spdlog::warn(
        "CloudDeskewing: insufficient IMU poses ({}), copying points as-is",
        imu_poses.size());
    deskewed_points = points;
    return;
  }

  deskewed_points.reserve(points.size());

  const Eigen::Isometry3d T_scan_start =
      interpolate_pose(scan_stamp, imu_poses, imu_times);

  const std::size_t n = points.size();
  const bool have_per_point_times = (times.size() == n);

  for (std::size_t i = 0; i < n; ++i) {
    const Eigen::Vector4d& pt = points[i];

    double pt_time = scan_stamp;
    if (have_per_point_times) {
      pt_time = times[i];
    }

    const Eigen::Isometry3d T_i = interpolate_pose(pt_time, imu_poses, imu_times);
    const Eigen::Isometry3d T_delta =
        T_scan_start.inverse() * T_i;

    const Eigen::Vector3d pt_xyz = pt.head<3>();
    const Eigen::Vector3d deskewed_xyz = T_delta * pt_xyz;

    Eigen::Vector4d deskewed_pt;
    deskewed_pt.head<3>() = deskewed_xyz;
    deskewed_pt.w() = pt.w();
    deskewed_points.push_back(deskewed_pt);
  }

  spdlog::debug("CloudDeskewing: deskewed {} points", deskewed_points.size());
}

} // namespace cuda_slam