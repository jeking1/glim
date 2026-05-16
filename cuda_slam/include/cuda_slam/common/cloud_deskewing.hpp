#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace cuda_slam {

class CloudDeskewing {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CloudDeskewing() = default;

  void deskew(
      const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
      const std::vector<double>& times,
      double scan_stamp,
      const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& imu_poses,
      const std::vector<double>& imu_times,
      const Eigen::Isometry3d& T_imu_lidar,
      std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& deskewed_points) const;

private:
  [[nodiscard]] static Eigen::Isometry3d interpolate_pose(
      double target_time,
      const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses,
      const std::vector<double>& times);
};

} // namespace cuda_slam