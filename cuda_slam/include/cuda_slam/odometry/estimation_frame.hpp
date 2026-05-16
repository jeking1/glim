#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

#include "cuda_slam/preprocess/preprocessed_frame.hpp"

namespace cuda_slam {
namespace cuda {
class CudaVoxelGrid;
}

struct PointCloudCPU {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> normals;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covs;
};

struct EstimationFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<EstimationFrame>;
  using ConstPtr = std::shared_ptr<const EstimationFrame>;

  long id = -1;
  double stamp = 0.0;

  Eigen::Isometry3d T_lidar_imu = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_world_imu = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();

  Eigen::Vector3d v_world_imu = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 1> imu_bias = Eigen::Matrix<double, 6, 1>::Zero();

  PreprocessedFrame::ConstPtr raw_frame;

  PointCloudCPU frame;

  std::vector<std::shared_ptr<cuda::CudaVoxelGrid>> voxelmaps;
};

} // namespace cuda_slam