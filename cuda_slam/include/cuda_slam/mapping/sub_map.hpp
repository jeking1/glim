#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

#include "cuda_slam/odometry/estimation_frame.hpp"

namespace cuda_slam {
namespace cuda {
class CudaVoxelGrid;
}

struct SubMap {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<SubMap>;
  using ConstPtr = std::shared_ptr<const SubMap>;

  long id = -1;
  Eigen::Isometry3d T_world_origin = Eigen::Isometry3d::Identity();

  std::vector<EstimationFrame::ConstPtr> odom_frames;

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      merged_points;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      merged_normals;

  std::shared_ptr<cuda::CudaVoxelGrid> voxel_grid;

  void merge_frames() {
    merged_points.clear();
    merged_normals.clear();

    size_t total_points = 0;
    for (const auto& frame : odom_frames) {
      total_points += frame->frame.points.size();
    }

    merged_points.reserve(total_points);
    merged_normals.reserve(total_points);

    for (const auto& frame : odom_frames) {
      for (size_t i = 0; i < frame->frame.points.size(); ++i) {
        Eigen::Vector4d pt_world = frame->T_world_lidar * frame->frame.points[i];
        Eigen::Vector4d pt_local = T_world_origin.inverse() * pt_world;
        merged_points.push_back(pt_local);

        if (i < frame->frame.normals.size()) {
          Eigen::Isometry3d T_origin_frame =
              T_world_origin.inverse() * frame->T_world_lidar;
          Eigen::Vector4d n_local =
              T_origin_frame.linear() * frame->frame.normals[i].head<3>().homogeneous();
          n_local.w() = frame->frame.normals[i].w();
          merged_normals.push_back(n_local);
        }
      }
    }
  }

  size_t point_count() const { return merged_points.size(); }

  size_t frame_count() const { return odom_frames.size(); }
};

} // namespace cuda_slam