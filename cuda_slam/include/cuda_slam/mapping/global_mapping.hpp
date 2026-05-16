#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cuda_slam/mapping/sub_map.hpp"
#include "cuda_slam/odometry/estimation_frame.hpp"

namespace cuda_slam {

struct PoseGraphEdge {
  int from_id;
  int to_id;
  Eigen::Isometry3d relative_pose;
  Eigen::Matrix<double, 6, 6> information;
  bool is_loop_closure = false;
};

struct GlobalMappingParams {
  double min_overlap_ratio = 0.3;
  double loop_closure_search_radius = 20.0;
  double icp_max_correspondence_distance = 2.0;
  int icp_max_iterations = 30;
  double icp_transformation_epsilon = 1e-6;
  int lm_max_iterations = 20;
  double lm_lambda_initial = 1e-3;
  double lm_lambda_factor = 10.0;
  double lm_convergence_threshold = 1e-6;
  double voxel_resolution = 0.5;
};

class GlobalMapping {
public:
  explicit GlobalMapping(const GlobalMappingParams& params = GlobalMappingParams());

  void insert_submap(SubMap::Ptr submap);

  std::vector<std::pair<int, int>> find_overlapping_submaps(
      double min_overlap) const;

  bool optimize();

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
  export_points() const;

  const std::vector<SubMap::Ptr>& get_submaps() const;

  const std::vector<PoseGraphEdge>& get_edges() const;

  size_t submap_count() const;

  size_t edge_count() const;

private:
  void build_odometry_edges();
  void detect_loop_closures();
  double compute_overlap_ratio(const SubMap& a, const SubMap& b) const;

  Eigen::Matrix<double, 6, 1> compute_relative_error(
      const Eigen::Isometry3d& T_i,
      const Eigen::Isometry3d& T_j,
      const Eigen::Isometry3d& T_ij) const;

  void build_linear_system(
      const std::vector<Eigen::Isometry3d>& poses,
      Eigen::SparseMatrix<double>& H,
      Eigen::VectorXd& b,
      double& total_error) const;

  GlobalMappingParams params_;

  std::vector<SubMap::Ptr> submaps_;

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      optimized_poses_;

  std::vector<PoseGraphEdge> edges_;

  int next_submap_id_ = 0;
};

} // namespace cuda_slam