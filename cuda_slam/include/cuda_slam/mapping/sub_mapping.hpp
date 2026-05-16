#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <memory>
#include <vector>

#include "cuda_slam/mapping/sub_map.hpp"
#include "cuda_slam/odometry/estimation_frame.hpp"

namespace cuda_slam {

struct SubMappingParams {
  int max_frames_per_submap = 100;
  double submap_delta_trans = 10.0;
  double submap_delta_rot = 0.5;
  double keyframe_delta_trans = 1.0;
  double keyframe_delta_rot = 0.2;
  int max_keyframes_per_submap = 20;
  double voxel_resolution = 0.5;
};

class SubMapping {
public:
  explicit SubMapping(const SubMappingParams& params = SubMappingParams());

  void insert_frame(EstimationFrame::ConstPtr frame);

  std::vector<SubMap::Ptr> get_submaps() const;

  std::vector<SubMap::Ptr> submit_end_of_sequence();

  size_t pending_frame_count() const;

  size_t submap_count() const;

private:
  bool is_keyframe(EstimationFrame::ConstPtr frame) const;
  void create_submap();
  void select_keyframes_for_submap(SubMap& submap) const;

  SubMappingParams params_;

  std::deque<EstimationFrame::ConstPtr> pending_frames_;
  std::vector<EstimationFrame::ConstPtr> keyframe_candidates_;

  std::vector<SubMap::Ptr> submaps_;

  long submap_counter_ = 0;

  Eigen::Isometry3d last_keyframe_pose_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d accumulated_pose_ = Eigen::Isometry3d::Identity();
  size_t frames_since_last_submap_ = 0;
};

} // namespace cuda_slam