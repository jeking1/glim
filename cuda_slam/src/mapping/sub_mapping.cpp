#include "cuda_slam/mapping/sub_mapping.hpp"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace cuda_slam {

namespace {

Eigen::Isometry3d compute_relative_transform(const Eigen::Isometry3d& from,
                                              const Eigen::Isometry3d& to) {
  return from.inverse() * to;
}

double compute_translation_distance(const Eigen::Isometry3d& T) {
  return T.translation().norm();
}

double compute_rotation_angle(const Eigen::Isometry3d& T) {
  Eigen::AngleAxisd aa(T.linear());
  return std::abs(aa.angle());
}

} // namespace

SubMapping::SubMapping(const SubMappingParams& params) : params_(params) {
  spdlog::info("SubMapping initialized: max_frames_per_submap={}, "
               "submap_delta_trans={}, voxel_resolution={}",
               params.max_frames_per_submap, params.submap_delta_trans,
               params.voxel_resolution);
}

void SubMapping::insert_frame(EstimationFrame::ConstPtr frame) {
  if (!frame) {
    spdlog::warn("SubMapping::insert_frame called with null frame");
    return;
  }

  pending_frames_.push_back(frame);
  ++frames_since_last_submap_;

  bool is_kf = is_keyframe(frame);
  if (is_kf) {
    keyframe_candidates_.push_back(frame);
    last_keyframe_pose_ = frame->T_world_lidar;
  }

  bool should_create = false;

  if (static_cast<int>(pending_frames_.size()) >= params_.max_frames_per_submap) {
    should_create = true;
  }

  if (!should_create && keyframe_candidates_.size() >= 2) {
    const auto& first_kf = keyframe_candidates_.front();
    const auto& last_kf = keyframe_candidates_.back();
    Eigen::Isometry3d T_rel =
        compute_relative_transform(first_kf->T_world_lidar, last_kf->T_world_lidar);

    double trans = compute_translation_distance(T_rel);
    double rot = compute_rotation_angle(T_rel);

    if (trans > params_.submap_delta_trans || rot > params_.submap_delta_rot) {
      should_create = true;
    }
  }

  if (should_create) {
    create_submap();
  }
}

std::vector<SubMap::Ptr> SubMapping::get_submaps() const {
  return submaps_;
}

std::vector<SubMap::Ptr> SubMapping::submit_end_of_sequence() {
  if (!pending_frames_.empty()) {
    create_submap();
  }

  auto result = std::move(submaps_);
  submaps_.clear();
  spdlog::info("SubMapping::submit_end_of_sequence: {} submaps finalized",
               result.size());
  return result;
}

size_t SubMapping::pending_frame_count() const {
  return pending_frames_.size();
}

size_t SubMapping::submap_count() const {
  return submaps_.size();
}

bool SubMapping::is_keyframe(EstimationFrame::ConstPtr frame) const {
  if (keyframe_candidates_.empty()) {
    return true;
  }

  Eigen::Isometry3d T_rel = compute_relative_transform(
      last_keyframe_pose_, frame->T_world_lidar);

  double trans = compute_translation_distance(T_rel);
  double rot = compute_rotation_angle(T_rel);

  return trans > params_.keyframe_delta_trans ||
         rot > params_.keyframe_delta_rot;
}

void SubMapping::create_submap() {
  if (pending_frames_.empty()) {
    return;
  }

  auto submap = std::make_shared<SubMap>();
  submap->id = submap_counter_++;

  if (!keyframe_candidates_.empty()) {
    submap->T_world_origin = keyframe_candidates_.front()->T_world_lidar;
  } else {
    submap->T_world_origin = pending_frames_.front()->T_world_lidar;
  }

  select_keyframes_for_submap(*submap);

  if (submap->odom_frames.empty()) {
    for (const auto& f : pending_frames_) {
      submap->odom_frames.push_back(f);
    }
  }

  submap->merge_frames();

  spdlog::info("SubMap {} created: {} frames, {} keyframes, {} points",
               submap->id, pending_frames_.size(),
               submap->odom_frames.size(), submap->merged_points.size());

  submaps_.push_back(submap);

  pending_frames_.clear();
  keyframe_candidates_.clear();
  frames_since_last_submap_ = 0;
}

void SubMapping::select_keyframes_for_submap(SubMap& submap) const {
  if (keyframe_candidates_.empty()) {
    return;
  }

  submap.odom_frames.push_back(keyframe_candidates_.front());

  for (size_t i = 1; i < keyframe_candidates_.size(); ++i) {
    const auto& last_added = submap.odom_frames.back();
    const auto& candidate = keyframe_candidates_[i];

    Eigen::Isometry3d T_rel = compute_relative_transform(
        last_added->T_world_lidar, candidate->T_world_lidar);

    double trans = compute_translation_distance(T_rel);
    double rot = compute_rotation_angle(T_rel);

    if (trans > params_.keyframe_delta_trans * 0.5 ||
        rot > params_.keyframe_delta_rot * 0.5) {
      submap.odom_frames.push_back(candidate);
    }

    if (static_cast<int>(submap.odom_frames.size()) >=
        params_.max_keyframes_per_submap) {
      break;
    }
  }

  if (static_cast<int>(submap.odom_frames.size()) <
      params_.max_keyframes_per_submap) {
    const auto& last_added = submap.odom_frames.back();

    for (auto it = pending_frames_.rbegin(); it != pending_frames_.rend(); ++it) {
      if (static_cast<int>(submap.odom_frames.size()) >=
          params_.max_keyframes_per_submap) {
        break;
      }

      if ((*it)->id == last_added->id) continue;

      bool already_in = false;
      for (const auto& f : submap.odom_frames) {
        if (f->id == (*it)->id) {
          already_in = true;
          break;
        }
      }
      if (already_in) continue;

      Eigen::Isometry3d T_rel = compute_relative_transform(
          last_added->T_world_lidar, (*it)->T_world_lidar);

      double trans = compute_translation_distance(T_rel);
      double rot = compute_rotation_angle(T_rel);

      if (trans > params_.keyframe_delta_trans * 0.5 ||
          rot > params_.keyframe_delta_rot * 0.5) {
        submap.odom_frames.push_back(*it);
      }
    }
  }
}

} // namespace cuda_slam