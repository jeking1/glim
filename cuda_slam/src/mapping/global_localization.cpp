#include "cuda_slam/mapping/global_localization.hpp"

#include "cuda_slam/cuda/cuda_voxel_grid.hpp"
#include "cuda_slam/cuda/cuda_icp.hpp"

#include <spdlog/spdlog.h>

#include <cstring>
#include <algorithm>
#include <cmath>

namespace cuda_slam {

GlobalLocalization::GlobalLocalization(const GlobalLocalizationParams& params)
    : params_(params),
      map_loaded_(false),
      map_points_count_(0),
      d_map_points_(nullptr),
      d_map_raw_(nullptr) {
  icp_ = std::make_unique<cuda::CudaICP>();
  icp_->set_max_correspondence_distance(
      static_cast<float>(params_.icp_max_correspondence_distance));
  icp_->set_convergence_threshold(
      static_cast<float>(params_.icp_transformation_epsilon));
}

GlobalLocalization::~GlobalLocalization() {
  release_map_resources();
}

bool GlobalLocalization::load_map(
    const std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d>>& points) {
  if (points.empty()) {
    spdlog::error("[GlobalLocalization] Empty map points");
    return false;
  }

  map_points_count_ = static_cast<int>(points.size());

  std::vector<float> flat_points(map_points_count_ * 4);
  for (int i = 0; i < map_points_count_; i++) {
    flat_points[i * 4 + 0] = static_cast<float>(points[i].x());
    flat_points[i * 4 + 1] = static_cast<float>(points[i].y());
    flat_points[i * 4 + 2] = static_cast<float>(points[i].z());
    flat_points[i * 4 + 3] = 1.0f;
  }

  upload_points(flat_points.data(), map_points_count_);
  return build_map_voxel_grids();
}

bool GlobalLocalization::load_map(const float* points, int num_points) {
  if (num_points <= 0) {
    spdlog::error("[GlobalLocalization] Invalid map point count: {}", num_points);
    return false;
  }

  map_points_count_ = num_points;
  upload_points(points, num_points);
  return build_map_voxel_grids();
}

void GlobalLocalization::upload_points(const float* points, int num_points) {
  release_map_resources();

  cudaMalloc(&d_map_raw_, num_points * 4 * sizeof(float));
  cudaMalloc(&d_map_points_, num_points * sizeof(float4));

  cudaMemcpy(d_map_raw_, points, num_points * 4 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_map_points_, points, num_points * sizeof(float4),
             cudaMemcpyHostToDevice);

  spdlog::info("[GlobalLocalization] Uploaded {} map points to GPU", num_points);
}

bool GlobalLocalization::build_map_voxel_grids() {
  map_voxel_grids_.clear();

  for (int level = 0; level < params_.voxelmap_levels; level++) {
    float resolution = static_cast<float>(
        params_.voxel_resolution_base *
        std::pow(params_.voxelmap_scaling_factor, level));

    auto grid = std::make_unique<cuda::CudaVoxelGrid>(resolution);
    grid->insert(d_map_points_, map_points_count_);

    int nv = grid->num_voxels();
    spdlog::info("[GlobalLocalization] Level {} voxel grid: resolution={:.3f}, "
                 "{} voxels", level, resolution, nv);

    if (nv == 0) {
      spdlog::error("[GlobalLocalization] Level {} voxel grid is empty", level);
      return false;
    }

    map_voxel_grids_.push_back(std::move(grid));
  }

  map_loaded_ = true;
  spdlog::info("[GlobalLocalization] Map loaded: {} points, {} voxel grid levels",
               map_points_count_, params_.voxelmap_levels);
  return true;
}

bool GlobalLocalization::localize(const float4* source_points, int source_count,
                                  float* transform_3x4, float* final_error) {
  if (!map_loaded_) {
    spdlog::error("[GlobalLocalization] Map not loaded");
    return false;
  }

  std::lock_guard<std::mutex> lock(localization_mutex_);

  icp_->set_max_correspondence_distance(
      static_cast<float>(params_.icp_max_correspondence_distance));
  icp_->set_convergence_threshold(
      static_cast<float>(params_.icp_transformation_epsilon));

  float best_error = 1e30f;
  float best_transform[12];

  float T_identity[12] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f
  };
  memcpy(best_transform, T_identity, 12 * sizeof(float));

  if (transform_3x4[0] == 1.0f && transform_3x4[1] == 0.0f &&
      transform_3x4[2] == 0.0f && transform_3x4[4] == 0.0f &&
      transform_3x4[5] == 1.0f && transform_3x4[6] == 0.0f &&
      transform_3x4[8] == 0.0f && transform_3x4[9] == 0.0f &&
      transform_3x4[10] == 1.0f &&
      transform_3x4[3] == 0.0f && transform_3x4[7] == 0.0f &&
      transform_3x4[11] == 0.0f) {

    float radius = static_cast<float>(params_.initial_pose_search_radius);
    float angular_range = static_cast<float>(params_.initial_pose_search_angular_range);

    int trans_samples = 3;
    int rot_samples = 3;

    for (int ti = 0; ti < trans_samples; ti++) {
      for (int tj = 0; tj < trans_samples; tj++) {
        for (int tk = 0; tk < trans_samples; tk++) {
          float dx = radius * (static_cast<float>(ti) / trans_samples - 0.5f) * 2.0f;
          float dy = radius * (static_cast<float>(tj) / trans_samples - 0.5f) * 2.0f;
          float dz = radius * (static_cast<float>(tk) / trans_samples - 0.5f) * 2.0f;

          for (int ri = 0; ri < rot_samples; ri++) {
            float yaw = angular_range * (static_cast<float>(ri) / rot_samples - 0.5f) * 2.0f;

            float T_init[12] = {
              cosf(yaw), -sinf(yaw), 0.0f, dx,
              sinf(yaw), cosf(yaw),  0.0f, dy,
              0.0f,      0.0f,       1.0f, dz
            };

            float T_try[12];
            memcpy(T_try, T_init, 12 * sizeof(float));
            float error = 0.0f;

            bool ok = localize_with_guess(source_points, source_count,
                                          T_init, T_try, &error);
            if (ok && error < best_error) {
              best_error = error;
              memcpy(best_transform, T_try, 12 * sizeof(float));
            }
          }
        }
      }
    }
  } else {
    float error = 0.0f;
    localize_with_guess(source_points, source_count,
                        transform_3x4, best_transform, &error);
    best_error = error;
  }

  memcpy(transform_3x4, best_transform, 12 * sizeof(float));
  if (final_error) *final_error = best_error;

  return best_error < 1e10f;
}

bool GlobalLocalization::localize_with_guess(
    const float4* source_points, int source_count,
    const float* initial_transform_3x4,
    float* transform_3x4,
    float* final_error) {

  if (!map_loaded_ || map_voxel_grids_.empty()) {
    return false;
  }

  memcpy(transform_3x4, initial_transform_3x4, 12 * sizeof(float));

  for (int level = map_voxel_grids_.size() - 1; level >= 0; level--) {
    icp_->set_max_correspondence_distance(
        static_cast<float>(params_.icp_max_correspondence_distance *
                           std::pow(params_.voxelmap_scaling_factor, level)));

    float error = 0.0f;
    bool ok = icp_->align(source_points, source_count,
                          *map_voxel_grids_[level],
                          transform_3x4,
                          params_.icp_max_iterations,
                          &error);

    if (!ok) {
      spdlog::warn("[GlobalLocalization] ICP failed at level {}", level);
      return false;
    }
  }

  icp_->set_max_correspondence_distance(
      static_cast<float>(params_.icp_max_correspondence_distance));

  float error = 0.0f;
  bool ok = icp_->align(source_points, source_count,
                        *map_voxel_grids_[0],
                        transform_3x4,
                        params_.icp_max_iterations * 2,
                        &error);

  if (final_error) *final_error = error;
  return ok;
}

void GlobalLocalization::set_params(const GlobalLocalizationParams& params) {
  params_ = params;
  icp_->set_max_correspondence_distance(
      static_cast<float>(params_.icp_max_correspondence_distance));
  icp_->set_convergence_threshold(
      static_cast<float>(params_.icp_transformation_epsilon));

  if (map_loaded_) {
    build_map_voxel_grids();
  }
}

int GlobalLocalization::map_voxel_count() const {
  if (map_voxel_grids_.empty()) return 0;
  return map_voxel_grids_[0]->num_voxels();
}

void GlobalLocalization::release_map_resources() {
  if (d_map_points_) { cudaFree(d_map_points_); d_map_points_ = nullptr; }
  if (d_map_raw_) { cudaFree(d_map_raw_); d_map_raw_ = nullptr; }
  map_voxel_grids_.clear();
  map_loaded_ = false;
}

}  // namespace cuda_slam