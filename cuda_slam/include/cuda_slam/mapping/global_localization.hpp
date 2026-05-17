#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <vector>
#include <mutex>

namespace cuda_slam {
namespace cuda {
class CudaVoxelGrid;
class CudaICP;
}  // namespace cuda

struct GlobalLocalizationParams {
  int voxelmap_levels = 3;
  double voxel_resolution_base = 0.5;
  double voxelmap_scaling_factor = 2.0;
  int icp_max_iterations = 30;
  double icp_max_correspondence_distance = 5.0;
  double icp_transformation_epsilon = 1e-5;
  double initial_pose_search_radius = 10.0;
  double initial_pose_search_angular_range = 0.5;
};

class GlobalLocalization {
public:
  using Ptr = std::shared_ptr<GlobalLocalization>;

  explicit GlobalLocalization(const GlobalLocalizationParams& params = {});
  ~GlobalLocalization();

  GlobalLocalization(const GlobalLocalization&) = delete;
  GlobalLocalization& operator=(const GlobalLocalization&) = delete;

  bool load_map(const std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Vector3d>>& points);

  bool load_map(const float* points, int num_points);

  bool is_map_loaded() const { return map_loaded_; }

  bool localize(const float4* source_points, int source_count,
                float* transform_3x4, float* final_error = nullptr);

  bool localize_with_guess(const float4* source_points, int source_count,
                           const float* initial_transform_3x4,
                           float* transform_3x4,
                           float* final_error = nullptr);

  int map_point_count() const { return map_points_count_; }
  int map_voxel_count() const;

  void set_params(const GlobalLocalizationParams& params);

private:
  bool build_map_voxel_grids();

  void upload_points(const float* points, int num_points);

  void release_map_resources();

  GlobalLocalizationParams params_;
  bool map_loaded_;

  int map_points_count_;
  float4* d_map_points_;
  float* d_map_raw_;

  std::vector<std::unique_ptr<cuda::CudaVoxelGrid>> map_voxel_grids_;
  std::unique_ptr<cuda::CudaICP> icp_;

  std::mutex localization_mutex_;
};

}  // namespace cuda_slam