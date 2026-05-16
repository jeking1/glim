#pragma once

#include "cuda_slam/preprocess/preprocessed_frame.hpp"
#include "cuda_slam/preprocess/raw_points.hpp"
#include "cuda_slam/util/config.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

namespace cuda_slam {

class CloudPreprocessor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Params {
    double distance_near = 0.5;
    double distance_far = 100.0;
    double downsample_resolution = 0.2;
    int num_threads = 4;
  };

  CloudPreprocessor() = default;

  explicit CloudPreprocessor(const Params& params) : params_(params) {}

  bool load_params(const std::string& config_path,
                   const std::string& key_prefix = "preprocess") {
    Config cfg;
    if (!cfg.load(config_path)) {
      return false;
    }
    params_.distance_near =
        cfg.get<double>(key_prefix + ".distance_near", params_.distance_near);
    params_.distance_far =
        cfg.get<double>(key_prefix + ".distance_far", params_.distance_far);
    params_.downsample_resolution =
        cfg.get<double>(key_prefix + ".downsample_resolution",
                        params_.downsample_resolution);
    params_.num_threads =
        cfg.get<int>(key_prefix + ".num_threads", params_.num_threads);
    return true;
  }

  bool load_params(const Config& cfg,
                   const std::string& key_prefix = "preprocess") {
    params_.distance_near =
        cfg.get<double>(key_prefix + ".distance_near", params_.distance_near);
    params_.distance_far =
        cfg.get<double>(key_prefix + ".distance_far", params_.distance_far);
    params_.downsample_resolution =
        cfg.get<double>(key_prefix + ".downsample_resolution",
                        params_.downsample_resolution);
    params_.num_threads =
        cfg.get<int>(key_prefix + ".num_threads", params_.num_threads);
    return true;
  }

  PreprocessedFrame::Ptr preprocess(const RawPoints& raw,
                                    const Eigen::Isometry3d& T_imu_lidar) const;

  const Params& params() const { return params_; }
  void set_params(const Params& params) { params_ = params; }

private:
  static void distance_filter(const RawPoints& raw,
                              std::vector<std::size_t>& kept_indices,
                              double near, double far);

  static void voxel_downsample(
      const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
      const std::vector<std::size_t>& input_indices,
      double resolution,
      std::vector<std::size_t>& output_indices);

  static void transform_points(
      const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& src,
      const std::vector<std::size_t>& indices,
      const Eigen::Isometry3d& T,
      std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& dst);

  static void build_neighbors(
      const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
      int k,
      std::vector<int>& neighbors,
      std::vector<int>& offsets);

  Params params_;
};

} // namespace cuda_slam