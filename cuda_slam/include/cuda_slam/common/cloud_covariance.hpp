#pragma once

#include <Eigen/Core>
#include <vector>

namespace cuda_slam {

class CloudCovarianceEstimation {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Params {
    int num_threads = 4;
    double regularization = 1e-3;
  };

  CloudCovarianceEstimation() = default;

  explicit CloudCovarianceEstimation(const Params& params) : params_(params) {}

  void estimate(
      const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
      const std::vector<int>& neighbors,
      const std::vector<int>& neighbor_offsets,
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& normals,
      std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& covariances) const;

  const Params& params() const { return params_; }
  void set_params(const Params& params) { params_ = params; }

private:
  Params params_;
};

} // namespace cuda_slam