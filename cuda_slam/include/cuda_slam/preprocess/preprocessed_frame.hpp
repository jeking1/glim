#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace cuda_slam {

struct PreprocessedFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<PreprocessedFrame>;
  using ConstPtr = std::shared_ptr<const PreprocessedFrame>;

  double stamp = 0.0;

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;

  std::vector<int> neighbors;

  std::vector<int> neighbor_offsets;

  [[nodiscard]] std::size_t size() const { return points.size(); }

  [[nodiscard]] bool empty() const { return points.empty(); }

  [[nodiscard]] int neighbor_count(std::size_t point_idx) const {
    if (point_idx + 1 >= neighbor_offsets.size()) {
      return 0;
    }
    return neighbor_offsets[point_idx + 1] - neighbor_offsets[point_idx];
  }

  [[nodiscard]] const int* neighbor_ptr(std::size_t point_idx) const {
    return neighbors.data() + neighbor_offsets[point_idx];
  }
};

} // namespace cuda_slam