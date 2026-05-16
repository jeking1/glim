#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

namespace cuda_slam {

struct RawPoints {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<RawPoints>;
  using ConstPtr = std::shared_ptr<const RawPoints>;

  double timestamp = 0.0;

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;

  std::vector<double> times;

  std::vector<float> intensities;

  [[nodiscard]] std::size_t size() const { return points.size(); }

  [[nodiscard]] bool empty() const { return points.empty(); }

  void reserve(std::size_t n) {
    points.reserve(n);
    times.reserve(n);
    intensities.reserve(n);
  }

  void clear() {
    timestamp = 0.0;
    points.clear();
    times.clear();
    intensities.clear();
  }

  void add_point(const Eigen::Vector4d& pt, double t, float intensity) {
    points.push_back(pt);
    times.push_back(t);
    intensities.push_back(intensity);
  }
};

} // namespace cuda_slam