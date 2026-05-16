#include "cuda_slam/common/cloud_covariance.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <spdlog/spdlog.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cuda_slam {

void CloudCovarianceEstimation::estimate(
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    const std::vector<int>& neighbors,
    const std::vector<int>& neighbor_offsets,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& normals,
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& covariances) const {
  const std::size_t n = points.size();

  normals.clear();
  normals.resize(n, Eigen::Vector3d::UnitZ());

  covariances.clear();
  covariances.resize(n, Eigen::Matrix3d::Identity() * params_.regularization);

  if (n == 0) {
    spdlog::warn("CloudCovarianceEstimation: no points");
    return;
  }

  if (neighbor_offsets.size() < n + 1) {
    spdlog::error(
        "CloudCovarianceEstimation: neighbor_offsets size {} < n+1={}",
        neighbor_offsets.size(), n + 1);
    return;
  }

  const int num_threads = params_.num_threads;

#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    const int k_start = neighbor_offsets[i];
    const int k_end = neighbor_offsets[i + 1];
    const int k_count = k_end - k_start;

    if (k_count < 3) {
      normals[i] = Eigen::Vector3d::UnitZ();
      covariances[i] =
          Eigen::Matrix3d::Identity() * params_.regularization;
      continue;
    }

    const auto& pt_i = points[i];
    const Eigen::Vector3d centroid(pt_i.x(), pt_i.y(), pt_i.z());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    for (int j = k_start; j < k_end; ++j) {
      const auto& pt_j = points[static_cast<std::size_t>(neighbors[j])];
      const Eigen::Vector3d diff(pt_j.x() - centroid.x(),
                                 pt_j.y() - centroid.y(),
                                 pt_j.z() - centroid.z());
      cov += diff * diff.transpose();
    }

    cov /= static_cast<double>(k_count);

    cov.diagonal().array() += params_.regularization;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov);

    if (eigensolver.info() == Eigen::Success) {
      const auto& eigenvalues = eigensolver.eigenvalues();
      const auto& eigenvectors = eigensolver.eigenvectors();

      int min_idx = 0;
      if (eigenvalues(1) < eigenvalues(min_idx)) min_idx = 1;
      if (eigenvalues(2) < eigenvalues(min_idx)) min_idx = 2;

      normals[i] = eigenvectors.col(min_idx).normalized();

      covariances[i] = cov;
    } else {
      normals[i] = Eigen::Vector3d::UnitZ();
      covariances[i] =
          Eigen::Matrix3d::Identity() * params_.regularization;
    }
  }

  spdlog::debug(
      "CloudCovarianceEstimation: estimated normals/covariances for {} points",
      n);
}

} // namespace cuda_slam