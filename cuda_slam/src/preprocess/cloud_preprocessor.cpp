#include "cuda_slam/preprocess/cloud_preprocessor.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cuda_slam {

void CloudPreprocessor::distance_filter(
    const RawPoints& raw, std::vector<std::size_t>& kept_indices, double near,
    double far) {
  const double near2 = near * near;
  const double far2 = far * far;

  kept_indices.clear();
  kept_indices.reserve(raw.size());

  for (std::size_t i = 0; i < raw.points.size(); ++i) {
    const auto& pt = raw.points[i];
    const double dist2 = pt.x() * pt.x() + pt.y() * pt.y() + pt.z() * pt.z();
    if (dist2 >= near2 && dist2 <= far2) {
      kept_indices.push_back(i);
    }
  }
}

namespace {

struct VoxelKey {
  int64_t ix, iy, iz;

  bool operator==(const VoxelKey& o) const {
    return ix == o.ix && iy == o.iy && iz == o.iz;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& k) const {
    std::size_t hx = static_cast<std::size_t>(k.ix) * 73856093;
    std::size_t hy = static_cast<std::size_t>(k.iy) * 19349663;
    std::size_t hz = static_cast<std::size_t>(k.iz) * 83492791;
    return hx ^ hy ^ hz;
  }
};

} // namespace

void CloudPreprocessor::voxel_downsample(
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    const std::vector<std::size_t>& input_indices, double resolution,
    std::vector<std::size_t>& output_indices) {
  if (input_indices.empty()) {
    output_indices.clear();
    return;
  }

  const double inv_res = 1.0 / resolution;
  std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> voxel_map;
  voxel_map.reserve(input_indices.size());

  output_indices.clear();
  output_indices.reserve(input_indices.size());

  for (const auto idx : input_indices) {
    const auto& pt = points[idx];
    VoxelKey key;
    key.ix = static_cast<int64_t>(std::floor(pt.x() * inv_res));
    key.iy = static_cast<int64_t>(std::floor(pt.y() * inv_res));
    key.iz = static_cast<int64_t>(std::floor(pt.z() * inv_res));

    auto [it, inserted] = voxel_map.try_emplace(key, idx);
    if (!inserted) {
      const auto& prev_pt = points[it->second];
      const auto& cur_pt = pt;
      double prev_dist2 = prev_pt.x() * prev_pt.x() + prev_pt.y() * prev_pt.y() +
                          prev_pt.z() * prev_pt.z();
      double cur_dist2 = cur_pt.x() * cur_pt.x() + cur_pt.y() * cur_pt.y() +
                         cur_pt.z() * cur_pt.z();
      if (cur_dist2 < prev_dist2) {
        it->second = idx;
      }
    } else {
      output_indices.push_back(idx);
    }
  }
}

void CloudPreprocessor::transform_points(
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& src,
    const std::vector<std::size_t>& indices,
    const Eigen::Isometry3d& T,
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& dst) {
  dst.clear();
  dst.reserve(indices.size());

  const Eigen::Matrix4d mat = T.matrix();

  for (const auto idx : indices) {
    const auto& pt = src[idx];
    Eigen::Vector4d transformed = mat * pt;
    dst.push_back(transformed);
  }
}

void CloudPreprocessor::build_neighbors(
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
    int k,
    std::vector<int>& neighbors,
    std::vector<int>& offsets) {
  const std::size_t n = points.size();
  if (n == 0) {
    neighbors.clear();
    offsets.clear();
    return;
  }

  const int effective_k = std::min(k, static_cast<int>(n));
  const std::size_t total_neighbors = n * effective_k;

  neighbors.clear();
  neighbors.reserve(total_neighbors);
  offsets.clear();
  offsets.reserve(n + 1);
  offsets.push_back(0);

  const int num_threads = params_.num_threads;

#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    const auto& pt_i = points[i];
    const double xi = pt_i.x();
    const double yi = pt_i.y();
    const double zi = pt_i.z();

    std::vector<std::pair<double, int>> dists;
    dists.reserve(n);

    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;
      const auto& pt_j = points[j];
      const double dx = xi - pt_j.x();
      const double dy = yi - pt_j.y();
      const double dz = zi - pt_j.z();
      dists.emplace_back(dx * dx + dy * dy + dz * dz,
                         static_cast<int>(j));
    }

    std::nth_element(dists.begin(), dists.begin() + effective_k - 1,
                     dists.end());
    std::sort(dists.begin(), dists.begin() + effective_k);

#pragma omp critical
    {
      for (int m = 0; m < effective_k; ++m) {
        neighbors.push_back(dists[m].second);
      }
      offsets.push_back(static_cast<int>(neighbors.size()));
    }
  }

  spdlog::debug("CloudPreprocessor: built {} neighbors for {} points (k={})",
                neighbors.size(), n, effective_k);
}

PreprocessedFrame::Ptr CloudPreprocessor::preprocess(
    const RawPoints& raw, const Eigen::Isometry3d& T_imu_lidar) const {
  auto frame = std::make_shared<PreprocessedFrame>();
  frame->stamp = raw.timestamp;

  if (raw.empty()) {
    spdlog::warn("CloudPreprocessor: empty raw points at stamp {:.6f}",
                 raw.timestamp);
    return frame;
  }

  std::vector<std::size_t> filtered_indices;
  distance_filter(raw, filtered_indices, params_.distance_near,
                  params_.distance_far);

  spdlog::debug(
      "CloudPreprocessor: distance filter {} -> {} points (near={:.2f}, "
      "far={:.2f})",
      raw.size(), filtered_indices.size(), params_.distance_near,
      params_.distance_far);

  if (filtered_indices.empty()) {
    spdlog::warn(
        "CloudPreprocessor: no points after distance filter at stamp {:.6f}",
        raw.timestamp);
    return frame;
  }

  std::vector<std::size_t> downsampled_indices;
  voxel_downsample(raw.points, filtered_indices, params_.downsample_resolution,
                   downsampled_indices);

  spdlog::debug(
      "CloudPreprocessor: voxel downsample {} -> {} points "
      "(resolution={:.3f})",
      filtered_indices.size(), downsampled_indices.size(),
      params_.downsample_resolution);

  if (downsampled_indices.empty()) {
    spdlog::warn(
        "CloudPreprocessor: no points after downsample at stamp {:.6f}",
        raw.timestamp);
    return frame;
  }

  transform_points(raw.points, downsampled_indices, T_imu_lidar, frame->points);

  spdlog::debug("CloudPreprocessor: transformed {} points to IMU frame",
                frame->points.size());

  const int k_neighbors = 20;
  build_neighbors(frame->points, k_neighbors, frame->neighbors,
                  frame->neighbor_offsets);

  spdlog::info(
      "CloudPreprocessor: preprocessed frame stamp={:.6f}, {} points",
      frame->stamp, frame->size());

  return frame;
}

} // namespace cuda_slam