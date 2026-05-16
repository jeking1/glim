#include "cuda_slam/odometry/odometry_gpu.hpp"

#include <Eigen/Cholesky>
#include <Eigen/StdVector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "cuda_slam/cuda/cuda_icp.hpp"
#include "cuda_slam/cuda/cuda_voxel_grid.hpp"
#include "cuda_slam/odometry/callbacks.hpp"
#include <spdlog/spdlog.h>

namespace cuda_slam {

namespace {

Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(),
       v.z(), 0, -v.x(),
       -v.y(), v.x(), 0;
  return S;
}

Eigen::Isometry3d exp_map(const Eigen::Matrix<double, 6, 1>& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d phi = xi.tail<3>();
  const double angle = phi.norm();

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  if (angle < 1e-12) {
    T.translation() = rho;
    T.linear() = Eigen::Matrix3d::Identity() + skew_symmetric(phi);
  } else {
    const Eigen::Matrix3d S = skew_symmetric(phi / angle);
    T.linear() = Eigen::Matrix3d::Identity() + std::sin(angle) * S +
                 (1.0 - std::cos(angle)) * S * S;
    const Eigen::Matrix3d V = Eigen::Matrix3d::Identity() +
                              (1.0 - std::cos(angle)) / angle * S +
                              (angle - std::sin(angle)) / angle * S * S;
    T.translation() = V * rho;
  }
  return T;
}

Eigen::Matrix<double, 6, 1> log_map(const Eigen::Isometry3d& T) {
  const Eigen::Matrix3d R = T.linear();
  const Eigen::Vector3d t = T.translation();

  double cos_angle = (R.trace() - 1.0) / 2.0;
  cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
  const double angle = std::acos(cos_angle);

  Eigen::Matrix<double, 6, 1> xi;
  if (angle < 1e-12) {
    xi.head<3>() = t;
    xi.tail<3>().setZero();
  } else {
    const Eigen::Matrix3d S_mat = (R - R.transpose()) / (2.0 * std::sin(angle)) * angle;
    const Eigen::Vector3d phi(angle * S_mat(2, 1),
                               angle * S_mat(0, 2),
                               angle * S_mat(1, 0));
    const Eigen::Matrix3d S_phi = skew_symmetric(phi / angle);
    const Eigen::Matrix3d V_inv =
        Eigen::Matrix3d::Identity() - 0.5 * S_phi +
        (1.0 / (angle * angle) -
         (1.0 + std::cos(angle)) / (2.0 * angle * std::sin(angle))) *
            S_phi * S_phi;
    xi.head<3>() = V_inv * t;
    xi.tail<3>() = phi;
  }
  return xi;
}

void eigen_to_float4_array(
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& pts,
    std::vector<float4>& out) {
  out.resize(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    out[i] = make_float4(static_cast<float>(pts[i].x()),
                         static_cast<float>(pts[i].y()),
                         static_cast<float>(pts[i].z()),
                         static_cast<float>(pts[i].w()));
  }
}

void eigen_to_float_array(const Eigen::Isometry3d& T, float transform_3x4[12]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      transform_3x4[r * 4 + c] = static_cast<float>(T.linear()(r, c));
    }
    transform_3x4[r * 4 + 3] = static_cast<float>(T.translation()(r));
  }
}

void float_array_to_eigen(const float transform_3x4[12], Eigen::Isometry3d& T) {
  T.setIdentity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      T.linear()(r, c) = static_cast<double>(transform_3x4[r * 4 + c]);
    }
    T.translation()(r) = static_cast<double>(transform_3x4[r * 4 + 3]);
  }
}

class CpuVoxelGrid {
public:
  explicit CpuVoxelGrid(double resolution) : resolution_(resolution) {}

  void set_points(
      const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
          points) {
    points_ = points;
    build_index();
  }

  void build_index() {
    voxel_map_.clear();
    const double inv_res = 1.0 / resolution_;
    for (size_t i = 0; i < points_.size(); ++i) {
      const auto& pt = points_[i];
      int64_t ix = static_cast<int64_t>(std::floor(pt.x() * inv_res));
      int64_t iy = static_cast<int64_t>(std::floor(pt.y() * inv_res));
      int64_t iz = static_cast<int64_t>(std::floor(pt.z() * inv_res));
      voxel_map_[hash_coord(ix, iy, iz)].push_back(i);
    }
  }

  bool nearest_neighbor(const Eigen::Vector3d& query, Eigen::Vector3d& neighbor,
                        double max_distance) const {
    const double inv_res = 1.0 / resolution_;
    int64_t cx = static_cast<int64_t>(std::floor(query.x() * inv_res));
    int64_t cy = static_cast<int64_t>(std::floor(query.y() * inv_res));
    int64_t cz = static_cast<int64_t>(std::floor(query.z() * inv_res));

    double best_dist = max_distance * max_distance;
    bool found = false;

    for (int64_t dx = -1; dx <= 1; ++dx) {
      for (int64_t dy = -1; dy <= 1; ++dy) {
        for (int64_t dz = -1; dz <= 1; ++dz) {
          auto it = voxel_map_.find(hash_coord(cx + dx, cy + dy, cz + dz));
          if (it == voxel_map_.end()) continue;

          for (size_t idx : it->second) {
            Eigen::Vector3d diff = points_[idx].head<3>() - query;
            double dist = diff.squaredNorm();
            if (dist < best_dist) {
              best_dist = dist;
              neighbor = points_[idx].head<3>();
              found = true;
            }
          }
        }
      }
    }
    return found;
  }

  bool is_empty() const { return points_.empty(); }
  size_t size() const { return points_.size(); }

private:
  static int64_t hash_coord(int64_t x, int64_t y, int64_t z) {
    const int64_t p1 = 73856093;
    const int64_t p2 = 19349663;
    const int64_t p3 = 83492791;
    return x * p1 + y * p2 + z * p3;
  }

  double resolution_;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_;
  std::unordered_map<int64_t, std::vector<size_t>> voxel_map_;
};

} // namespace

OdometryEstimationGPU::OdometryEstimationGPU(const OdometryParams& params)
    : params_(params) {
  imu_integrator_.set_gravity(Eigen::Vector3d(0, 0, -params.imu_gravity));
  imu_integrator_.set_max_queue_size(1000);

  CloudCovarianceEstimation::Params cov_params;
  cov_params.num_threads = params.covariance_num_neighbors;
  cov_params.regularization = 1e-3;
  covariance_estimator_ = CloudCovarianceEstimation(cov_params);

  spdlog::info("OdometryEstimationGPU initialized: {} voxelmap levels, "
               "resolution={:.3f}, smoother_lag={}, max_keyframes={}",
               params.voxelmap_levels, params.voxel_resolution,
               params.smoother_lag, params.max_num_keyframes);
}

OdometryEstimationGPU::~OdometryEstimationGPU() = default;

void OdometryEstimationGPU::insert_imu(double stamp,
                                        const Eigen::Vector3d& linear_acc,
                                        const Eigen::Vector3d& angular_vel) {
  std::lock_guard<std::mutex> lock(imu_mutex_);
  imu_integrator_.insert_imu(stamp, linear_acc, angular_vel);
  OdometryCallbacks::on_insert_imu.call(stamp, linear_acc, angular_vel);
}

EstimationFrame::ConstPtr OdometryEstimationGPU::insert_frame(
    std::shared_ptr<PreprocessedFrame> raw_frame) {
  if (!raw_frame) {
    spdlog::warn("insert_frame called with null frame");
    return nullptr;
  }

  OdometryCallbacks::on_insert_frame.call(raw_frame);

  auto ef = std::make_shared<EstimationFrame>();
  ef->id = frame_counter_++;
  ef->stamp = raw_frame->stamp;
  ef->T_lidar_imu = Eigen::Isometry3d::Identity();
  ef->imu_bias = imu_state_.bias;
  ef->raw_frame = raw_frame;

  Eigen::Isometry3d T_predicted = predict_pose(*raw_frame);

  PointCloudCPU deskewed;
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    Eigen::Isometry3d T_start = imu_state_.T_world_imu;
    deskew_cloud(*raw_frame, T_start, T_predicted, deskewed);
  }

  estimate_covariances(deskewed, *raw_frame);

  ef->frame = std::move(deskewed);

  build_voxel_grids(*ef);

  std::vector<PointCloudCPU> target_clouds;
  {
    std::lock_guard<std::mutex> lock(frames_mutex_);
    for (const auto& f : frames_) {
      target_clouds.push_back(f->frame);
    }
    for (const auto& kf : keyframes_) {
      target_clouds.push_back(kf->frame);
    }
  }

  ef->T_world_lidar = icp_align(ef->frame, target_clouds, T_predicted);
  ef->T_world_imu = ef->T_world_lidar * ef->T_lidar_imu.inverse();

  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    ImuBias imu_bias;
    imu_bias.accel_bias = imu_state_.bias.head<3>();
    imu_bias.gyro_bias = imu_state_.bias.tail<3>();

    NavState nav_state = imu_integrator_.integrate_imu(
        raw_frame->stamp - 0.1, raw_frame->stamp, imu_bias);
    ef->v_world_imu = nav_state.velocity;

    imu_state_.T_world_imu = ef->T_world_imu;
    imu_state_.v_world_imu = ef->v_world_imu;
  }

  spdlog::debug("Frame {}: T=[{:.3f},{:.3f},{:.3f}]",
                ef->id,
                ef->T_world_lidar.translation().x(),
                ef->T_world_lidar.translation().y(),
                ef->T_world_lidar.translation().z());

  OdometryCallbacks::on_new_frame.call(ef);

  EstimationFrame::ConstPtr const_ef = ef;
  update_sliding_window(const_ef);
  manage_keyframes(const_ef);
  marginalize_old_frames();
  optimize_sliding_window();

  return const_ef;
}

std::vector<EstimationFrame::ConstPtr>
OdometryEstimationGPU::get_remaining_frames() const {
  std::lock_guard<std::mutex> lock(frames_mutex_);
  return std::vector<EstimationFrame::ConstPtr>(frames_.begin(), frames_.end());
}

const std::vector<EstimationFrame::ConstPtr>&
OdometryEstimationGPU::get_keyframes() const {
  std::lock_guard<std::mutex> lock(frames_mutex_);
  return keyframes_;
}

void OdometryEstimationGPU::set_params(const OdometryParams& params) {
  params_ = params;
  imu_integrator_.set_gravity(Eigen::Vector3d(0, 0, -params.imu_gravity));

  CloudCovarianceEstimation::Params cov_params;
  cov_params.num_threads = params.covariance_num_neighbors;
  cov_params.regularization = 1e-3;
  covariance_estimator_ = CloudCovarianceEstimation(cov_params);
}

Eigen::Isometry3d OdometryEstimationGPU::predict_pose(
    const PreprocessedFrame& frame) const {
  std::lock_guard<std::mutex> lock(imu_mutex_);

  if (imu_integrator_.empty()) {
    return imu_state_.T_world_imu;
  }

  ImuBias imu_bias;
  imu_bias.accel_bias = imu_state_.bias.head<3>();
  imu_bias.gyro_bias = imu_state_.bias.tail<3>();

  NavState nav_state = imu_integrator_.integrate_imu(
      frame.stamp - 0.1, frame.stamp, imu_bias);

  return nav_state.pose();
}

void OdometryEstimationGPU::deskew_cloud(const PreprocessedFrame& raw,
                                          const Eigen::Isometry3d& T_start,
                                          const Eigen::Isometry3d& T_end,
                                          PointCloudCPU& deskewed) const {
  const size_t n = raw.points.size();
  deskewed.points.resize(n);

  if (n == 0) return;

  const double t_start = raw.stamp;
  const double t_end = raw.stamp;
  const double duration = t_end - t_start;

  Eigen::Matrix<double, 6, 1> xi_start = log_map(T_start);
  Eigen::Matrix<double, 6, 1> xi_end = log_map(T_end);

  if (duration > 1e-12) {
    for (size_t i = 0; i < n; ++i) {
      double alpha = static_cast<double>(i) / static_cast<double>(n - 1);
      Eigen::Matrix<double, 6, 1> xi_interp =
          (1.0 - alpha) * xi_start + alpha * xi_end;
      Eigen::Isometry3d T_interp = exp_map(xi_interp);

      Eigen::Vector4d pt = raw.points[i];
      deskewed.points[i] = T_interp * pt;
      deskewed.points[i].w() = pt.w();
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      deskewed.points[i] = T_start * raw.points[i];
    }
  }
}

void OdometryEstimationGPU::estimate_covariances(
    PointCloudCPU& cloud, const PreprocessedFrame& raw) const {
  const size_t n = cloud.points.size();
  cloud.normals.resize(n, Eigen::Vector4d::Zero());
  cloud.covs.resize(n, Eigen::Matrix4d::Zero());

  if (n == 0) return;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> normals_3d(n);
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> covs_3d(n);

  covariance_estimator_.estimate(cloud.points, raw.neighbors,
                                  raw.neighbor_offsets, normals_3d, covs_3d);

  for (size_t i = 0; i < n; ++i) {
    cloud.normals[i] = Eigen::Vector4d(normals_3d[i].x(), normals_3d[i].y(),
                                        normals_3d[i].z(), 0.0);
    cloud.covs[i].setIdentity();
    cloud.covs[i].topLeftCorner<3, 3>() = covs_3d[i];
  }
}

void OdometryEstimationGPU::build_voxel_grids(EstimationFrame& ef) const {
  ef.voxelmaps.clear();
  ef.voxelmaps.reserve(params_.voxelmap_levels);

  for (int level = 0; level < params_.voxelmap_levels; ++level) {
    double resolution = params_.voxel_resolution *
                        std::pow(params_.voxelmap_scaling_factor, level);
    auto vg = std::make_shared<cuda::CudaVoxelGrid>(
        static_cast<float>(resolution));

    std::vector<float4> pts_f4;
    eigen_to_float4_array(ef.frame.points, pts_f4);
    vg->insert(pts_f4.data(), static_cast<int>(pts_f4.size()));

    ef.voxelmaps.push_back(vg);
  }
}

Eigen::Isometry3d OdometryEstimationGPU::icp_align(
    const PointCloudCPU& source,
    const std::vector<PointCloudCPU>& target_clouds,
    const Eigen::Isometry3d& initial_guess) const {
  if (source.points.empty() || target_clouds.empty()) {
    return initial_guess;
  }

  Eigen::Isometry3d T = initial_guess;
  const int max_iter = params_.icp_max_iterations;
  const double max_dist = params_.icp_max_correspondence_distance;

  std::vector<CpuVoxelGrid> tgt_grids;
  tgt_grids.reserve(target_clouds.size());
  for (const auto& cloud : target_clouds) {
    tgt_grids.emplace_back(max_dist);
    tgt_grids.back().set_points(cloud.points);
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    int num_correspondences = 0;

    for (size_t i = 0; i < source.points.size(); ++i) {
      Eigen::Vector3d src_pt = source.points[i].head<3>();
      Eigen::Vector3d src_transformed = T * src_pt;

      Eigen::Vector3d tgt_pt = Eigen::Vector3d::Zero();
      bool found = false;

      for (auto& tgt_grid : tgt_grids) {
        Eigen::Vector3d neighbor;
        if (tgt_grid.nearest_neighbor(src_transformed, neighbor, max_dist)) {
          tgt_pt = neighbor;
          found = true;
          break;
        }
      }

      if (!found) continue;

      Eigen::Vector3d residual_vec = src_transformed - tgt_pt;
      double residual = residual_vec.norm();

      Eigen::Matrix<double, 1, 6> J;
      J.head<3>() = residual_vec.normalized().transpose();
      J.tail<3>() =
          -residual_vec.normalized().transpose() * skew_symmetric(src_transformed);

      H += J.transpose() * J;
      b -= J.transpose() * residual;
      ++num_correspondences;
    }

    if (num_correspondences < 6) break;

    Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
    T = exp_map(dx) * T;

    if (dx.norm() < params_.icp_transformation_epsilon) break;
  }

  return T;
}

void OdometryEstimationGPU::update_sliding_window(
    EstimationFrame::ConstPtr new_frame) {
  std::lock_guard<std::mutex> lock(frames_mutex_);
  frames_.push_back(new_frame);

  std::vector<EstimationFrame::ConstPtr> all_frames(frames_.begin(),
                                                     frames_.end());
  OdometryCallbacks::on_update_frames.call(all_frames);
}

void OdometryEstimationGPU::manage_keyframes(
    EstimationFrame::ConstPtr new_frame) {
  std::lock_guard<std::mutex> lock(frames_mutex_);

  bool is_kf = false;
  if (keyframes_.empty()) {
    is_kf = true;
  } else {
    switch (params_.keyframe_strategy) {
      case KeyframeStrategy::OVERLAP:
        is_kf = is_keyframe_overlap(new_frame);
        break;
      case KeyframeStrategy::DISPLACEMENT:
        is_kf = is_keyframe_displacement(new_frame);
        break;
    }
  }

  if (is_kf) {
    keyframes_.push_back(new_frame);
    spdlog::debug("New keyframe {} added (total: {})", new_frame->id,
                  keyframes_.size());

    while (keyframes_.size() > static_cast<size_t>(params_.max_num_keyframes)) {
      keyframes_.erase(keyframes_.begin());
    }

    OdometryCallbacks::on_update_keyframes.call(keyframes_);
  }
}

bool OdometryEstimationGPU::is_keyframe_overlap(
    EstimationFrame::ConstPtr frame) const {
  if (keyframes_.empty()) return true;

  const auto& last_kf = keyframes_.back();

  int total_pts = static_cast<int>(frame->frame.points.size());
  int overlapping_pts = 0;

  if (total_pts == 0) return false;

  const double dist_threshold = params_.voxel_resolution * 2.0;

  CpuVoxelGrid kf_grid(dist_threshold);
  kf_grid.set_points(last_kf->frame.points);

  for (const auto& pt : frame->frame.points) {
    Eigen::Vector3d p = frame->T_world_lidar * pt.head<3>();
    Eigen::Vector3d p_in_kf =
        last_kf->T_world_lidar.inverse() * frame->T_world_lidar * pt.head<3>();

    Eigen::Vector3d neighbor;
    if (kf_grid.nearest_neighbor(p_in_kf, neighbor, dist_threshold)) {
      ++overlapping_pts;
    }
  }

  double overlap_ratio =
      static_cast<double>(overlapping_pts) / static_cast<double>(total_pts);

  return overlap_ratio < params_.keyframe_max_overlap &&
         overlap_ratio > params_.keyframe_min_overlap;
}

bool OdometryEstimationGPU::is_keyframe_displacement(
    EstimationFrame::ConstPtr frame) const {
  if (keyframes_.empty()) return true;

  const auto& last_kf = keyframes_.back();

  Eigen::Vector3d delta_trans =
      frame->T_world_lidar.translation() - last_kf->T_world_lidar.translation();

  Eigen::Matrix3d delta_rot_mat =
      frame->T_world_lidar.linear().transpose() * last_kf->T_world_lidar.linear();
  Eigen::AngleAxisd delta_rot(delta_rot_mat);

  double trans_dist = delta_trans.norm();
  double rot_angle = std::abs(delta_rot.angle());

  return trans_dist > params_.keyframe_delta_trans ||
         rot_angle > params_.keyframe_delta_rot;
}

void OdometryEstimationGPU::marginalize_old_frames() {
  std::lock_guard<std::mutex> lock(frames_mutex_);

  int lag = params_.smoother_lag;
  if (lag <= 0 || frames_.empty()) return;

  std::vector<EstimationFrame::ConstPtr> marginalized;

  while (frames_.size() > static_cast<size_t>(lag)) {
    marginalized.push_back(frames_.front());
    frames_.pop_front();
    ++marginalized_cursor_;
  }

  if (!marginalized.empty()) {
    OdometryCallbacks::on_marginalized_frames.call(marginalized);
    spdlog::debug("Marginalized {} frames, cursor={}", marginalized.size(),
                  marginalized_cursor_);
  }
}

void OdometryEstimationGPU::optimize_sliding_window() {
  std::lock_guard<std::mutex> lock(frames_mutex_);

  if (frames_.size() < 2) return;

  const int max_gn_iterations = 5;
  const double lambda = 0.01;

  std::vector<Eigen::Matrix<double, 6, 1>> dx_stack(
      frames_.size(), Eigen::Matrix<double, 6, 1>::Zero());

  for (int gn_iter = 0; gn_iter < max_gn_iterations; ++gn_iter) {
    Eigen::MatrixXd H_all(6 * frames_.size(), 6 * frames_.size());
    Eigen::VectorXd b_all(6 * frames_.size());
    H_all.setZero();
    b_all.setZero();

    for (size_t i = 0; i < frames_.size(); ++i) {
      for (size_t j = i + 1; j < frames_.size(); ++j) {
        const auto& fi = frames_[i];
        const auto& fj = frames_[j];

        if (fi->frame.points.empty() || fj->frame.points.empty()) continue;

        CpuVoxelGrid grid_j(params_.icp_max_correspondence_distance);
        grid_j.set_points(fj->frame.points);

        for (const auto& pt_i : fi->frame.points) {
          Eigen::Vector3d pi = fi->T_world_lidar * pt_i.head<3>();
          Eigen::Vector3d pi_in_j =
              fj->T_world_lidar.inverse() * fi->T_world_lidar * pt_i.head<3>();

          Eigen::Vector3d nearest_pt;
          if (!grid_j.nearest_neighbor(
                  pi_in_j, nearest_pt,
                  params_.icp_max_correspondence_distance)) {
            continue;
          }

          Eigen::Vector3d nearest_pt_world = fj->T_world_lidar * nearest_pt;

          Eigen::Vector3d residual_vec = pi - nearest_pt_world;
          double residual = residual_vec.norm();

          Eigen::Matrix<double, 1, 6> Ji, Jj;
          Ji.head<3>() = residual_vec.normalized().transpose();
          Ji.tail<3>() =
              -residual_vec.normalized().transpose() * skew_symmetric(pi);
          Jj.head<3>() = -residual_vec.normalized().transpose();
          Jj.tail<3>() =
              residual_vec.normalized().transpose() *
              skew_symmetric(nearest_pt_world);

          int idx_i = static_cast<int>(i * 6);
          int idx_j = static_cast<int>(j * 6);

          H_all.block<6, 6>(idx_i, idx_i) += Ji.transpose() * Ji;
          H_all.block<6, 6>(idx_j, idx_j) += Jj.transpose() * Jj;
          H_all.block<6, 6>(idx_i, idx_j) += Ji.transpose() * Jj;
          H_all.block<6, 6>(idx_j, idx_i) += Jj.transpose() * Ji;

          b_all.segment<6>(idx_i) -= Ji.transpose() * residual;
          b_all.segment<6>(idx_j) -= Jj.transpose() * residual;
        }
      }
    }

    for (int k = 0; k < static_cast<int>(6 * frames_.size()); ++k) {
      H_all(k, k) += lambda;
    }

    Eigen::VectorXd dx = H_all.ldlt().solve(b_all);

    double max_dx = 0.0;
    for (size_t i = 0; i < frames_.size(); ++i) {
      Eigen::Matrix<double, 6, 1> dxi = dx.segment<6>(i * 6);
      dx_stack[i] += dxi;
      max_dx = std::max(max_dx, dxi.norm());
    }

    if (max_dx < 1e-6) break;
  }

  std::vector<EstimationFrame::Ptr> mutable_frames;
  for (const auto& f : frames_) {
    auto mf = std::make_shared<EstimationFrame>(*f);
    mutable_frames.push_back(mf);
  }

  for (size_t i = 0; i < frames_.size(); ++i) {
    mutable_frames[i]->T_world_lidar =
        exp_map(dx_stack[i]) * mutable_frames[i]->T_world_lidar;
    mutable_frames[i]->T_world_imu =
        mutable_frames[i]->T_world_lidar *
        mutable_frames[i]->T_lidar_imu.inverse();
  }

  frames_.clear();
  for (const auto& mf : mutable_frames) {
    frames_.push_back(mf);
  }
}

} // namespace cuda_slam