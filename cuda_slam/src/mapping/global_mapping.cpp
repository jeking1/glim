#include "cuda_slam/mapping/global_mapping.hpp"

#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <limits>
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

Eigen::Isometry3d exp_map(const Eigen::Matrix<double, 6, 1>& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d phi = xi.tail<3>();
  const double angle = phi.norm();

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  if (angle < 1e-12) {
    T.translation() = rho;
    T.linear() = Eigen::Matrix3d::Identity() + skew_symmetric(phi);
  } else {
    const Eigen::Matrix3d S_mat = skew_symmetric(phi / angle);
    T.linear() = Eigen::Matrix3d::Identity() +
                 std::sin(angle) * S_mat +
                 (1.0 - std::cos(angle)) * S_mat * S_mat;
    const Eigen::Matrix3d V =
        Eigen::Matrix3d::Identity() +
        (1.0 - std::cos(angle)) / angle * S_mat +
        (angle - std::sin(angle)) / angle * S_mat * S_mat;
    T.translation() = V * rho;
  }
  return T;
}

} // namespace

GlobalMapping::GlobalMapping(const GlobalMappingParams& params)
    : params_(params) {
  spdlog::info("GlobalMapping initialized: min_overlap={}, "
               "loop_search_radius={}, lm_max_iter={}",
               params.min_overlap_ratio, params.loop_closure_search_radius,
               params.lm_max_iterations);
}

void GlobalMapping::insert_submap(SubMap::Ptr submap) {
  if (!submap) {
    spdlog::warn("GlobalMapping::insert_submap called with null submap");
    return;
  }

  submap->id = next_submap_id_++;
  submaps_.push_back(submap);
  optimized_poses_.push_back(submap->T_world_origin);

  if (submaps_.size() >= 2) {
    size_t n = submaps_.size();
    PoseGraphEdge edge;
    edge.from_id = static_cast<int>(n - 2);
    edge.to_id = static_cast<int>(n - 1);
    edge.relative_pose =
        submaps_[n - 2]->T_world_origin.inverse() * submaps_[n - 1]->T_world_origin;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.is_loop_closure = false;
    edges_.push_back(edge);
  }

  spdlog::debug("SubMap {} inserted, total: {}", submap->id, submaps_.size());
}

std::vector<std::pair<int, int>> GlobalMapping::find_overlapping_submaps(
    double min_overlap) const {
  std::vector<std::pair<int, int>> overlaps;

  for (size_t i = 0; i < submaps_.size(); ++i) {
    for (size_t j = i + 2; j < submaps_.size(); ++j) {
      if (j - i <= 1) continue;

      Eigen::Vector3d delta =
          optimized_poses_[i].translation() - optimized_poses_[j].translation();
      if (delta.norm() > params_.loop_closure_search_radius) continue;

      double overlap = compute_overlap_ratio(*submaps_[i], *submaps_[j]);
      if (overlap >= min_overlap) {
        overlaps.emplace_back(static_cast<int>(i), static_cast<int>(j));
      }
    }
  }

  return overlaps;
}

bool GlobalMapping::optimize() {
  if (submaps_.size() < 2) {
    spdlog::info("GlobalMapping::optimize: not enough submaps ({}), skipping",
                 submaps_.size());
    return true;
  }

  build_odometry_edges();
  detect_loop_closures();

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      current_poses = optimized_poses_;

  double lambda = params_.lm_lambda_initial;
  double prev_error = std::numeric_limits<double>::max();

  for (int lm_iter = 0; lm_iter < params_.lm_max_iterations; ++lm_iter) {
    Eigen::SparseMatrix<double> H;
    Eigen::VectorXd b;
    double current_error = 0.0;

    build_linear_system(current_poses, H, b, current_error);

    for (int k = 0; k < H.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it) {
        if (it.row() == it.col()) {
          it.valueRef() += lambda;
        }
      }
    }

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(H);

    if (solver.info() != Eigen::Success) {
      spdlog::warn("GlobalMapping: Cholesky decomposition failed at iteration {}",
                   lm_iter);
      lambda *= params_.lm_lambda_factor;
      continue;
    }

    Eigen::VectorXd dx = solver.solve(b);

    if (solver.info() != Eigen::Success) {
      spdlog::warn("GlobalMapping: linear solve failed at iteration {}", lm_iter);
      lambda *= params_.lm_lambda_factor;
      continue;
    }

    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
        proposed_poses;
    proposed_poses.reserve(current_poses.size());

    for (size_t i = 0; i < current_poses.size(); ++i) {
      Eigen::Matrix<double, 6, 1> dxi = dx.segment<6>(static_cast<int>(i) * 6);
      proposed_poses.push_back(exp_map(dxi) * current_poses[i]);
    }

    Eigen::SparseMatrix<double> H_proposed;
    Eigen::VectorXd b_proposed;
    double proposed_error = 0.0;
    build_linear_system(proposed_poses, H_proposed, b_proposed, proposed_error);

    if (proposed_error < current_error) {
      current_poses = std::move(proposed_poses);
      lambda /= params_.lm_lambda_factor;

      double error_change = std::abs(proposed_error - prev_error);
      spdlog::debug("LM iter {}: error={:.6f}, lambda={:.6f}",
                    lm_iter, proposed_error, lambda);

      if (error_change < params_.lm_convergence_threshold) {
        spdlog::info("GlobalMapping converged at iteration {}: error={:.6f}",
                     lm_iter, proposed_error);
        optimized_poses_ = std::move(current_poses);

        for (size_t i = 0; i < submaps_.size(); ++i) {
          submaps_[i]->T_world_origin = optimized_poses_[i];
        }

        return true;
      }

      prev_error = proposed_error;
    } else {
      lambda *= params_.lm_lambda_factor;
      spdlog::debug("LM iter {}: rejected, lambda increased to {:.6f}",
                    lm_iter, lambda);
    }
  }

  spdlog::info("GlobalMapping optimization finished: max iterations reached");
  optimized_poses_ = std::move(current_poses);

  for (size_t i = 0; i < submaps_.size(); ++i) {
    submaps_[i]->T_world_origin = optimized_poses_[i];
  }

  return true;
}

std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
GlobalMapping::export_points() const {
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> all_points;

  size_t total = 0;
  for (const auto& submap : submaps_) {
    total += submap->merged_points.size();
  }
  all_points.reserve(total);

  for (const auto& submap : submaps_) {
    const Eigen::Isometry3d& T = submap->T_world_origin;
    for (const auto& pt : submap->merged_points) {
      all_points.push_back(T * pt);
    }
  }

  return all_points;
}

const std::vector<SubMap::Ptr>& GlobalMapping::get_submaps() const {
  return submaps_;
}

const std::vector<PoseGraphEdge>& GlobalMapping::get_edges() const {
  return edges_;
}

size_t GlobalMapping::submap_count() const {
  return submaps_.size();
}

size_t GlobalMapping::edge_count() const {
  return edges_.size();
}

void GlobalMapping::build_odometry_edges() {
  edges_.clear();

  for (size_t i = 0; i + 1 < submaps_.size(); ++i) {
    PoseGraphEdge edge;
    edge.from_id = static_cast<int>(i);
    edge.to_id = static_cast<int>(i + 1);
    edge.relative_pose =
        submaps_[i]->T_world_origin.inverse() * submaps_[i + 1]->T_world_origin;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.is_loop_closure = false;
    edges_.push_back(edge);
  }
}

void GlobalMapping::detect_loop_closures() {
  if (submaps_.size() < 3) return;

  auto overlaps = find_overlapping_submaps(params_.min_overlap_ratio);

  for (const auto& pair : overlaps) {
    int i = pair.first;
    int j = pair.second;

    const auto& submap_i = submaps_[i];
    const auto& submap_j = submaps_[j];

    if (submap_i->merged_points.empty() || submap_j->merged_points.empty()) {
      continue;
    }

    Eigen::Isometry3d T_ij_init =
        optimized_poses_[i].inverse() * optimized_poses_[j];

    Eigen::Matrix<double, 6, 6> H_icp = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b_icp = Eigen::Matrix<double, 6, 1>::Zero();
    int num_correspondences = 0;
    double total_error = 0.0;

    const int max_icp_iter = params_.icp_max_iterations;
    Eigen::Isometry3d T_ij = T_ij_init;

    for (int icp_iter = 0; icp_iter < max_icp_iter; ++icp_iter) {
      H_icp.setZero();
      b_icp.setZero();
      num_correspondences = 0;
      total_error = 0.0;

      const double max_dist_sq =
          params_.icp_max_correspondence_distance *
          params_.icp_max_correspondence_distance;

      for (const auto& pt_i : submap_i->merged_points) {
        Eigen::Vector3d pi = T_ij * pt_i.head<3>();

        double best_dist = max_dist_sq;
        Eigen::Vector3d best_pt = Eigen::Vector3d::Zero();
        bool found = false;

        for (const auto& pt_j : submap_j->merged_points) {
          Eigen::Vector3d pj = pt_j.head<3>();
          double dist = (pi - pj).squaredNorm();
          if (dist < best_dist) {
            best_dist = dist;
            best_pt = pj;
            found = true;
          }
        }

        if (!found) continue;

        Eigen::Vector3d residual_vec = pi - best_pt;

        Eigen::Matrix<double, 1, 6> J;
        J.head<3>() = residual_vec.normalized().transpose();
        J.tail<3>() = -residual_vec.normalized().transpose() * skew_symmetric(pi);

        double residual = residual_vec.norm();

        H_icp += J.transpose() * J;
        b_icp -= J.transpose() * residual;
        total_error += residual * residual;
        ++num_correspondences;
      }

      if (num_correspondences < 10) break;

      H_icp += Eigen::Matrix<double, 6, 6>::Identity() * 1e-3;

      Eigen::Matrix<double, 6, 1> dx = H_icp.ldlt().solve(b_icp);
      T_ij = exp_map(dx) * T_ij;

      if (dx.norm() < params_.icp_transformation_epsilon) break;
    }

    if (num_correspondences >= 10) {
      PoseGraphEdge edge;
      edge.from_id = i;
      edge.to_id = j;
      edge.relative_pose = T_ij;
      edge.information = H_icp;
      edge.is_loop_closure = true;
      edges_.push_back(edge);

      spdlog::info("Loop closure detected: submap {} <-> {} ({} correspondences)",
                   i, j, num_correspondences);
    }
  }
}

double GlobalMapping::compute_overlap_ratio(const SubMap& a,
                                             const SubMap& b) const {
  if (a.merged_points.empty() || b.merged_points.empty()) {
    return 0.0;
  }

  Eigen::Isometry3d T_ab =
      a.T_world_origin.inverse() * b.T_world_origin;

  const double dist_threshold = params_.voxel_resolution * 2.0;
  const double dist_sq = dist_threshold * dist_threshold;

  int overlapping = 0;
  const size_t sample_size = std::min(a.merged_points.size(), size_t(1000));

  for (size_t i = 0; i < sample_size; ++i) {
    size_t idx = (i * a.merged_points.size()) / sample_size;
    Eigen::Vector3d pt_a_local = a.merged_points[idx].head<3>();
    Eigen::Vector3d pt_a_in_b = T_ab * pt_a_local;

    for (const auto& pt_b : b.merged_points) {
      if ((pt_a_in_b - pt_b.head<3>()).squaredNorm() < dist_sq) {
        ++overlapping;
        break;
      }
    }
  }

  return static_cast<double>(overlapping) / static_cast<double>(sample_size);
}

Eigen::Matrix<double, 6, 1> GlobalMapping::compute_relative_error(
    const Eigen::Isometry3d& T_i,
    const Eigen::Isometry3d& T_j,
    const Eigen::Isometry3d& T_ij) const {
  Eigen::Isometry3d T_error = T_ij.inverse() * T_i.inverse() * T_j;
  return log_map(T_error);
}

void GlobalMapping::build_linear_system(
    const std::vector<Eigen::Isometry3d>& poses,
    Eigen::SparseMatrix<double>& H,
    Eigen::VectorXd& b,
    double& total_error) const {
  const int n = static_cast<int>(poses.size());
  const int dim = 6;

  H.resize(n * dim, n * dim);
  b.resize(n * dim);
  b.setZero();

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(edges_.size() * dim * dim * 2);

  total_error = 0.0;

  for (const auto& edge : edges_) {
    if (edge.from_id >= n || edge.to_id >= n) continue;

    const auto& T_i = poses[edge.from_id];
    const auto& T_j = poses[edge.to_id];

    Eigen::Matrix<double, 6, 1> error_vec =
        compute_relative_error(T_i, T_j, edge.relative_pose);

    double error = error_vec.squaredNorm();
    total_error += error;

    Eigen::Isometry3d T_ij_hat = T_i.inverse() * T_j;
    Eigen::Matrix<double, 6, 6> Adj_T_ij_inv =
        Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix<double, 6, 6> J_i = -Adj_T_ij_inv;
    Eigen::Matrix<double, 6, 6> J_j = Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix<double, 6, 6> info = edge.information;
    if (edge.is_loop_closure) {
      info *= 0.1;
    }

    Eigen::Matrix<double, 6, 6> H_ii = J_i.transpose() * info * J_i;
    Eigen::Matrix<double, 6, 6> H_ij = J_i.transpose() * info * J_j;
    Eigen::Matrix<double, 6, 6> H_jj = J_j.transpose() * info * J_j;

    Eigen::Matrix<double, 6, 1> b_i = -J_i.transpose() * info * error_vec;
    Eigen::Matrix<double, 6, 1> b_j = -J_j.transpose() * info * error_vec;

    int idx_i = edge.from_id * dim;
    int idx_j = edge.to_id * dim;

    for (int r = 0; r < dim; ++r) {
      for (int c = 0; c < dim; ++c) {
        if (std::abs(H_ii(r, c)) > 1e-12) {
          triplets.emplace_back(idx_i + r, idx_i + c, H_ii(r, c));
        }
        if (std::abs(H_ij(r, c)) > 1e-12) {
          triplets.emplace_back(idx_i + r, idx_j + c, H_ij(r, c));
          triplets.emplace_back(idx_j + r, idx_i + c, H_ij(c, r));
        }
        if (std::abs(H_jj(r, c)) > 1e-12) {
          triplets.emplace_back(idx_j + r, idx_j + c, H_jj(r, c));
        }
      }
    }

    for (int r = 0; r < dim; ++r) {
      b(idx_i + r) += b_i(r);
      b(idx_j + r) += b_j(r);
    }
  }

  H.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::Matrix<double, 6, 6> anchor_info =
      Eigen::Matrix<double, 6, 6>::Identity() * 1e6;
  for (int r = 0; r < dim; ++r) {
    triplets.emplace_back(r, r, anchor_info(r, r));
  }
  H.setFromTriplets(triplets.begin(), triplets.end());
}

} // namespace cuda_slam