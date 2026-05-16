#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_slam/common/cloud_covariance.hpp"
#include "cuda_slam/common/cloud_deskewing.hpp"
#include "cuda_slam/common/imu_integration.hpp"
#include "cuda_slam/odometry/estimation_frame.hpp"
#include "cuda_slam/odometry/odometry_base.hpp"

namespace cuda_slam {
namespace cuda {
class CudaVoxelGrid;
class CudaICP;
}

enum class KeyframeStrategy {
  OVERLAP,
  DISPLACEMENT,
};

struct OdometryParams {
  double voxel_resolution = 0.5;
  int voxelmap_levels = 3;
  double voxelmap_scaling_factor = 2.0;
  int smoother_lag = 5;
  int max_num_keyframes = 50;
  KeyframeStrategy keyframe_strategy = KeyframeStrategy::OVERLAP;
  double keyframe_min_overlap = 0.2;
  double keyframe_max_overlap = 0.8;
  double keyframe_delta_trans = 1.0;
  double keyframe_delta_rot = 0.2;
  int num_threads = 4;

  double imu_gravity = 9.80665;
  double icp_max_correspondence_distance = 2.0;
  int icp_max_iterations = 30;
  double icp_transformation_epsilon = 1e-6;
  double icp_euclidean_fitness_epsilon = 1e-6;
  int deskew_num_threads = 4;
  int covariance_num_neighbors = 20;
  double covariance_max_distance = 2.0;
};

class OdometryEstimationGPU : public OdometryEstimationBase {
public:
  explicit OdometryEstimationGPU(const OdometryParams& params);
  ~OdometryEstimationGPU() override;

  void insert_imu(double stamp,
                  const Eigen::Vector3d& linear_acc,
                  const Eigen::Vector3d& angular_vel) override;

  EstimationFrame::ConstPtr insert_frame(
      std::shared_ptr<PreprocessedFrame> frame) override;

  std::vector<EstimationFrame::ConstPtr> get_remaining_frames()
      const override;

  bool requires_imu() const override { return true; }

  const std::vector<EstimationFrame::ConstPtr>& get_keyframes() const;

  void set_params(const OdometryParams& params);

private:
  struct ImuState {
    Eigen::Isometry3d T_world_imu = Eigen::Isometry3d::Identity();
    Eigen::Vector3d v_world_imu = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 6, 1> bias = Eigen::Matrix<double, 6, 1>::Zero();
  };

  Eigen::Isometry3d predict_pose(const PreprocessedFrame& frame) const;
  void deskew_cloud(const PreprocessedFrame& raw,
                    const Eigen::Isometry3d& T_start,
                    const Eigen::Isometry3d& T_end,
                    PointCloudCPU& deskewed) const;
  void estimate_covariances(PointCloudCPU& cloud,
                             const PreprocessedFrame& raw) const;
  void build_voxel_grids(EstimationFrame& ef) const;
  Eigen::Isometry3d icp_align(
      const PointCloudCPU& source,
      const std::vector<PointCloudCPU>& target_clouds,
      const Eigen::Isometry3d& initial_guess) const;
  void update_sliding_window(EstimationFrame::ConstPtr new_frame);
  void manage_keyframes(EstimationFrame::ConstPtr new_frame);
  bool is_keyframe_overlap(EstimationFrame::ConstPtr frame) const;
  bool is_keyframe_displacement(EstimationFrame::ConstPtr frame) const;
  void marginalize_old_frames();
  void optimize_sliding_window();

  OdometryParams params_;
  mutable std::mutex imu_mutex_;
  ImuState imu_state_;

  mutable std::mutex frames_mutex_;
  std::deque<EstimationFrame::ConstPtr> frames_;
  std::vector<EstimationFrame::ConstPtr> keyframes_;
  size_t marginalized_cursor_ = 0;

  IMUIntegration imu_integrator_;
  CloudDeskewing deskewer_;
  CloudCovarianceEstimation covariance_estimator_;

  long frame_counter_ = 0;
};

} // namespace cuda_slam