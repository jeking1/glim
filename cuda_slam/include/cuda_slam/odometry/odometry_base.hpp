#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace cuda_slam {

struct PreprocessedFrame;
struct EstimationFrame;

class OdometryEstimationBase {
public:
  OdometryEstimationBase() = default;
  virtual ~OdometryEstimationBase() = default;

  OdometryEstimationBase(const OdometryEstimationBase&) = delete;
  OdometryEstimationBase& operator=(const OdometryEstimationBase&) = delete;
  OdometryEstimationBase(OdometryEstimationBase&&) = delete;
  OdometryEstimationBase& operator=(OdometryEstimationBase&&) = delete;

  virtual void insert_imu(double stamp,
                          const Eigen::Vector3d& linear_acc,
                          const Eigen::Vector3d& angular_vel) = 0;

  virtual EstimationFrame::ConstPtr insert_frame(
      std::shared_ptr<PreprocessedFrame> frame) = 0;

  virtual std::vector<EstimationFrame::ConstPtr> get_remaining_frames()
      const = 0;

  virtual bool requires_imu() const = 0;
};

} // namespace cuda_slam