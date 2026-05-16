#pragma once

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <vector>

#include "cuda_slam/util/callback_slot.hpp"

namespace cuda_slam {

struct PreprocessedFrame;
struct EstimationFrame;

struct OdometryCallbacks {
  using CallbackSlot_Imu =
      util::CallbackSlot<void(double, const Eigen::Vector3d&, const Eigen::Vector3d&)>;
  using CallbackSlot_Frame =
      util::CallbackSlot<void(std::shared_ptr<PreprocessedFrame>)>;
  using CallbackSlot_EstFrame =
      util::CallbackSlot<void(EstimationFrame::ConstPtr)>;
  using CallbackSlot_EstFrameVec =
      util::CallbackSlot<void(const std::vector<EstimationFrame::ConstPtr>&)>;

  static CallbackSlot_Imu on_insert_imu;
  static CallbackSlot_Frame on_insert_frame;
  static CallbackSlot_EstFrame on_new_frame;
  static CallbackSlot_EstFrameVec on_update_frames;
  static CallbackSlot_EstFrameVec on_marginalized_frames;
  static CallbackSlot_EstFrameVec on_update_keyframes;
};

} // namespace cuda_slam