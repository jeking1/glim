#include "cuda_slam/odometry/callbacks.hpp"

namespace cuda_slam {

OdometryCallbacks::CallbackSlot_Imu OdometryCallbacks::on_insert_imu;

OdometryCallbacks::CallbackSlot_Frame OdometryCallbacks::on_insert_frame;

OdometryCallbacks::CallbackSlot_EstFrame OdometryCallbacks::on_new_frame;

OdometryCallbacks::CallbackSlot_EstFrameVec OdometryCallbacks::on_update_frames;

OdometryCallbacks::CallbackSlot_EstFrameVec OdometryCallbacks::on_marginalized_frames;

OdometryCallbacks::CallbackSlot_EstFrameVec OdometryCallbacks::on_update_keyframes;

} // namespace cuda_slam