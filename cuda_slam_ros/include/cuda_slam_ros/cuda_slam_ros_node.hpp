#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <mutex>
#include <deque>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cuda_slam/preprocess/raw_points.hpp>
#include <cuda_slam/preprocess/cloud_preprocessor.hpp>
#include <cuda_slam/odometry/odometry_gpu.hpp>
#include <cuda_slam/mapping/global_localization.hpp>

namespace cuda_slam_ros {

struct ImuInitSample {
  double stamp;
  Eigen::Vector3d linear_acc;
  Eigen::Vector3d angular_vel;
};

class CudaSlamRosNode : public rclcpp::Node {
public:
  explicit CudaSlamRosNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~CudaSlamRosNode() override;

private:
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void odometry_timer_callback();

  void process_imu_init(const sensor_msgs::msg::Imu& msg);
  bool finalize_imu_init();

  bool load_global_map();

  bool run_global_localization(
      const std::vector<Eigen::Vector3d,
                         Eigen::aligned_allocator<Eigen::Vector3d>>& scan_points,
      const Eigen::Isometry3d& initial_pose,
      Eigen::Isometry3d& global_pose);

  static cuda_slam::RawPoints convert_pointcloud(const sensor_msgs::msg::PointCloud2& msg);

  nav_msgs::msg::Odometry build_odometry_msg(
      const cuda_slam::EstimationFrame::ConstPtr& frame) const;

  nav_msgs::msg::Odometry build_global_odometry_msg(
      const Eigen::Isometry3d& pose, double stamp) const;

  geometry_msgs::msg::TransformStamped build_transform_msg(
      const cuda_slam::EstimationFrame::ConstPtr& frame) const;

  void release_gpu_resources();
  void ensure_gpu_scan_buffer(int num_points);

  cuda_slam::CloudPreprocessor::Params preprocess_params_;
  cuda_slam::OdometryParams odometry_params_;
  cuda_slam::GlobalLocalizationParams gloc_params_;

  std::unique_ptr<cuda_slam::CloudPreprocessor> preprocessor_;
  std::unique_ptr<cuda_slam::OdometryEstimationGPU> odometry_;

  std::unique_ptr<cuda_slam::GlobalLocalization> global_localization_;

  Eigen::Isometry3d T_imu_lidar_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;

  rclcpp::TimerBase::SharedPtr odometry_timer_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr global_odom_pub_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::string odom_frame_id_;
  std::string base_frame_id_;
  std::string map_frame_id_;
  std::string map_file_;

  bool publish_tf_;
  bool global_localization_enabled_;

  std::mutex state_mutex_;
  cuda_slam::EstimationFrame::ConstPtr latest_frame_;
  Eigen::Isometry3d latest_pose_;
  Eigen::Vector3d latest_velocity_;
  double latest_stamp_;

  std::mutex gloc_state_mutex_;
  Eigen::Isometry3d latest_global_pose_;
  double latest_global_stamp_;

  std::mutex imu_init_mutex_;
  bool imu_initialized_;
  double imu_init_duration_;
  std::deque<ImuInitSample> imu_init_buffer_;
  Eigen::Vector3d init_gravity_;
  Eigen::Vector3d init_accel_bias_;
  Eigen::Vector3d init_gyro_bias_;

  float4* d_scan_points_;
  int d_scan_capacity_;
};

}  // namespace cuda_slam_ros