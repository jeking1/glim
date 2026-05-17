#include "cuda_slam_ros/cuda_slam_ros_node.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace cuda_slam_ros {

namespace {

double stamp_to_sec(const builtin_interfaces::msg::Time& t) {
  return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

builtin_interfaces::msg::Time sec_to_stamp(double t) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(std::floor(t));
  stamp.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
  return stamp;
}

constexpr int POINT_FIELD_FLOAT32 = 7;
constexpr int POINT_FIELD_FLOAT64 = 8;
constexpr int POINT_FIELD_UINT16 = 4;
constexpr int POINT_FIELD_UINT32 = 6;
constexpr int POINT_FIELD_UINT8 = 2;

static bool load_pcd_file(
    const std::string& filepath,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points) {

  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    spdlog::error("[cuda_slam_ros] Cannot open map file: {}", filepath);
    return false;
  }

  std::unordered_map<std::string, std::string> fields;
  int width = 0, height = 0, num_points = 0;
  bool is_binary = false;
  bool header_done = false;

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    if (line.find("FIELDS") == 0) {
      std::istringstream iss(line);
      std::string token, field_name;
      iss >> token;
      int idx = 0;
      while (iss >> field_name) {
        fields[field_name] = std::to_string(idx++);
      }
    } else if (line.find("SIZE") == 0) {
      std::istringstream iss(line);
      std::string token;
      iss >> token;
      int idx = 0;
      while (iss >> token) {
        for (auto& [name, value] : fields) {
          if (value == std::to_string(idx)) {
            value = "size_" + token;
            break;
          }
        }
        idx++;
      }
    } else if (line.find("TYPE") == 0) {
      std::istringstream iss(line);
      std::string token;
      iss >> token;
      int idx = 0;
      while (iss >> token) {
        for (auto& [name, value] : fields) {
          if (value.find("size_") == 0) {
            if (std::to_string(idx) == value.substr(value.rfind('_') + 1) ||
                value == std::to_string(idx)) {
              value += "_" + token;
              break;
            }
          }
        }
        idx++;
      }
    } else if (line.find("WIDTH") == 0) {
      std::istringstream iss(line);
      std::string token;
      iss >> token >> width;
    } else if (line.find("HEIGHT") == 0) {
      std::istringstream iss(line);
      std::string token;
      iss >> token >> height;
    } else if (line.find("POINTS") == 0) {
      std::istringstream iss(line);
      std::string token;
      iss >> token >> num_points;
    } else if (line.find("DATA") == 0) {
      std::istringstream iss(line);
      std::string token, data_type;
      iss >> token >> data_type;
      is_binary = (data_type == "binary" || data_type == "binary_compressed");
      header_done = true;
      break;
    }
  }

  if (!header_done) {
    spdlog::error("[cuda_slam_ros] Incomplete PCD header in: {}", filepath);
    return false;
  }

  num_points = (num_points > 0) ? num_points : width * height;
  if (num_points <= 0) {
    spdlog::error("[cuda_slam_ros] Zero points in PCD file");
    return false;
  }

  bool has_x = fields.count("x") > 0;
  bool has_y = fields.count("y") > 0;
  bool has_z = fields.count("z") > 0;

  if (!has_x || !has_y || !has_z) {
    spdlog::error("[cuda_slam_ros] PCD file missing x/y/z fields");
    return false;
  }

  points.resize(num_points);

  int point_size = 0;
  for (const auto& [name, info] : fields) {
    std::string size_str;
    size_t pos = info.find("size_");
    if (pos != std::string::npos) {
      size_t end = info.find('_', pos + 5);
      if (end != std::string::npos) {
        size_str = info.substr(pos + 5, end - pos - 5);
      } else {
        size_str = info.substr(pos + 5);
      }
      point_size += std::stoi(size_str);
    }
  }

  if (point_size == 0) {
    point_size = 4 * static_cast<int>(fields.size());
  }

  if (is_binary) {
    char peek_char;
    file.get(peek_char);
    if (peek_char != '\n') {
      file.putback(peek_char);
    }

    std::vector<uint8_t> raw_data(num_points * point_size);
    file.read(reinterpret_cast<char*>(raw_data.data()), raw_data.size());

    if (!file) {
      spdlog::error("[cuda_slam_ros] Failed to read PCD binary data");
      return false;
    }

    for (int i = 0; i < num_points; i++) {
      float x = 0.0f, y = 0.0f, z = 0.0f;

      for (const auto& [name, info] : fields) {
        if (name == "x" || name == "y" || name == "z") {
          size_t field_idx = 0;
          for (const auto& [n, v] : fields) {
            if (n == name) break;
            field_idx++;
          }

          int offset = 0;
          int idx = 0;
          for (const auto& [n, info2] : fields) {
            if (idx >= static_cast<int>(field_idx)) break;
            std::string sz;
            size_t p = info2.find("size_");
            if (p != std::string::npos) {
              size_t e = info2.find('_', p + 5);
              if (e != std::string::npos) {
                sz = info2.substr(p + 5, e - p - 5);
              } else {
                sz = info2.substr(p + 5);
              }
              offset += std::stoi(sz);
            }
            idx++;
          }

          std::string type_str;
          size_t tpos = info.find("size_");
          if (tpos != std::string::npos) {
            size_t tend = info.find('_', tpos + 5);
            if (tend != std::string::npos) {
              type_str = info.substr(tend + 1);
            }
          }

          const uint8_t* ptr = raw_data.data() + i * point_size + offset;
          if (type_str == "F" || type_str == "f") {
            float val = *reinterpret_cast<const float*>(ptr);
            if (name == "x") x = val;
            else if (name == "y") y = val;
            else if (name == "z") z = val;
          } else if (type_str == "D" || type_str == "d") {
            double val = *reinterpret_cast<const double*>(ptr);
            if (name == "x") x = static_cast<float>(val);
            else if (name == "y") y = static_cast<float>(val);
            else if (name == "z") z = static_cast<float>(val);
          }
        }
      }
      points[i] = Eigen::Vector3d(x, y, z);
    }
  } else {
    file.clear();
    file.seekg(0);
    for (int i = 0; i < num_points; i++) {
      while (file.peek() == '#' || file.peek() == '\n') {
        std::string skip;
        std::getline(file, skip);
        if (file.eof()) break;
      }
      if (file.eof()) break;

      float x = 0.0f, y = 0.0f, z = 0.0f;
      std::vector<float> values(fields.size());
      for (size_t j = 0; j < fields.size(); j++) {
        file >> values[j];
      }

      int idx = 0;
      for (const auto& [name, info] : fields) {
        if (name == "x") x = values[idx];
        else if (name == "y") y = values[idx];
        else if (name == "z") z = values[idx];
        idx++;
      }

      points[i] = Eigen::Vector3d(x, y, z);
    }
  }

  size_t valid = 0;
  for (const auto& p : points) {
    if (p.norm() > 0.01 || p.x() != 0 || p.y() != 0 || p.z() != 0) {
      valid++;
    }
  }

  spdlog::info("[cuda_slam_ros] Loaded {} points ({} valid) from {}",
               points.size(), valid, filepath);
  return valid > 0;
}

}  // namespace

CudaSlamRosNode::CudaSlamRosNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("cuda_slam_ros", options),
      imu_initialized_(false),
      d_scan_points_(nullptr),
      d_scan_capacity_(0) {

  this->declare_parameter("odom_frame_id", "odom");
  this->declare_parameter("base_frame_id", "base_link");
  this->declare_parameter("map_frame_id", "map");
  this->declare_parameter("publish_tf", true);
  this->declare_parameter("imu_init_duration", 2.0);

  this->declare_parameter("global_localization.enabled", false);
  this->declare_parameter("global_localization.map_file", "");

  this->declare_parameter("T_imu_lidar.x", 0.0);
  this->declare_parameter("T_imu_lidar.y", 0.0);
  this->declare_parameter("T_imu_lidar.z", 0.0);
  this->declare_parameter("T_imu_lidar.roll", 0.0);
  this->declare_parameter("T_imu_lidar.pitch", 0.0);
  this->declare_parameter("T_imu_lidar.yaw", 0.0);

  this->declare_parameter("preprocess.distance_near", 0.5);
  this->declare_parameter("preprocess.distance_far", 100.0);
  this->declare_parameter("preprocess.downsample_resolution", 0.2);
  this->declare_parameter("preprocess.num_threads", 4);

  this->declare_parameter("odometry.voxel_resolution", 1.0);
  this->declare_parameter("odometry.voxelmap_levels", 3);
  this->declare_parameter("odometry.voxelmap_scaling_factor", 2.0);
  this->declare_parameter("odometry.smoother_lag", 5);
  this->declare_parameter("odometry.max_num_keyframes", 50);
  this->declare_parameter("odometry.keyframe_delta_trans", 1.0);
  this->declare_parameter("odometry.keyframe_delta_rot", 0.2);
  this->declare_parameter("odometry.num_threads", 4);
  this->declare_parameter("odometry.imu_gravity", 9.80665);
  this->declare_parameter("odometry.icp_max_correspondence_distance", 2.0);
  this->declare_parameter("odometry.icp_max_iterations", 30);
  this->declare_parameter("odometry.icp_transformation_epsilon", 1e-6);
  this->declare_parameter("odometry.icp_euclidean_fitness_epsilon", 1e-6);
  this->declare_parameter("odometry.covariance_num_neighbors", 20);
  this->declare_parameter("odometry.covariance_max_distance", 2.0);

  this->declare_parameter("global_localization.voxelmap_levels", 3);
  this->declare_parameter("global_localization.voxel_resolution_base", 0.5);
  this->declare_parameter("global_localization.voxelmap_scaling_factor", 2.0);
  this->declare_parameter("global_localization.icp_max_iterations", 30);
  this->declare_parameter("global_localization.icp_max_correspondence_distance", 5.0);
  this->declare_parameter("global_localization.icp_transformation_epsilon", 1e-5);
  this->declare_parameter("global_localization.initial_pose_search_radius", 10.0);
  this->declare_parameter("global_localization.initial_pose_search_angular_range", 0.5);

  odom_frame_id_ = this->get_parameter("odom_frame_id").as_string();
  base_frame_id_ = this->get_parameter("base_frame_id").as_string();
  map_frame_id_ = this->get_parameter("map_frame_id").as_string();
  publish_tf_ = this->get_parameter("publish_tf").as_bool();
  imu_init_duration_ = this->get_parameter("imu_init_duration").as_double();

  global_localization_enabled_ =
      this->get_parameter("global_localization.enabled").as_bool();
  map_file_ = this->get_parameter("global_localization.map_file").as_string();

  double tx = this->get_parameter("T_imu_lidar.x").as_double();
  double ty = this->get_parameter("T_imu_lidar.y").as_double();
  double tz = this->get_parameter("T_imu_lidar.z").as_double();
  double roll = this->get_parameter("T_imu_lidar.roll").as_double();
  double pitch = this->get_parameter("T_imu_lidar.pitch").as_double();
  double yaw = this->get_parameter("T_imu_lidar.yaw").as_double();

  T_imu_lidar_ = Eigen::Isometry3d::Identity();
  T_imu_lidar_.translation() = Eigen::Vector3d(tx, ty, tz);
  T_imu_lidar_.linear() =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();

  preprocess_params_.distance_near =
      this->get_parameter("preprocess.distance_near").as_double();
  preprocess_params_.distance_far =
      this->get_parameter("preprocess.distance_far").as_double();
  preprocess_params_.downsample_resolution =
      this->get_parameter("preprocess.downsample_resolution").as_double();
  preprocess_params_.num_threads =
      this->get_parameter("preprocess.num_threads").as_int();

  odometry_params_.voxel_resolution =
      this->get_parameter("odometry.voxel_resolution").as_double();
  odometry_params_.voxelmap_levels =
      this->get_parameter("odometry.voxelmap_levels").as_int();
  odometry_params_.voxelmap_scaling_factor =
      this->get_parameter("odometry.voxelmap_scaling_factor").as_double();
  odometry_params_.smoother_lag =
      this->get_parameter("odometry.smoother_lag").as_int();
  odometry_params_.max_num_keyframes =
      this->get_parameter("odometry.max_num_keyframes").as_int();
  odometry_params_.keyframe_delta_trans =
      this->get_parameter("odometry.keyframe_delta_trans").as_double();
  odometry_params_.keyframe_delta_rot =
      this->get_parameter("odometry.keyframe_delta_rot").as_double();
  odometry_params_.num_threads =
      this->get_parameter("odometry.num_threads").as_int();
  odometry_params_.imu_gravity =
      this->get_parameter("odometry.imu_gravity").as_double();
  odometry_params_.icp_max_correspondence_distance =
      this->get_parameter("odometry.icp_max_correspondence_distance").as_double();
  odometry_params_.icp_max_iterations =
      this->get_parameter("odometry.icp_max_iterations").as_int();
  odometry_params_.icp_transformation_epsilon =
      this->get_parameter("odometry.icp_transformation_epsilon").as_double();
  odometry_params_.icp_euclidean_fitness_epsilon =
      this->get_parameter("odometry.icp_euclidean_fitness_epsilon").as_double();
  odometry_params_.covariance_num_neighbors =
      this->get_parameter("odometry.covariance_num_neighbors").as_int();
  odometry_params_.covariance_max_distance =
      this->get_parameter("odometry.covariance_max_distance").as_double();

  gloc_params_.voxelmap_levels =
      this->get_parameter("global_localization.voxelmap_levels").as_int();
  gloc_params_.voxel_resolution_base =
      this->get_parameter("global_localization.voxel_resolution_base").as_double();
  gloc_params_.voxelmap_scaling_factor =
      this->get_parameter("global_localization.voxelmap_scaling_factor").as_double();
  gloc_params_.icp_max_iterations =
      this->get_parameter("global_localization.icp_max_iterations").as_int();
  gloc_params_.icp_max_correspondence_distance =
      this->get_parameter("global_localization.icp_max_correspondence_distance").as_double();
  gloc_params_.icp_transformation_epsilon =
      this->get_parameter("global_localization.icp_transformation_epsilon").as_double();
  gloc_params_.initial_pose_search_radius =
      this->get_parameter("global_localization.initial_pose_search_radius").as_double();
  gloc_params_.initial_pose_search_angular_range =
      this->get_parameter("global_localization.initial_pose_search_angular_range").as_double();

  preprocessor_ = std::make_unique<cuda_slam::CloudPreprocessor>(preprocess_params_);
  odometry_ = std::make_unique<cuda_slam::OdometryEstimationGPU>(odometry_params_);

  global_localization_ = std::make_unique<cuda_slam::GlobalLocalization>(gloc_params_);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  auto qos = rclcpp::QoS(rclcpp::KeepLast(100));
  qos.reliable();

  auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(1000));
  imu_qos.best_effort();

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu", imu_qos,
      std::bind(&CudaSlamRosNode::imu_callback, this, std::placeholders::_1));

  points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "points", qos,
      std::bind(&CudaSlamRosNode::pointcloud_callback, this, std::placeholders::_1));

  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "odometry", rclcpp::QoS(100));

  global_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "global_odometry", rclcpp::QoS(100));

  odometry_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&CudaSlamRosNode::odometry_timer_callback, this));

  latest_pose_ = Eigen::Isometry3d::Identity();
  latest_velocity_ = Eigen::Vector3d::Zero();
  latest_stamp_ = 0.0;

  latest_global_pose_ = Eigen::Isometry3d::Identity();
  latest_global_stamp_ = 0.0;

  spdlog::info("[cuda_slam_ros] Node initialized");
  spdlog::info("[cuda_slam_ros] Waiting for IMU initialization ({} s)...",
               imu_init_duration_);

  if (global_localization_enabled_) {
    spdlog::info("[cuda_slam_ros] Global localization enabled");
    if (!map_file_.empty()) {
      if (load_global_map()) {
        spdlog::info("[cuda_slam_ros] Global map loaded: {} points, {} voxels",
                     global_localization_->map_point_count(),
                     global_localization_->map_voxel_count());
      } else {
        spdlog::error("[cuda_slam_ros] Failed to load global map, "
                      "global localization disabled");
        global_localization_enabled_ = false;
      }
    } else {
      spdlog::warn("[cuda_slam_ros] Global localization enabled but no map_file "
                   "specified, global localization disabled");
      global_localization_enabled_ = false;
    }
  }
}

CudaSlamRosNode::~CudaSlamRosNode() {
  release_gpu_resources();
}

void CudaSlamRosNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  double stamp = stamp_to_sec(msg->header.stamp);

  Eigen::Vector3d linear_acc(
      msg->linear_acceleration.x,
      msg->linear_acceleration.y,
      msg->linear_acceleration.z);

  Eigen::Vector3d angular_vel(
      msg->angular_velocity.x,
      msg->angular_velocity.y,
      msg->angular_velocity.z);

  if (!imu_initialized_) {
    process_imu_init(*msg);
    return;
  }

  linear_acc -= init_accel_bias_;
  angular_vel -= init_gyro_bias_;

  odometry_->insert_imu(stamp, linear_acc, angular_vel);
}

void CudaSlamRosNode::pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

  if (!imu_initialized_) {
    spdlog::warn("[cuda_slam_ros] IMU not initialized yet, skipping point cloud");
    return;
  }

  auto raw = std::make_shared<cuda_slam::RawPoints>();
  *raw = convert_pointcloud(*msg);
  if (raw->empty()) {
    return;
  }

  auto preprocessed = preprocessor_->preprocess(*raw, T_imu_lidar_);
  auto frame = odometry_->insert_frame(preprocessed);

  if (frame) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_frame_ = frame;
    latest_pose_ = frame->T_world_imu;
    latest_velocity_ = frame->v_world_imu;
    latest_stamp_ = frame->stamp;
  }

  if (global_localization_enabled_ &&
      global_localization_->is_map_loaded() && frame) {

    Eigen::Isometry3d odom_pose = frame->T_world_lidar;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        scan_xyz;
    scan_xyz.reserve(preprocessed->size());
    for (size_t i = 0; i < preprocessed->size(); i++) {
      const auto& pt = preprocessed->points[i];
      Eigen::Vector3d p_lidar =
          T_imu_lidar_.inverse() * Eigen::Vector3d(pt.x(), pt.y(), pt.z());
      Eigen::Vector3d p_world = odom_pose * p_lidar;
      scan_xyz.push_back(p_world);
    }

    Eigen::Isometry3d global_pose;
    if (run_global_localization(scan_xyz, odom_pose, global_pose)) {
      std::lock_guard<std::mutex> lock(gloc_state_mutex_);
      latest_global_pose_ = global_pose;
      latest_global_stamp_ = frame->stamp;
    }
  }
}

void CudaSlamRosNode::odometry_timer_callback() {
  {
    Eigen::Isometry3d pose;
    Eigen::Vector3d velocity;
    double stamp;
    cuda_slam::EstimationFrame::ConstPtr frame;

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!latest_frame_) goto publish_global;
      pose = latest_pose_;
      velocity = latest_velocity_;
      stamp = latest_stamp_;
      frame = latest_frame_;
    }

    auto odom_msg = build_odometry_msg(frame);
    odom_msg.header.stamp = this->now();
    odom_msg.header.frame_id = odom_frame_id_;
    odom_msg.child_frame_id = base_frame_id_;

    odom_pub_->publish(odom_msg);

    if (publish_tf_) {
      auto tf_msg = build_transform_msg(frame);
      tf_msg.header.stamp = this->now();
      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

publish_global:
  {
    std::lock_guard<std::mutex> lock(gloc_state_mutex_);
    if (latest_global_stamp_ > 0.0) {
      auto global_odom = build_global_odometry_msg(
          latest_global_pose_, latest_global_stamp_);
      global_odom.header.stamp = this->now();
      global_odom.header.frame_id = map_frame_id_;
      global_odom.child_frame_id = base_frame_id_;
      global_odom_pub_->publish(global_odom);
    }
  }
}

void CudaSlamRosNode::process_imu_init(const sensor_msgs::msg::Imu& msg) {
  std::lock_guard<std::mutex> lock(imu_init_mutex_);

  ImuInitSample sample;
  sample.stamp = stamp_to_sec(msg.header.stamp);
  sample.linear_acc = Eigen::Vector3d(
      msg.linear_acceleration.x,
      msg.linear_acceleration.y,
      msg.linear_acceleration.z);
  sample.angular_vel = Eigen::Vector3d(
      msg.angular_velocity.x,
      msg.angular_velocity.y,
      msg.angular_velocity.z);

  imu_init_buffer_.push_back(sample);

  while (!imu_init_buffer_.empty() &&
         sample.stamp - imu_init_buffer_.front().stamp > imu_init_duration_) {
    imu_init_buffer_.pop_front();
  }

  if (sample.stamp - imu_init_buffer_.front().stamp >= imu_init_duration_ &&
      imu_init_buffer_.size() > 50) {
    if (finalize_imu_init()) {
      imu_initialized_ = true;
      spdlog::info("[cuda_slam_ros] IMU initialized successfully");
      spdlog::info("[cuda_slam_ros] Gravity vector: [{:.3f}, {:.3f}, {:.3f}]",
                   init_gravity_.x(), init_gravity_.y(), init_gravity_.z());
      spdlog::info("[cuda_slam_ros] Accel bias: [{:.4f}, {:.4f}, {:.4f}]",
                   init_accel_bias_.x(), init_accel_bias_.y(), init_accel_bias_.z());
      spdlog::info("[cuda_slam_ros] Gyro bias: [{:.4f}, {:.4f}, {:.4f}]",
                   init_gyro_bias_.x(), init_gyro_bias_.y(), init_gyro_bias_.z());
    }
  }
}

bool CudaSlamRosNode::finalize_imu_init() {
  Eigen::Vector3d sum_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d sum_gyro = Eigen::Vector3d::Zero();

  for (const auto& sample : imu_init_buffer_) {
    sum_acc += sample.linear_acc;
    sum_gyro += sample.angular_vel;
  }

  double n = static_cast<double>(imu_init_buffer_.size());
  Eigen::Vector3d mean_acc = sum_acc / n;
  Eigen::Vector3d mean_gyro = sum_gyro / n;

  init_gyro_bias_ = mean_gyro;

  double acc_norm = mean_acc.norm();
  if (acc_norm < 1.0 || acc_norm > 20.0) {
    spdlog::warn("[cuda_slam_ros] Unexpected accelerometer norm: {:.3f}. "
                 "Skipping IMU init, resetting buffer.", acc_norm);
    imu_init_buffer_.clear();
    return false;
  }

  init_gravity_ = mean_acc;
  double g = odometry_params_.imu_gravity;

  init_accel_bias_ = mean_acc - (mean_acc.normalized() * g);

  spdlog::info("[cuda_slam_ros] Mean acc norm: {:.3f} (expected ~{:.3f})",
               acc_norm, g);

  imu_init_buffer_.clear();
  return true;
}

bool CudaSlamRosNode::load_global_map() {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;

  if (!load_pcd_file(map_file_, points)) {
    spdlog::error("[cuda_slam_ros] Failed to load PCD map: {}", map_file_);
    return false;
  }

  if (points.empty()) {
    spdlog::error("[cuda_slam_ros] Empty PCD map");
    return false;
  }

  bool ok = global_localization_->load_map(points);
  if (!ok) {
    spdlog::error("[cuda_slam_ros] Failed to build GPU voxel grids for map");
    return false;
  }

  return true;
}

bool CudaSlamRosNode::run_global_localization(
    const std::vector<Eigen::Vector3d,
                       Eigen::aligned_allocator<Eigen::Vector3d>>& scan_points,
    const Eigen::Isometry3d& initial_pose,
    Eigen::Isometry3d& global_pose) {

  int n = static_cast<int>(scan_points.size());
  if (n == 0) return false;

  ensure_gpu_scan_buffer(n);

  std::vector<float> host_points(n * 4);
  for (int i = 0; i < n; i++) {
    host_points[i * 4 + 0] = static_cast<float>(scan_points[i].x());
    host_points[i * 4 + 1] = static_cast<float>(scan_points[i].y());
    host_points[i * 4 + 2] = static_cast<float>(scan_points[i].z());
    host_points[i * 4 + 3] = 1.0f;
  }

  cudaMemcpy(d_scan_points_, host_points.data(), n * sizeof(float4),
             cudaMemcpyHostToDevice);

  float T_guess[12] = {
    static_cast<float>(initial_pose(0, 0)),
    static_cast<float>(initial_pose(0, 1)),
    static_cast<float>(initial_pose(0, 2)),
    static_cast<float>(initial_pose(0, 3)),
    static_cast<float>(initial_pose(1, 0)),
    static_cast<float>(initial_pose(1, 1)),
    static_cast<float>(initial_pose(1, 2)),
    static_cast<float>(initial_pose(1, 3)),
    static_cast<float>(initial_pose(2, 0)),
    static_cast<float>(initial_pose(2, 1)),
    static_cast<float>(initial_pose(2, 2)),
    static_cast<float>(initial_pose(2, 3))
  };

  float T_result[12];
  memcpy(T_result, T_guess, 12 * sizeof(float));

  float final_error = 0.0f;
  bool ok = global_localization_->localize_with_guess(
      d_scan_points_, n, T_guess, T_result, &final_error);

  if (ok) {
    global_pose.setIdentity();
    global_pose(0, 0) = T_result[0];
    global_pose(0, 1) = T_result[1];
    global_pose(0, 2) = T_result[2];
    global_pose(0, 3) = T_result[3];
    global_pose(1, 0) = T_result[4];
    global_pose(1, 1) = T_result[5];
    global_pose(1, 2) = T_result[6];
    global_pose(1, 3) = T_result[7];
    global_pose(2, 0) = T_result[8];
    global_pose(2, 1) = T_result[9];
    global_pose(2, 2) = T_result[10];
    global_pose(2, 3) = T_result[11];

    spdlog::debug("[cuda_slam_ros] Global localization OK, error={:.4f}",
                  final_error);
    return true;
  }

  spdlog::debug("[cuda_slam_ros] Global localization failed, error={:.4f}",
                final_error);
  return false;
}

void CudaSlamRosNode::release_gpu_resources() {
  if (d_scan_points_) {
    cudaFree(d_scan_points_);
    d_scan_points_ = nullptr;
  }
  d_scan_capacity_ = 0;
}

void CudaSlamRosNode::ensure_gpu_scan_buffer(int num_points) {
  if (num_points <= d_scan_capacity_) return;
  release_gpu_resources();
  cudaMalloc(&d_scan_points_, num_points * sizeof(float4));
  d_scan_capacity_ = num_points;
}

cuda_slam::RawPoints CudaSlamRosNode::convert_pointcloud(
    const sensor_msgs::msg::PointCloud2& msg) {

  cuda_slam::RawPoints raw;
  raw.timestamp = stamp_to_sec(msg.header.stamp);

  int num_points = msg.width * msg.height;
  if (num_points == 0) {
    return raw;
  }

  int x_offset = -1, y_offset = -1, z_offset = -1;
  int time_offset = -1, intensity_offset = -1;
  int x_type = 0;

  for (const auto& field : msg.fields) {
    if (field.name == "x") {
      x_type = field.datatype;
      x_offset = field.offset;
    } else if (field.name == "y") {
      y_offset = field.offset;
    } else if (field.name == "z") {
      z_offset = field.offset;
    } else if (field.name == "t" || field.name == "time" ||
               field.name == "time_stamp" || field.name == "timestamp") {
      time_offset = field.offset;
    } else if (field.name == "intensity" || field.name == "intensities") {
      intensity_offset = field.offset;
    }
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    spdlog::warn("[cuda_slam_ros] Missing x/y/z fields in point cloud");
    return raw;
  }

  raw.points.resize(num_points);
  if (time_offset >= 0) {
    raw.times.resize(num_points);
  }
  if (intensity_offset >= 0) {
    raw.intensities.resize(num_points);
  }

  for (int i = 0; i < num_points; i++) {
    const uint8_t* base = msg.data.data() + msg.point_step * i;

    if (x_type == POINT_FIELD_FLOAT32 &&
        y_offset == x_offset + 4 &&
        z_offset == y_offset + 4) {
      const float* xyz = reinterpret_cast<const float*>(base + x_offset);
      raw.points[i] = Eigen::Vector4d(xyz[0], xyz[1], xyz[2], 1.0);
    } else if (x_type == POINT_FIELD_FLOAT64 &&
               y_offset == x_offset + 8 &&
               z_offset == y_offset + 8) {
      const double* xyz = reinterpret_cast<const double*>(base + x_offset);
      raw.points[i] = Eigen::Vector4d(xyz[0], xyz[1], xyz[2], 1.0);
    } else if (x_type == POINT_FIELD_FLOAT32) {
      float x = *reinterpret_cast<const float*>(base + x_offset);
      float y = *reinterpret_cast<const float*>(base + y_offset);
      float z = *reinterpret_cast<const float*>(base + z_offset);
      raw.points[i] = Eigen::Vector4d(x, y, z, 1.0);
    } else {
      double x = *reinterpret_cast<const double*>(base + x_offset);
      double y = *reinterpret_cast<const double*>(base + y_offset);
      double z = *reinterpret_cast<const double*>(base + z_offset);
      raw.points[i] = Eigen::Vector4d(x, y, z, 1.0);
    }

    if (time_offset >= 0) {
      const uint8_t* time_ptr = base + time_offset;
      int time_type = 0;
      for (const auto& field : msg.fields) {
        if (field.offset == time_offset) {
          time_type = field.datatype;
          break;
        }
      }
      switch (time_type) {
        case POINT_FIELD_UINT32:
          raw.times[i] = *reinterpret_cast<const uint32_t*>(time_ptr) / 1e9;
          break;
        case POINT_FIELD_FLOAT32:
          raw.times[i] = *reinterpret_cast<const float*>(time_ptr);
          break;
        case POINT_FIELD_FLOAT64:
          raw.times[i] = *reinterpret_cast<const double*>(time_ptr);
          break;
        default:
          raw.times[i] = 0.0;
          break;
      }
    }

    if (intensity_offset >= 0) {
      const uint8_t* intensity_ptr = base + intensity_offset;
      int intensity_type = 0;
      for (const auto& field : msg.fields) {
        if (field.offset == intensity_offset) {
          intensity_type = field.datatype;
          break;
        }
      }
      switch (intensity_type) {
        case POINT_FIELD_UINT8:
          raw.intensities[i] = *reinterpret_cast<const uint8_t*>(intensity_ptr);
          break;
        case POINT_FIELD_UINT16:
          raw.intensities[i] = *reinterpret_cast<const uint16_t*>(intensity_ptr);
          break;
        case POINT_FIELD_FLOAT32:
          raw.intensities[i] = *reinterpret_cast<const float*>(intensity_ptr);
          break;
        case POINT_FIELD_FLOAT64:
          raw.intensities[i] = *reinterpret_cast<const double*>(intensity_ptr);
          break;
        default:
          raw.intensities[i] = 0.0f;
          break;
      }
    }
  }

  return raw;
}

nav_msgs::msg::Odometry CudaSlamRosNode::build_odometry_msg(
    const cuda_slam::EstimationFrame::ConstPtr& frame) const {

  nav_msgs::msg::Odometry odom;

  const auto& T = frame->T_world_imu;
  Eigen::Quaterniond q(T.linear());
  const auto& p = T.translation();

  odom.pose.pose.position.x = p.x();
  odom.pose.pose.position.y = p.y();
  odom.pose.pose.position.z = p.z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();

  odom.pose.covariance.fill(0.0);
  for (int i = 0; i < 6; i++) {
    odom.pose.covariance[i * 6 + i] = 0.01;
  }

  const auto& v = frame->v_world_imu;
  odom.twist.twist.linear.x = v.x();
  odom.twist.twist.linear.y = v.y();
  odom.twist.twist.linear.z = v.z();

  odom.twist.covariance.fill(0.0);
  for (int i = 0; i < 6; i++) {
    odom.twist.covariance[i * 6 + i] = 0.05;
  }

  return odom;
}

nav_msgs::msg::Odometry CudaSlamRosNode::build_global_odometry_msg(
    const Eigen::Isometry3d& pose, double stamp) const {

  nav_msgs::msg::Odometry odom;

  Eigen::Quaterniond q(pose.linear());
  const auto& p = pose.translation();

  odom.pose.pose.position.x = p.x();
  odom.pose.pose.position.y = p.y();
  odom.pose.pose.position.z = p.z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();

  odom.pose.covariance.fill(0.0);
  for (int i = 0; i < 6; i++) {
    odom.pose.covariance[i * 6 + i] = 0.02;
  }

  odom.twist.covariance.fill(0.0);
  for (int i = 0; i < 6; i++) {
    odom.twist.covariance[i * 6 + i] = 0.1;
  }

  return odom;
}

geometry_msgs::msg::TransformStamped CudaSlamRosNode::build_transform_msg(
    const cuda_slam::EstimationFrame::ConstPtr& frame) const {

  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = odom_frame_id_;
  tf.child_frame_id = base_frame_id_;

  const auto& T = frame->T_world_imu;
  Eigen::Quaterniond q(T.linear());
  const auto& p = T.translation();

  tf.transform.translation.x = p.x();
  tf.transform.translation.y = p.y();
  tf.transform.translation.z = p.z();
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  tf.transform.rotation.w = q.w();

  return tf;
}

}  // namespace cuda_slam_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cuda_slam_ros::CudaSlamRosNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}