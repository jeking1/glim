cmake_minimum_required(VERSION 3.16)
project(cuda_slam_ros VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

find_package(cuda_slam REQUIRED
  PATHS ${CMAKE_SOURCE_DIR}/../cuda_slam
  NO_DEFAULT_PATH
)

add_executable(cuda_slam_ros_node
  src/cuda_slam_ros_node.cpp
)

target_include_directories(cuda_slam_ros_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(cuda_slam_ros_node
  cuda_slam
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  tf2_ros::tf2_ros
  tf2_eigen::tf2_eigen
  spdlog::spdlog
)

target_compile_definitions(cuda_slam_ros_node PRIVATE EIGEN_NO_AUTOMATIC_RESIZING)

install(TARGETS cuda_slam_ros_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()