cmake_minimum_required(VERSION 3.18)
project(cuda_slam VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_WITH_CUDA "Build with CUDA GPU acceleration" ON)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP)

if(BUILD_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
  set(CUDA_SEPARABLE_COMPILATION ON)
  add_definitions(-DBUILD_WITH_CUDA)
endif()

set(CUDA_SLAM_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
)

file(GLOB_RECURSE CUDA_SLAM_CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(cuda_slam ${CUDA_SLAM_CPP_SOURCES})

if(BUILD_WITH_CUDA)
  file(GLOB_RECURSE CUDA_SLAM_CU_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
  )
  target_sources(cuda_slam PRIVATE ${CUDA_SLAM_CU_SOURCES})
  target_include_directories(cuda_slam PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(cuda_slam PUBLIC
      CUDA::cudart
      CUDA::cublas
      CUDA::cusolver
  )
  target_compile_definitions(cuda_slam PRIVATE CUDA_SLAM_USE_CUDA)
  set_target_properties(cuda_slam PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

target_include_directories(cuda_slam PUBLIC ${CUDA_SLAM_INCLUDE_DIR})

target_link_libraries(cuda_slam PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

if(OpenMP_FOUND)
    target_link_libraries(cuda_slam PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(cuda_slam PRIVATE _OPENMP)
endif()

target_compile_options(cuda_slam PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

target_compile_definitions(cuda_slam PRIVATE EIGEN_NO_AUTOMATIC_RESIZING)

install(TARGETS cuda_slam
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY config/
    DESTINATION share/cuda_slam/config
)