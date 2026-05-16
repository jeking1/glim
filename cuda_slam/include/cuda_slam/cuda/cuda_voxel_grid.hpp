#ifndef CUDA_SLAM_CUDA_VOXEL_GRID_HPP
#define CUDA_SLAM_CUDA_VOXEL_GRID_HPP

#include <cuda_runtime.h>
#include <vector_types.h>

namespace cuda_slam {
namespace cuda {

class CudaVoxelGrid {
public:
    explicit CudaVoxelGrid(float resolution);
    ~CudaVoxelGrid();

    CudaVoxelGrid(const CudaVoxelGrid&) = delete;
    CudaVoxelGrid& operator=(const CudaVoxelGrid&) = delete;
    CudaVoxelGrid(CudaVoxelGrid&& other) noexcept;
    CudaVoxelGrid& operator=(CudaVoxelGrid&& other) noexcept;

    void insert(const float4* points, int n);

    const float4* get_voxel_means() const { return d_voxel_means_; }
    const float* get_voxel_covs() const { return d_voxel_covs_; }
    int num_voxels() const { return num_voxels_; }
    float resolution() const { return resolution_; }

private:
    void release();

    float resolution_;
    int num_voxels_;
    float4* d_voxel_means_;
    float* d_voxel_covs_;
};

} // namespace cuda
} // namespace cuda_slam

#endif // CUDA_SLAM_CUDA_VOXEL_GRID_HPP