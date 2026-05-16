#ifndef CUDA_SLAM_CUDA_ICP_HPP
#define CUDA_SLAM_CUDA_ICP_HPP

#include <cuda_runtime.h>
#include <vector_types.h>

namespace cuda_slam {
namespace cuda {

class CudaVoxelGrid;

class CudaICP {
public:
    CudaICP();
    ~CudaICP();

    CudaICP(const CudaICP&) = delete;
    CudaICP& operator=(const CudaICP&) = delete;

    bool align(const float4* source_points, int source_count,
               const CudaVoxelGrid& target_grid,
               float* transform_3x4,
               int max_iterations,
               float* final_error = nullptr);

    void set_max_correspondence_distance(float d) { max_corr_dist_ = d; }
    void set_convergence_threshold(float t) { convergence_thresh_ = t; }

private:
    bool solve_cholesky_6x6(const float* H, const float* b, float* delta);

    float max_corr_dist_;
    float convergence_thresh_;

    float* d_transformed_;
    float* d_residuals_;
    float* d_jacobians_;
    float* d_H_;
    float* d_b_;
    int* d_count_;
    int allocated_points_;
};

} // namespace cuda
} // namespace cuda_slam

#endif // CUDA_SLAM_CUDA_ICP_HPP