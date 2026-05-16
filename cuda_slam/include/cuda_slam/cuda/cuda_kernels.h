#ifndef CUDA_SLAM_CUDA_KERNELS_H
#define CUDA_SLAM_CUDA_KERNELS_H

namespace cuda_slam {
namespace cuda {

void transform_points(float4* output, const float4* input, int n, const float* transform_3x4);

void compute_normals(float4* normals, const float4* points, const int* neighbors, const int* offsets, int n, int max_neighbors);

void compute_covariance(float* covs_6, const float4* points, const int* neighbors, const int* offsets, int n, int max_neighbors);

void voxel_downsample(float4* output, int* output_count, const float4* input, int n, float resolution);

void gicp_residual(float* residuals, float* jacobians, const float4* target_means, const float* target_covs_inv, const float4* source_points, int n);

void overlap_count(int* count, const float4* target_means, const float4* source_points, int n, const float* transform, float resolution);

} // namespace cuda
} // namespace cuda_slam

#endif // CUDA_SLAM_CUDA_KERNELS_H