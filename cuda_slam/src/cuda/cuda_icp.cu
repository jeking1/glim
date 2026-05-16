#include "cuda_slam/cuda/cuda_icp.hpp"
#include "cuda_slam/cuda/cuda_kernels.h"
#include "cuda_slam/cuda/cuda_voxel_grid.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstring>

namespace cuda_slam {
namespace cuda {

__global__ void invert_covariance_kernel(float* covs_inv, const float* covs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* C = covs + idx * 6;
    float* Cinv = covs_inv + idx * 6;

    float c00 = C[0], c01 = C[1], c02 = C[2];
    float c11 = C[3], c12 = C[4];
    float c22 = C[5];

    float det = c00 * (c11 * c22 - c12 * c12)
              - c01 * (c01 * c22 - c12 * c02)
              + c02 * (c01 * c12 - c11 * c02);

    if (fabsf(det) < 1e-12f) {
        Cinv[0] = 1.0f; Cinv[1] = 0.0f; Cinv[2] = 0.0f;
        Cinv[3] = 1.0f; Cinv[4] = 0.0f;
        Cinv[5] = 1.0f;
        return;
    }

    float inv_det = 1.0f / det;
    Cinv[0] = (c11 * c22 - c12 * c12) * inv_det;
    Cinv[1] = -(c01 * c22 - c12 * c02) * inv_det;
    Cinv[2] = (c01 * c12 - c11 * c02) * inv_det;
    Cinv[3] = (c00 * c22 - c02 * c02) * inv_det;
    Cinv[4] = -(c00 * c12 - c02 * c01) * inv_det;
    Cinv[5] = (c00 * c11 - c01 * c01) * inv_det;
}

__global__ void reduce_to_hessian_kernel(float* H_global, float* b_global,
                                          const float* residuals,
                                          const float* jacobians,
                                          const float* target_covs_inv,
                                          int n) {
    extern __shared__ float shared_mem[];
    float* H_sh = shared_mem;
    float* b_sh = shared_mem + 21;

    int tid = threadIdx.x;
    for (int i = tid; i < 21; i += blockDim.x) H_sh[i] = 0.0f;
    for (int i = tid; i < 6; i += blockDim.x) b_sh[i] = 0.0f;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* r = residuals + idx * 3;
    const float* J = jacobians + idx * 18;
    const float* Cinv = target_covs_inv + idx * 6;

    float w00 = Cinv[0], w01 = Cinv[1], w02 = Cinv[2];
    float w11 = Cinv[3], w12 = Cinv[4];
    float w22 = Cinv[5];

    float wr0 = w00 * r[0] + w01 * r[1] + w02 * r[2];
    float wr1 = w01 * r[0] + w11 * r[1] + w12 * r[2];
    float wr2 = w02 * r[0] + w12 * r[1] + w22 * r[2];

    float WJ_col[6][3];
    for (int c = 0; c < 6; c++) {
        float j0 = J[c];
        float j1 = J[6 + c];
        float j2 = J[12 + c];
        WJ_col[c][0] = w00 * j0 + w01 * j1 + w02 * j2;
        WJ_col[c][1] = w01 * j0 + w11 * j1 + w12 * j2;
        WJ_col[c][2] = w02 * j0 + w12 * j1 + w22 * j2;
    }

    float local_H[21];
    int k = 0;
    for (int row = 0; row < 6; row++) {
        float jr0 = J[row];
        float jr1 = J[6 + row];
        float jr2 = J[12 + row];
        for (int col = row; col < 6; col++) {
            local_H[k] = jr0 * WJ_col[col][0] + jr1 * WJ_col[col][1] + jr2 * WJ_col[col][2];
            k++;
        }
    }

    float local_b[6];
    for (int row = 0; row < 6; row++) {
        float jr0 = J[row];
        float jr1 = J[6 + row];
        float jr2 = J[12 + row];
        local_b[row] = jr0 * wr0 + jr1 * wr1 + jr2 * wr2;
    }

    for (int i = 0; i < 21; i++) atomicAdd(&H_sh[i], local_H[i]);
    for (int i = 0; i < 6; i++) atomicAdd(&b_sh[i], local_b[i]);

    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < 21; i++) atomicAdd(&H_global[i], H_sh[i]);
        for (int i = 0; i < 6; i++) atomicAdd(&b_global[i], b_sh[i]);
    }
}

__device__ void rodrigues_exp(const float3& omega, float* R_3x3) {
    float theta = sqrtf(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z);
    if (theta < 1e-12f) {
        R_3x3[0] = 1.0f;       R_3x3[1] = -omega.z;  R_3x3[2] = omega.y;
        R_3x3[3] = omega.z;    R_3x3[4] = 1.0f;       R_3x3[5] = -omega.x;
        R_3x3[6] = -omega.y;   R_3x3[7] = omega.x;    R_3x3[8] = 1.0f;
        return;
    }

    float inv_theta = 1.0f / theta;
    float c = cosf(theta);
    float s = sinf(theta);
    float a = s * inv_theta;
    float b = (1.0f - c) * inv_theta * inv_theta;

    float wx = omega.x, wy = omega.y, wz = omega.z;

    R_3x3[0] = 1.0f + b * (-wz * wz - wy * wy);
    R_3x3[1] = -a * wz + b * wx * wy;
    R_3x3[2] = a * wy + b * wx * wz;
    R_3x3[3] = a * wz + b * wx * wy;
    R_3x3[4] = 1.0f + b * (-wz * wz - wx * wx);
    R_3x3[5] = -a * wx + b * wy * wz;
    R_3x3[6] = -a * wy + b * wx * wz;
    R_3x3[7] = a * wx + b * wy * wz;
    R_3x3[8] = 1.0f + b * (-wy * wy - wx * wx);
}

CudaICP::CudaICP()
    : max_corr_dist_(1.0f), convergence_thresh_(1e-5f),
      d_transformed_(nullptr), d_residuals_(nullptr), d_jacobians_(nullptr),
      d_H_(nullptr), d_b_(nullptr), d_count_(nullptr), allocated_points_(0) {
    cudaMalloc(&d_H_, 21 * sizeof(float));
    cudaMalloc(&d_b_, 6 * sizeof(float));
    cudaMalloc(&d_count_, sizeof(int));
}

CudaICP::~CudaICP() {
    if (d_transformed_) cudaFree(d_transformed_);
    if (d_residuals_) cudaFree(d_residuals_);
    if (d_jacobians_) cudaFree(d_jacobians_);
    if (d_H_) cudaFree(d_H_);
    if (d_b_) cudaFree(d_b_);
    if (d_count_) cudaFree(d_count_);
}

bool CudaICP::align(const float4* source_points, int source_count,
                    const CudaVoxelGrid& target_grid,
                    float* transform_3x4,
                    int max_iterations,
                    float* final_error) {
    int target_count = target_grid.num_voxels();
    if (source_count == 0 || target_count == 0) return false;

    if (source_count > allocated_points_) {
        if (d_transformed_) cudaFree(d_transformed_);
        if (d_residuals_) cudaFree(d_residuals_);
        if (d_jacobians_) cudaFree(d_jacobians_);
        allocated_points_ = source_count;

        cudaMalloc(&d_transformed_, source_count * sizeof(float4));
        cudaMalloc(&d_residuals_, source_count * 3 * sizeof(float));
        cudaMalloc(&d_jacobians_, source_count * 18 * sizeof(float));
    }

    int n_pairs = min(source_count, target_count);

    const float4* target_means = target_grid.get_voxel_means();
    const float* target_covs = target_grid.get_voxel_covs();

    thrust::device_vector<float> d_covs_inv(target_count * 6);

    {
        int block_size = 256;
        int grid_size = (target_count + block_size - 1) / block_size;
        invert_covariance_kernel<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_covs_inv.data()), target_covs, target_count);
    }
    cudaDeviceSynchronize();

    float prev_error = 1e30f;
    float current_T[12];
    memcpy(current_T, transform_3x4, 12 * sizeof(float));

    for (int iter = 0; iter < max_iterations; iter++) {
        transform_points(d_transformed_, source_points, source_count, current_T);
        cudaDeviceSynchronize();

        cudaMemset(d_H_, 0, 21 * sizeof(float));
        cudaMemset(d_b_, 0, 6 * sizeof(float));

        gicp_residual(d_residuals_, d_jacobians_, target_means,
                      thrust::raw_pointer_cast(d_covs_inv.data()),
                      d_transformed_, n_pairs);
        cudaDeviceSynchronize();

        int block_size = 256;
        int grid_size = (n_pairs + block_size - 1) / block_size;
        size_t shared_bytes = 27 * sizeof(float);
        reduce_to_hessian_kernel<<<grid_size, block_size, shared_bytes>>>(
            d_H_, d_b_, d_residuals_, d_jacobians_,
            thrust::raw_pointer_cast(d_covs_inv.data()), n_pairs);
        cudaDeviceSynchronize();

        float H_host[21], b_host[6];
        cudaMemcpy(H_host, d_H_, 21 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(b_host, d_b_, 6 * sizeof(float), cudaMemcpyDeviceToHost);

        float delta[6];
        if (!solve_cholesky_6x6(H_host, b_host, delta)) {
            break;
        }

        float delta_v[3] = {delta[0], delta[1], delta[2]};
        float delta_w[3] = {delta[3], delta[4], delta[5]};

        float trans_delta = sqrtf(delta_v[0] * delta_v[0] + delta_v[1] * delta_v[1] + delta_v[2] * delta_v[2]);
        float rot_delta = sqrtf(delta_w[0] * delta_w[0] + delta_w[1] * delta_w[1] + delta_w[2] * delta_w[2]);

        float dR[9];
        float3 dw = make_float3(delta_w[0], delta_w[1], delta_w[2]);
        rodrigues_exp(dw, dR);

        float R_new[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_new[i * 3 + j] = dR[i * 3 + 0] * current_T[4 * 0 + j]
                                 + dR[i * 3 + 1] * current_T[4 * 1 + j]
                                 + dR[i * 3 + 2] * current_T[4 * 2 + j];
            }
        }

        current_T[0] = R_new[0]; current_T[1] = R_new[1]; current_T[2]  = R_new[2];
        current_T[4] = R_new[3]; current_T[5] = R_new[4]; current_T[6]  = R_new[5];
        current_T[8] = R_new[6]; current_T[9] = R_new[7]; current_T[10] = R_new[8];

        current_T[3]  += delta_v[0];
        current_T[7]  += delta_v[1];
        current_T[11] += delta_v[2];

        float b_norm = 0.0f;
        for (int i = 0; i < 6; i++) b_norm += b_host[i] * b_host[i];
        float error = sqrtf(b_norm / n_pairs);

        if (final_error) *final_error = error;

        if (trans_delta < convergence_thresh_ && rot_delta < convergence_thresh_) {
            break;
        }

        if (error > prev_error * 0.999f && iter > 2) {
            break;
        }
        prev_error = error;
    }

    memcpy(transform_3x4, current_T, 12 * sizeof(float));
    return true;
}

bool CudaICP::solve_cholesky_6x6(const float* H, const float* b, float* delta) {
    float L[36];
    memset(L, 0, 36 * sizeof(float));

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            int h_idx = j * 6 - j * (j - 1) / 2 + (i - j);
            float sum = H[h_idx];
            for (int k = 0; k < j; k++) {
                sum -= L[i * 6 + k] * L[j * 6 + k];
            }
            if (i == j) {
                if (sum <= 0.0f) return false;
                L[i * 6 + i] = sqrtf(sum);
            } else {
                L[i * 6 + j] = sum / L[j * 6 + j];
            }
        }
    }

    float y[6];
    for (int i = 0; i < 6; i++) {
        float sum = -b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * 6 + j] * y[j];
        }
        y[i] = sum / L[i * 6 + i];
    }

    for (int i = 5; i >= 0; i--) {
        float sum = y[i];
        for (int j = i + 1; j < 6; j++) {
            sum -= L[j * 6 + i] * delta[j];
        }
        delta[i] = sum / L[i * 6 + i];
    }

    return true;
}

} // namespace cuda
} // namespace cuda_slam