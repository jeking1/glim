#include "cuda_slam/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cmath>

namespace cuda_slam {
namespace cuda {

__device__ float3 make_float3_from_float4(const float4& v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ float4 make_float4_from_float3(const float3& v, float w) {
    return make_float4(v.x, v.y, v.z, w);
}

__device__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ float3 normalize3(const float3& v) {
    float len = sqrtf(dot3(v, v));
    if (len > 1e-10f) {
        float inv = 1.0f / len;
        return make_float3(v.x * inv, v.y * inv, v.z * inv);
    }
    return make_float3(0.0f, 0.0f, 1.0f);
}

__device__ void eigen_smallest_3x3(const float* A, float3* eigenvec) {
    float m00 = A[0], m01 = A[1], m02 = A[2];
    float m11 = A[3], m12 = A[4];
    float m22 = A[5];

    float p1 = m01 * m01 + m02 * m02 + m12 * m12;
    if (p1 < 1e-20f) {
        float min_val = m00;
        int min_idx = 0;
        if (m11 < min_val) { min_val = m11; min_idx = 1; }
        if (m22 < min_val) { min_val = m22; min_idx = 2; }

        if (min_idx == 0)      *eigenvec = make_float3(1.0f, 0.0f, 0.0f);
        else if (min_idx == 1) *eigenvec = make_float3(0.0f, 1.0f, 0.0f);
        else                   *eigenvec = make_float3(0.0f, 0.0f, 1.0f);
        return;
    }

    float trace = m00 + m11 + m22;
    float q = m00 * m11 + m00 * m22 + m11 * m22 - p1;
    float r = m00 * m11 * m22 + 2.0f * m01 * m12 * m02
              - m00 * m12 * m12 - m11 * m02 * m02 - m22 * m01 * m01;

    float p_over_3 = trace / 3.0f;
    float pp = q - p_over_3 * p_over_3;
    float qq = r - p_over_3 * q + 2.0f * p_over_3 * p_over_3 * p_over_3;

    float half_qq = 0.5f * qq;
    float rho_sq = (-pp * pp * pp) / 27.0f;

    float phi;
    float rho = sqrtf(fmaxf(0.0f, rho_sq));
    if (rho < 1e-14f) {
        phi = 0.0f;
    } else {
        float arg = half_qq / rho;
        arg = fmaxf(-1.0f, fminf(1.0f, arg));
        phi = acosf(arg) / 3.0f;
    }

    float sqrt_pp3 = 2.0f * sqrtf(fmaxf(0.0f, -pp / 3.0f));

    float lam0 = sqrt_pp3 * cosf(phi) + p_over_3;
    float lam1 = sqrt_pp3 * cosf(phi + 2.0943951023931953f) + p_over_3;
    float lam2 = sqrt_pp3 * cosf(phi + 4.1887902047863905f) + p_over_3;

    float lam_min = fminf(fminf(lam0, lam1), lam2);

    float B00 = m00 - lam_min;
    float B01 = m01;
    float B02 = m02;
    float B11 = m11 - lam_min;
    float B12 = m12;
    float B22 = m22 - lam_min;

    float3 v0 = make_float3(B00, B01, B02);
    float3 v1 = make_float3(B01, B11, B12);
    float3 v2 = make_float3(B02, B12, B22);

    float3 c01 = cross3(v0, v1);
    float3 c02 = cross3(v0, v2);
    float3 c12 = cross3(v1, v2);

    float n01 = dot3(c01, c01);
    float n02 = dot3(c02, c02);
    float n12 = dot3(c12, c12);

    float3 best;
    if (n01 >= n02 && n01 >= n12) {
        best = c01;
    } else if (n02 >= n01 && n02 >= n12) {
        best = c02;
    } else {
        best = c12;
    }

    *eigenvec = normalize3(best);
}

__global__ void transform_points_kernel(float4* output, const float4* input, int n, const float* T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float4 p = input[idx];
    float r00 = T[0], r01 = T[1], r02 = T[2],  tx = T[3];
    float r10 = T[4], r11 = T[5], r12 = T[6],  ty = T[7];
    float r20 = T[8], r21 = T[9], r22 = T[10], tz = T[11];

    float x = r00 * p.x + r01 * p.y + r02 * p.z + tx;
    float y = r10 * p.x + r11 * p.y + r12 * p.z + ty;
    float z = r20 * p.x + r21 * p.y + r22 * p.z + tz;
    output[idx] = make_float4(x, y, z, p.w);
}

__global__ void compute_normals_kernel(float4* normals, const float4* points,
                                        const int* neighbors, const int* offsets,
                                        int n, int max_neighbors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int start = offsets[idx];
    int end = offsets[idx + 1];
    int num_nbrs = end - start;

    if (num_nbrs < 3) {
        normals[idx] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
        return;
    }

    int actual = min(num_nbrs, max_neighbors);
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < actual; j++) {
        int nbr_idx = neighbors[start + j];
        float4 nbr = points[nbr_idx];
        centroid.x += nbr.x;
        centroid.y += nbr.y;
        centroid.z += nbr.z;
    }
    centroid.x /= actual;
    centroid.y /= actual;
    centroid.z /= actual;

    float cov[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < actual; j++) {
        int nbr_idx = neighbors[start + j];
        float4 nbr = points[nbr_idx];
        float dx = nbr.x - centroid.x;
        float dy = nbr.y - centroid.y;
        float dz = nbr.z - centroid.z;
        cov[0] += dx * dx;
        cov[1] += dx * dy;
        cov[2] += dx * dz;
        cov[3] += dy * dy;
        cov[4] += dy * dz;
        cov[5] += dz * dz;
    }

    float inv_n = 1.0f / actual;
    for (int k = 0; k < 6; k++) cov[k] *= inv_n;

    float3 normal;
    eigen_smallest_3x3(cov, &normal);
    normals[idx] = make_float4(normal.x, normal.y, normal.z, 0.0f);
}

__global__ void compute_covariance_kernel(float* covs_6, const float4* points,
                                           const int* neighbors, const int* offsets,
                                           int n, int max_neighbors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int start = offsets[idx];
    int end = offsets[idx + 1];
    int num_nbrs = end - start;

    if (num_nbrs < 2) {
        for (int k = 0; k < 6; k++) covs_6[idx * 6 + k] = 0.0f;
        return;
    }

    int actual = min(num_nbrs, max_neighbors);
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < actual; j++) {
        int nbr_idx = neighbors[start + j];
        float4 nbr = points[nbr_idx];
        centroid.x += nbr.x;
        centroid.y += nbr.y;
        centroid.z += nbr.z;
    }
    centroid.x /= actual;
    centroid.y /= actual;
    centroid.z /= actual;

    float cov[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < actual; j++) {
        int nbr_idx = neighbors[start + j];
        float4 nbr = points[nbr_idx];
        float dx = nbr.x - centroid.x;
        float dy = nbr.y - centroid.y;
        float dz = nbr.z - centroid.z;
        cov[0] += dx * dx;
        cov[1] += dx * dy;
        cov[2] += dx * dz;
        cov[3] += dy * dy;
        cov[4] += dy * dz;
        cov[5] += dz * dz;
    }

    float inv = 1.0f / (actual - 1);
    int base = idx * 6;
    covs_6[base + 0] = cov[0] * inv;
    covs_6[base + 1] = cov[1] * inv;
    covs_6[base + 2] = cov[2] * inv;
    covs_6[base + 3] = cov[3] * inv;
    covs_6[base + 4] = cov[4] * inv;
    covs_6[base + 5] = cov[5] * inv;
}

__device__ uint64_t voxel_key(float x, float y, float z, float inv_res) {
    int64_t vx = (int64_t)floorf(x * inv_res);
    int64_t vy = (int64_t)floorf(y * inv_res);
    int64_t vz = (int64_t)floorf(z * inv_res);
    uint64_t key = (uint64_t)(vx & 0x1FFFFF)
                 | ((uint64_t)(vy & 0x1FFFFF) << 21)
                 | ((uint64_t)(vz & 0x1FFFFF) << 42);
    return key;
}

struct VoxelDownsampleStats {
    float sum_x, sum_y, sum_z, sum_w;
    int count;
};

__host__ __device__ VoxelDownsampleStats operator+(const VoxelDownsampleStats& a,
                                                    const VoxelDownsampleStats& b) {
    VoxelDownsampleStats r;
    r.sum_x = a.sum_x + b.sum_x;
    r.sum_y = a.sum_y + b.sum_y;
    r.sum_z = a.sum_z + b.sum_z;
    r.sum_w = a.sum_w + b.sum_w;
    r.count = a.count + b.count;
    return r;
}

__global__ void init_voxel_keys_kernel(uint64_t* keys, VoxelDownsampleStats* stats,
                                        const float4* input, int n, float inv_res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = input[idx];
    keys[idx] = voxel_key(p.x, p.y, p.z, inv_res);
    stats[idx].sum_x = p.x;
    stats[idx].sum_y = p.y;
    stats[idx].sum_z = p.z;
    stats[idx].sum_w = p.w;
    stats[idx].count = 1;
}

__global__ void normalize_voxel_kernel(float4* output, const VoxelDownsampleStats* reduced_stats,
                                        int num_voxels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;
    const VoxelDownsampleStats& s = reduced_stats[idx];
    float inv = 1.0f / s.count;
    output[idx] = make_float4(s.sum_x * inv, s.sum_y * inv, s.sum_z * inv, s.sum_w * inv);
}

__global__ void gicp_residual_kernel(float* residuals, float* jacobians,
                                      const float4* target_means,
                                      const float* target_covs_inv,
                                      const float4* source_points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float4 src = source_points[idx];
    float4 tgt = target_means[idx];

    float rx = src.x - tgt.x;
    float ry = src.y - tgt.y;
    float rz = src.z - tgt.z;

    int r_base = idx * 3;
    residuals[r_base + 0] = rx;
    residuals[r_base + 1] = ry;
    residuals[r_base + 2] = rz;

    int j_base = idx * 18;
    jacobians[j_base + 0] = 1.0f;
    jacobians[j_base + 1] = 0.0f;
    jacobians[j_base + 2] = 0.0f;
    jacobians[j_base + 3] = 0.0f;
    jacobians[j_base + 4] = src.z;
    jacobians[j_base + 5] = -src.y;

    jacobians[j_base + 6] = 0.0f;
    jacobians[j_base + 7] = 1.0f;
    jacobians[j_base + 8] = 0.0f;
    jacobians[j_base + 9] = -src.z;
    jacobians[j_base + 10] = 0.0f;
    jacobians[j_base + 11] = src.x;

    jacobians[j_base + 12] = 0.0f;
    jacobians[j_base + 13] = 0.0f;
    jacobians[j_base + 14] = 1.0f;
    jacobians[j_base + 15] = src.y;
    jacobians[j_base + 16] = -src.x;
    jacobians[j_base + 17] = 0.0f;
}

__global__ void overlap_count_kernel(int* count, const float4* target_means,
                                      const float4* source_points, int n,
                                      const float* transform, float resolution) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float4 src = source_points[idx];
    float r00 = transform[0], r01 = transform[1], r02 = transform[2],  tx = transform[3];
    float r10 = transform[4], r11 = transform[5], r12 = transform[6],  ty = transform[7];
    float r20 = transform[8], r21 = transform[9], r22 = transform[10], tz = transform[11];

    float tx_src = r00 * src.x + r01 * src.y + r02 * src.z + tx;
    float ty_src = r10 * src.x + r11 * src.y + r12 * src.z + ty;
    float tz_src = r20 * src.x + r21 * src.y + r22 * src.z + tz;

    float half_res = resolution * 0.5f;
    int local_count = 0;
    for (int j = 0; j < n; j++) {
        float4 tgt = target_means[j];
        float dx = tx_src - tgt.x;
        float dy = ty_src - tgt.y;
        float dz = tz_src - tgt.z;
        if (fabsf(dx) < half_res && fabsf(dy) < half_res && fabsf(dz) < half_res) {
            local_count++;
            break;
        }
    }

    if (local_count > 0) {
        atomicAdd(count, 1);
    }
}

void transform_points(float4* output, const float4* input, int n, const float* transform_3x4) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    transform_points_kernel<<<grid_size, block_size>>>(output, input, n, transform_3x4);
}

void compute_normals(float4* normals, const float4* points, const int* neighbors,
                     const int* offsets, int n, int max_neighbors) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    compute_normals_kernel<<<grid_size, block_size>>>(normals, points, neighbors, offsets, n, max_neighbors);
}

void compute_covariance(float* covs_6, const float4* points, const int* neighbors,
                        const int* offsets, int n, int max_neighbors) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    compute_covariance_kernel<<<grid_size, block_size>>>(covs_6, points, neighbors, offsets, n, max_neighbors);
}

void voxel_downsample(float4* output, int* output_count, const float4* input, int n, float resolution) {
    if (n == 0) {
        cudaMemset(output_count, 0, sizeof(int));
        return;
    }

    float inv_res = 1.0f / resolution;

    thrust::device_vector<uint64_t> d_keys(n);
    thrust::device_vector<VoxelDownsampleStats> d_stats(n);

    {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        init_voxel_keys_kernel<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_stats.data()),
            input, n, inv_res);
    }
    cudaDeviceSynchronize();

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_stats.begin());

    thrust::device_vector<uint64_t> d_unique_keys(n);
    thrust::device_vector<VoxelDownsampleStats> d_reduced(n);

    auto end_pair = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(), d_stats.begin(),
        d_unique_keys.begin(), d_reduced.begin());

    int num_voxels = static_cast<int>(thrust::distance(d_unique_keys.begin(), end_pair.first));

    {
        int block_size = 256;
        int grid_size = (num_voxels + block_size - 1) / block_size;
        normalize_voxel_kernel<<<grid_size, block_size>>>(
            output, thrust::raw_pointer_cast(d_reduced.data()), num_voxels);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output_count, &num_voxels, sizeof(int), cudaMemcpyHostToDevice);
}

void gicp_residual(float* residuals, float* jacobians, const float4* target_means,
                   const float* target_covs_inv, const float4* source_points, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    gicp_residual_kernel<<<grid_size, block_size>>>(
        residuals, jacobians, target_means, target_covs_inv, source_points, n);
}

void overlap_count(int* count, const float4* target_means, const float4* source_points,
                   int n, const float* transform, float resolution) {
    cudaMemset(count, 0, sizeof(int));
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    overlap_count_kernel<<<grid_size, block_size>>>(
        count, target_means, source_points, n, transform, resolution);
}

} // namespace cuda
} // namespace cuda_slam