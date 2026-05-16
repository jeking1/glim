#include "cuda_slam/cuda/cuda_voxel_grid.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cmath>

namespace cuda_slam {
namespace cuda {

struct VoxelAccum {
    float sum_x, sum_y, sum_z, sum_w;
    float sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
    int count;

    __host__ __device__ VoxelAccum() : sum_x(0), sum_y(0), sum_z(0), sum_w(0),
        sum_xx(0), sum_xy(0), sum_xz(0), sum_yy(0), sum_yz(0), sum_zz(0), count(0) {}

    __host__ __device__ VoxelAccum(float x, float y, float z, float w)
        : sum_x(x), sum_y(y), sum_z(z), sum_w(w),
          sum_xx(x*x), sum_xy(x*y), sum_xz(x*z),
          sum_yy(y*y), sum_yz(y*z), sum_zz(z*z), count(1) {}
};

__host__ __device__ VoxelAccum operator+(const VoxelAccum& a, const VoxelAccum& b) {
    VoxelAccum r;
    r.sum_x  = a.sum_x  + b.sum_x;
    r.sum_y  = a.sum_y  + b.sum_y;
    r.sum_z  = a.sum_z  + b.sum_z;
    r.sum_w  = a.sum_w  + b.sum_w;
    r.sum_xx = a.sum_xx + b.sum_xx;
    r.sum_xy = a.sum_xy + b.sum_xy;
    r.sum_xz = a.sum_xz + b.sum_xz;
    r.sum_yy = a.sum_yy + b.sum_yy;
    r.sum_yz = a.sum_yz + b.sum_yz;
    r.sum_zz = a.sum_zz + b.sum_zz;
    r.count  = a.count  + b.count;
    return r;
}

__device__ uint64_t encode_voxel_key(float x, float y, float z, float inv_res) {
    int64_t vx = (int64_t)floorf(x * inv_res);
    int64_t vy = (int64_t)floorf(y * inv_res);
    int64_t vz = (int64_t)floorf(z * inv_res);
    uint64_t kx = (uint64_t)(vx & 0x1FFFFF);
    uint64_t ky = (uint64_t)(vy & 0x1FFFFF);
    uint64_t kz = (uint64_t)(vz & 0x1FFFFF);
    return kx | (ky << 21) | (kz << 42);
}

__global__ void init_voxel_accum_kernel(uint64_t* keys, VoxelAccum* accum,
                                         const float4* points, int n, float inv_res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float4 p = points[idx];
    keys[idx] = encode_voxel_key(p.x, p.y, p.z, inv_res);
    accum[idx] = VoxelAccum(p.x, p.y, p.z, p.w);
}

__global__ void finalize_voxel_kernel(float4* means, float* covs,
                                       const uint64_t* keys,
                                       const VoxelAccum* reduced, int num_voxels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels) return;

    const VoxelAccum& a = reduced[idx];
    float inv = 1.0f / a.count;
    float mx = a.sum_x * inv;
    float my = a.sum_y * inv;
    float mz = a.sum_z * inv;
    float mw = a.sum_w * inv;

    means[idx] = make_float4(mx, my, mz, mw);

    int base = idx * 6;
    covs[base + 0] = fmaxf(a.sum_xx * inv - mx * mx, 1e-8f);
    covs[base + 1] = a.sum_xy * inv - mx * my;
    covs[base + 2] = a.sum_xz * inv - mx * mz;
    covs[base + 3] = fmaxf(a.sum_yy * inv - my * my, 1e-8f);
    covs[base + 4] = a.sum_yz * inv - my * mz;
    covs[base + 5] = fmaxf(a.sum_zz * inv - mz * mz, 1e-8f);
}

CudaVoxelGrid::CudaVoxelGrid(float resolution)
    : resolution_(resolution), num_voxels_(0),
      d_voxel_means_(nullptr), d_voxel_covs_(nullptr) {}

CudaVoxelGrid::~CudaVoxelGrid() {
    release();
}

CudaVoxelGrid::CudaVoxelGrid(CudaVoxelGrid&& other) noexcept
    : resolution_(other.resolution_),
      num_voxels_(other.num_voxels_),
      d_voxel_means_(other.d_voxel_means_),
      d_voxel_covs_(other.d_voxel_covs_) {
    other.resolution_ = 0.0f;
    other.num_voxels_ = 0;
    other.d_voxel_means_ = nullptr;
    other.d_voxel_covs_ = nullptr;
}

CudaVoxelGrid& CudaVoxelGrid::operator=(CudaVoxelGrid&& other) noexcept {
    if (this != &other) {
        release();
        resolution_ = other.resolution_;
        num_voxels_ = other.num_voxels_;
        d_voxel_means_ = other.d_voxel_means_;
        d_voxel_covs_ = other.d_voxel_covs_;
        other.resolution_ = 0.0f;
        other.num_voxels_ = 0;
        other.d_voxel_means_ = nullptr;
        other.d_voxel_covs_ = nullptr;
    }
    return *this;
}

void CudaVoxelGrid::release() {
    if (d_voxel_means_) {
        cudaFree(d_voxel_means_);
        d_voxel_means_ = nullptr;
    }
    if (d_voxel_covs_) {
        cudaFree(d_voxel_covs_);
        d_voxel_covs_ = nullptr;
    }
    num_voxels_ = 0;
}

void CudaVoxelGrid::insert(const float4* points, int n) {
    release();

    if (n == 0) return;

    float inv_res = 1.0f / resolution_;

    thrust::device_vector<uint64_t> d_keys(n);
    thrust::device_vector<VoxelAccum> d_accum(n);

    {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        init_voxel_accum_kernel<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_accum.data()),
            points, n, inv_res);
    }
    cudaDeviceSynchronize();

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_accum.begin());

    thrust::device_vector<uint64_t> d_unique_keys(n);
    thrust::device_vector<VoxelAccum> d_reduced(n);

    auto end_pair = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(), d_accum.begin(),
        d_unique_keys.begin(), d_reduced.begin());

    num_voxels_ = static_cast<int>(thrust::distance(d_unique_keys.begin(), end_pair.first));

    cudaMalloc(&d_voxel_means_, num_voxels_ * sizeof(float4));
    cudaMalloc(&d_voxel_covs_, num_voxels_ * 6 * sizeof(float));

    {
        int block_size = 256;
        int grid_size = (num_voxels_ + block_size - 1) / block_size;
        finalize_voxel_kernel<<<grid_size, block_size>>>(
            d_voxel_means_, d_voxel_covs_,
            thrust::raw_pointer_cast(d_unique_keys.data()),
            thrust::raw_pointer_cast(d_reduced.data()), num_voxels_);
    }
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace cuda_slam