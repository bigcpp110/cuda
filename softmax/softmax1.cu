#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); }

const int MAX_BLOCK = 1024;
const int WARMUP_ITER = 10;    // 预热
const int BENCH_ITER = 50;     // 测速轮数

__global__ void softmax_cuda_online(
    float* out, const float* in, int B, int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    if (row >= B) return;

    const float* row_in = in + row * N;
    float* row_out = out + row * N;

    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + block_size;

    float loc_max = -1e20f;
    float loc_sum = 0.0f;

    for (int i = tid; i < N; i += block_size) {
        float x = row_in[i];
        if (x > loc_max) {
            loc_sum = loc_sum * expf(loc_max - x) + 1.0f;
            loc_max = x;
        } else {
            loc_sum += expf(x - loc_max);
        }
    }

    s_max[tid] = loc_max;
    s_sum[tid] = loc_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float m0 = s_max[tid];
            float m1 = s_max[tid + s];
            float s0 = s_sum[tid];
            float s1 = s_sum[tid + s];

            if (m1 > m0) {
                s0 = s0 * expf(m0 - m1);
                m0 = m1;
            } else {
                s1 = s1 * expf(m1 - m0);
            }
            s_max[tid] = m0;
            s_sum[tid] = s0 + s1;
        }
        __syncthreads();
    }

    float row_max = s_max[0];
    float row_sum = s_sum[0];

    for (int i = tid; i < N; i += block_size) {
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
    }
}

__global__ void softmax_online_shuffle_final(
    float* out, const float* in, int B, int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= B) return;

    const float* src = in + row * N;
    float* dst = out + row * N;

    // 1. 单遍 Online 计算局部 max & sum
    float loc_max = -1e20f;
    float loc_sum = 0.0f;

    for (int i = tid; i < N; i += block_size) {
        float x = src[i];
        if (x > loc_max) {
            loc_sum = loc_sum * expf(loc_max - x) + 1.0f;
            loc_max = x;
        } else {
            loc_sum += expf(x - loc_max);
        }
    }

    // 2. 🔥 Warp Shuffle 合并（无共享内存！无 syncthreads！）
    for (int delta = 16; delta > 0; delta >>= 1) {
        float n_max = __shfl_down_sync(0xffffffff, loc_max, delta);
        float n_sum = __shfl_down_sync(0xffffffff, loc_sum, delta);

        if (n_max > loc_max) {
            loc_sum = loc_sum * expf(loc_max - n_max) + n_sum;
            loc_max = n_max;
        } else {
            loc_sum += n_sum * expf(n_max - loc_max);
        }
    }

    // 3. 广播最终结果
    float row_max = __shfl_sync(0xffffffff, loc_max, 0);
    float row_sum = __shfl_sync(0xffffffff, loc_sum, 0);

    // 4. 计算输出
    for (int i = tid; i < N; i += block_size) {
        dst[i] = expf(src[i] - row_max) / row_sum;
    }
}
void launch_softmax(float* d_out, const float* d_in, int B, int N) {
    int block = min(MAX_BLOCK, N);
    int grid = B;
    size_t shmem = 2 * block * sizeof(float);
    softmax_online_shuffle_final<<<grid, block, shmem>>>(d_out, d_in, B, N);
    CHECK_CUDA(cudaGetLastError());
}

// ==============================
// ✅ BENCHMARK 核心函数
// ==============================
void benchmark_softmax(int B, int N) {
    size_t total_elems = B * N;
    size_t bytes = total_elems * sizeof(float);

    printf("===================================================\n");
    printf("            Softmax Benchmark (Online)\n");
    printf(" Shape: (%d, %d) | Elements: %zu | Size: %.2f MB\n",
           B, N, total_elems, bytes / 1024.0 / 1024.0);
    printf(" Warmup: %d iter, Bench: %d iter\n", WARMUP_ITER, BENCH_ITER);
    printf("===================================================\n");

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    std::vector<float> h_in(total_elems);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10, 10);
    for (auto& x : h_in) x = dist(gen);

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < WARMUP_ITER; i++) {
        launch_softmax(d_out, d_in, B, N);
    }
    cudaDeviceSynchronize();

    // Bench
    float min_ms = 1e9;
    float max_ms = 0;
    float total_ms = 0;

    for (int i = 0; i < BENCH_ITER; i++) {
        cudaEventRecord(start);
        launch_softmax(d_out, d_in, B, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    float avg_ms = total_ms / BENCH_ITER;
    float bandwidth = bytes / (avg_ms / 1000.0) / 1e9;

    printf(" GPU Time: avg=%.4f ms | min=%.4f ms | max=%.4f ms\n",
           avg_ms, min_ms, max_ms);
    printf(" Bandwidth: %.2f GB/s\n", bandwidth);
    printf("===================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    // 跑 Benchmark！
    benchmark_softmax(2048, 2048);
    return 0;
}