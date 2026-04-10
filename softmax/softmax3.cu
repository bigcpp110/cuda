#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <math.h>
#include <thread>
#include <mutex>

// CUDA 错误检查宏
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); }

const int BLOCK_SIZE = 256;
const int WARMUP_ITER = 10;
const int BENCH_ITER = 50;

// ==============================================
// 原版 Online Softmax 核函数
// ==============================================
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

// ==============================================
// 优化版 Online Softmax (float4)
// ==============================================
template <int BLOCK_SIZE>
__global__ void softmax_online_float4(
    float* out, const float* inp, int B, int N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= B) return;

    const float* row_in = inp + row * N;
    float* row_out = out + row * N;

    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    float loc_max = -1e20f;
    float loc_sum = 0.0f;

    int vec_n = N / 4;
    for (int i = tid; i < vec_n; i += BLOCK_SIZE) {
        float4 v = *(const float4*)(row_in + i * 4);
        float val[4] = {v.x, v.y, v.z, v.w};

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float x = val[k];
            if (x > loc_max) {
                loc_sum = loc_sum * expf(loc_max - x) + 1.0f;
                loc_max = x;
            } else {
                loc_sum += expf(x - loc_max);
            }
        }
    }

    int start = vec_n * 4;
    for (int i = start + tid; i < N; i += BLOCK_SIZE) {
        float x = row_in[i];
        if (x > loc_max) {
            loc_sum = loc_sum * expf(loc_max - x) + 1.0f;
            loc_max = x;
        } else {
            loc_sum += expf(x - loc_max);
        }
    }

    s_max[tid] = loc_max;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float row_max = s_max[0];

    s_sum[tid] = loc_sum * expf(loc_max - row_max);
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float row_sum = s_sum[0];
    float inv_sum = 1.0f / fmaxf(row_sum, 1e-10f);

    for (int i = tid; i < vec_n; i += BLOCK_SIZE) {
        float4 v = *(const float4*)(row_in + i * 4);
        float4 o;
        o.x = expf(v.x - row_max) * inv_sum;
        o.y = expf(v.y - row_max) * inv_sum;
        o.z = expf(v.z - row_max) * inv_sum;
        o.w = expf(v.w - row_max) * inv_sum;
        *(float4*)(row_out + i * 4) = o;
    }
    for (int i = start + tid; i < N; i += BLOCK_SIZE) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

// ==============================================
// CPU 参考实现
// ==============================================
void softmax_cpu(float* out, const float* in, int B, int N) {
    for (int b = 0; b < B; b++) {
        const float* r_in = in + b * N;
        float* r_out = out + b * N;
        float m = -1e20f, s = 0.0f;
        for (int i = 0; i < N; i++) m = fmax(m, r_in[i]);
        for (int i = 0; i < N; i++) s += expf(r_in[i] - m);
        float inv = 1.0f / s;
        for (int i = 0; i < N; i++) r_out[i] = expf(r_in[i] - m) * inv;
    }
}

// 验证结果
bool verify_result(const float* cpu, const float* gpu, int size, float& max_error) {
    max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        max_error = fmax(max_error, fabsf(cpu[i] - gpu[i]));
    }
    return max_error < 1e-4;
}

enum KernelType { ORIGIN, OPT_SAFE };

// ==============================================
// 单卡核函数启动
// ==============================================
void launch_softmax(float* d_out, const float* d_in, int B, int N, KernelType type) {
    int grid = B;
    int block = BLOCK_SIZE;

    if (type == ORIGIN) {
        size_t shmem = 2 * block * sizeof(float);
        softmax_cuda_online<<<grid, block, shmem>>>(d_out, d_in, B, N);
    } else {
        softmax_online_float4<BLOCK_SIZE><<<grid, block>>>(d_out, d_in, B, N);
    }

    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
}

// ==============================================
// 双GPU并行任务结构体
// ==============================================
struct GpuTask {
    int device;
    const float* h_in_part;
    float* h_out_part;
    int B_part;
    int N;
    KernelType type;
    float time_ms;
    float bw_gbs;
};

std::mutex g_mutex;

// ==============================================
// 单卡线程执行函数（被双线程同时调用）
// ==============================================
void run_gpu_task(GpuTask& task) {
    // 绑定到指定GPU
    CHECK_CUDA(cudaSetDevice(task.device));

    int B = task.B_part;
    int N = task.N;
    size_t bytes = B * N * sizeof(float);

    // 分配显存
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    // 拷贝数据到GPU
    CHECK_CUDA(cudaMemcpy(d_in, task.h_in_part, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // 预热
    for (int i = 0; i < WARMUP_ITER; i++)
        launch_softmax(d_out, d_in, B, N, task.type);

    // 基准测试
    cudaEventRecord(s);
    for (int i = 0; i < BENCH_ITER; i++)
        launch_softmax(d_out, d_in, B, N, task.type);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    // 计时
    float total;
    cudaEventElapsedTime(&total, s, e);
    task.time_ms = total / BENCH_ITER;
    task.bw_gbs = 2 * bytes / (task.time_ms / 1000.0f) / 1e9;

    // 拷贝结果回CPU
    CHECK_CUDA(cudaMemcpy(task.h_out_part, d_out, bytes, cudaMemcpyDeviceToHost));

    // 释放
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cudaFree(d_in);
    cudaFree(d_out);
}

// ==============================================
// 双GPU并行基准测试
// ==============================================
void benchmark_2gpu(int B, int N, KernelType type, const char* name, float& ms, float& bw) {
    if (N % 4 != 0) {
        printf("ERROR: N must be multiple of 4\n");
        exit(1);
    }

    // 获取GPU数量
    int nGpus;
    CHECK_CUDA(cudaGetDeviceCount(&nGpus));
    if (nGpus < 2) {
        printf("ERROR: Need at least 2 GPUs!\n");
        exit(1);
    }

    // 数据均分：GPU0 算前一半，GPU1 算后一半
    int B0 = B / 2;
    int B1 = B - B0;
    size_t total_bytes = B * N * sizeof(float);
    size_t bytes0 = B0 * N * sizeof(float);

    // 生成完整输入数据
    std::vector<float> h_in(B * N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10, 10);
    for (auto& x : h_in) x = dist(gen);

    // 输出缓冲区
    std::vector<float> h_out_total(B * N);

    // 构造双GPU任务
    GpuTask task0{0, h_in.data(), h_out_total.data(), B0, N, type, 0, 0};
    GpuTask task1{1, h_in.data() + B0 * N, h_out_total.data() + B0 * N, B1, N, type, 0, 0};

    // 双线程并行执行
    std::thread t1(run_gpu_task, std::ref(task0));
    std::thread t2(run_gpu_task, std::ref(task1));
    t1.join();
    t2.join();

    // 整体耗时 = 最慢的那张卡的时间（真正的端到端耗时）
    ms = std::max(task0.time_ms, task1.time_ms);
    // 整体带宽 = 总数据量 / 总时间
    bw = 2 * total_bytes / (ms / 1000.0f) / 1e9;

    // 打印双GPU性能
    std::lock_guard<std::mutex> lock(g_mutex);
    printf("[%s] %.4f ms | %.2f GB/s | ", name, ms, bw);

    // 正确性验证
    std::vector<float> cpu_ref(B * N);
    softmax_cpu(cpu_ref.data(), h_in.data(), B, N);
    float err;
    bool ok = verify_result(cpu_ref.data(), h_out_total.data(), B * N, err);
    printf("Check: %s | err:%.6f\n", ok ? "PASS" : "FAIL", err);
}

// ==============================================
// 打印双GPU信息
// ==============================================
void print_gpu_info() {
    int n;
    CHECK_CUDA(cudaGetDeviceCount(&n));
    printf("Detected %d GPUs:\n", n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp prop{};
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s\n", i, prop.name);
    }
}

// ==============================================
// 主函数：双GPU版本测试
// ==============================================
int main() {
    print_gpu_info();

    int shapes[][2] = {
        {4096, 1024},
        {4096, 2048},
        {4096, 4096},
        {4096, 8192}
    };

    for (auto& sh : shapes) {
        int B = sh[0];
        int N = sh[1];
        float t1, t2, b1, b2;

        printf("\n===== Softmax %d x %d (2-GPU Parallel) =====\n", B, N);
        benchmark_2gpu(B, N, ORIGIN,   "Original ", t1, b1);
        benchmark_2gpu(B, N, OPT_SAFE, "OptFloat4", t2, b2);
        printf("Speedup: %.2fx\n", t1 / t2);
    }

    return 0;
}