#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <math.h>

// CUDA 错误检查宏：调用 CUDA API/核函数后自动检查错误，打印报错信息并退出
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); }

// 线程块大小：256 线程/块（CUDA 常用最优配置）
const int BLOCK_SIZE = 256;
// 预热迭代次数：让 GPU 进入稳定状态，排除首次运行开销
const int WARMUP_ITER = 10;
// 基准测试迭代次数：多次运行取平均，保证性能数据准确
const int BENCH_ITER = 50;

// ==============================================
// 原版 Online Softmax 核函数（单精度浮点，无矢量优化）
// 功能：对二维矩阵每行做在线 Softmax 计算（一次遍历求 max + sum，数值稳定）
// 参数：out 输出数组 | in 输入数组 | B 行数 | N 列数
// ==============================================
__global__ void softmax_cuda_online(
    float* out, const float* in, int B, int N
) {
    // 每个线程块处理 1 行数据（blockIdx.x = 当前行号）
    int row = blockIdx.x;
    // 线程局部 ID（0 ~ 255）
    int tid = threadIdx.x;
    // 线程块总线程数
    int block_size = blockDim.x;
    // 越界保护：行号超过总行数直接返回
    if (row >= B) return;

    // 定位到当前行的输入/输出起始地址
    const float* row_in = in + row * N;
    float* row_out = out + row * N;

    // 动态共享内存：分为两块，分别存储局部最大值、局部求和值
    extern __shared__ float smem[];
    float* s_max = smem;                // 共享内存：存储每个线程的局部最大值
    float* s_sum = smem + block_size;    // 共享内存：存储每个线程的局部求和值

    // 线程局部最大值（初始化为极小值，保证任何输入都能更新它）
    float loc_max = -1e20f;
    // 线程局部求和值（Softmax 指数和）
    float loc_sum = 0.0f;

    // ======================
    // 第一步：在线遍历计算 局部 max + sum（数值稳定，仅一次遍历）
    // 线程跨步循环：每个线程处理多行/列，充分利用 GPU 并行
    // ======================
    for (int i = tid; i < N; i += block_size) {
        float x = row_in[i];
        // 在线更新最大值与和：新值更大 → 缩放旧和；新值更小 → 直接累加指数
        if (x > loc_max) {
            loc_sum = loc_sum * expf(loc_max - x) + 1.0f;
            loc_max = x;
        } else {
            loc_sum += expf(x - loc_max);
        }
    }

    // 将线程局部结果写入共享内存
    s_max[tid] = loc_max;
    s_sum[tid] = loc_sum;
    __syncthreads();    // 同步：等待所有线程写入完成

    // ======================
    // 第二步：共享内存归约 → 计算整行全局 max + sum
    // 树形归约：每次将数据折半合并
    // ======================
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 取出两个待合并的最大值与和
            float m0 = s_max[tid];
            float m1 = s_max[tid + s];
            float s0 = s_sum[tid];
            float s1 = s_sum[tid + s];

            // 数值稳定合并：以更大的值为基准，缩放较小值的和
            if (m1 > m0) {
                s0 = s0 * expf(m0 - m1);
                m0 = m1;
            } else {
                s1 = s1 * expf(m1 - m0);
            }
            // 写回合并结果
            s_max[tid] = m0;
            s_sum[tid] = s0 + s1;
        }
        __syncthreads();    // 每轮归约后同步
    }

    // 归约完成：0 号线程存储整行最终最大值与和
    float row_max = s_max[0];
    float row_sum = s_sum[0];

    // ======================
    // 第三步：计算最终 Softmax 结果并写回
    // ======================
    for (int i = tid; i < N; i += block_size) {
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
    }
}

// ==============================================
// 优化版 Online Softmax：使用 float4 矢量内存访问（核心性能优化）
// 一次加载/存储 4 个 float，大幅提升内存带宽利用率
// 模板参数 BLOCK_SIZE：编译期确定块大小，优化性能
// ==============================================
template <int BLOCK_SIZE>
__global__ void softmax_online_float4(
    float* out, const float* inp, int B, int N)
{
    // 每个线程块处理 1 行
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= B) return;

    // 定位当前行输入/输出地址
    const float* row_in = inp + row * N;
    float* row_out = out + row * N;

    // 静态共享内存：编译期确定大小，比动态共享内存更快
    __shared__ float s_max[BLOCK_SIZE];   // 存储局部最大值
    __shared__ float s_sum[BLOCK_SIZE];   // 存储局部求和值

    // 线程局部初始化
    float loc_max = -1e20f;
    float loc_sum = 0.0f;

    // ======================
    // 核心优化：float4 矢量读取（一次读 4 个 float）
    // ======================
    int vec_n = N / 4;    // 矢量个数：总列数 / 4
    // 线程跨步遍历矢量
    for (int i = tid; i < vec_n; i += BLOCK_SIZE) {
        // float4 矢量加载：内存访问合并，带宽提升 4 倍
        float4 v = *(const float4*)(row_in + i * 4);
        // 拆分为 4 个独立浮点数
        float val[4] = {v.x, v.y, v.z, v.w};

        // 循环展开：编译器优化，减少循环开销
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float x = val[k];
            // 在线更新 max + sum（与原版逻辑一致）
            if (x > loc_max) {
                loc_sum = loc_sum * expf(loc_max - x) + 1.0f;
                loc_max = x;
            } else {
                loc_sum += expf(x - loc_max);
            }
        }
    }

    // 处理非 4 对齐的剩余元素（尾巴）
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

    // ======================
    // 归约第一步：计算整行全局最大值
    // ======================
    s_max[tid] = loc_max;
    __syncthreads();
    // 树形归约求最大值
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float row_max = s_max[0];

    // ======================
    // 归约第二步：计算整行全局指数和
    // ======================
    // 先缩放局部和，保证数值稳定
    s_sum[tid] = loc_sum * expf(loc_max - row_max);
    __syncthreads();
    // 树形归约求和
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float row_sum = s_sum[0];
    // 安全倒数：防止除 0
    float inv_sum = 1.0f / fmaxf(row_sum, 1e-10f);

    // ======================
    // float4 矢量写回：一次写 4 个结果
    // ======================
    for (int i = tid; i < vec_n; i += BLOCK_SIZE) {
        float4 v = *(const float4*)(row_in + i * 4);
        float4 o;
        // 并行计算 4 个元素的 Softmax
        o.x = expf(v.x - row_max) * inv_sum;
        o.y = expf(v.y - row_max) * inv_sum;
        o.z = expf(v.z - row_max) * inv_sum;
        o.w = expf(v.w - row_max) * inv_sum;
        // 矢量存储
        *(float4*)(row_out + i * 4) = o;
    }
    // 写回剩余尾巴元素
    for (int i = start + tid; i < N; i += BLOCK_SIZE) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

// ==============================================
// CPU 参考实现：串行 Softmax，用于验证 GPU 结果正确性
// ==============================================
void softmax_cpu(float* out, const float* in, int B, int N) {
    for (int b = 0; b < B; b++) {
        const float* r_in = in + b * N;
        float* r_out = out + b * N;
        // 第一步：求行最大值
        float m = -1e20f, s = 0.0f;
        for (int i = 0; i < N; i++) m = fmax(m, r_in[i]);
        // 第二步：求指数和
        for (int i = 0; i < N; i++) s += expf(r_in[i] - m);
        float inv = 1.0f / s;
        // 第三步：计算最终结果
        for (int i = 0; i < N; i++) r_out[i] = expf(r_in[i] - m) * inv;
    }
}

// 验证 CPU 与 GPU 结果是否一致：误差 <1e-4 视为正确
bool verify_result(const float* cpu, const float* gpu, int size, float& max_error) {
    max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        max_error = fmax(max_error, fabsf(cpu[i] - gpu[i]));
    }
    return max_error < 1e-4;
}

// ==============================================
// 核函数启动封装：统一调用原版/优化版 Softmax
// ==============================================
// 核函数类型枚举：原版 / 矢量优化版
enum KernelType { ORIGIN, OPT_SAFE };

void launch_softmax(float* d_out, const float* d_in, int B, int N, KernelType type) {
    // 网格配置：一个块处理一行，网格大小 = 行数
    int grid = B;
    // 块大小：256 线程
    int block = BLOCK_SIZE;

    if (type == ORIGIN) {
        // 原版：需要动态共享内存（2 * 块大小）
        size_t shmem = 2 * block * sizeof(float);
        softmax_cuda_online<<<grid, block, shmem>>>(d_out, d_in, B, N);
    } else {
        // 优化版：静态共享内存，无需指定
         softmax_online_float4<BLOCK_SIZE><<<grid, block>>>(d_out, d_in, B, N);
    }

    // 检查核函数启动错误
    CHECK_CUDA(cudaGetLastError());
    // 等待核函数执行完成
    cudaDeviceSynchronize();
}

// ==============================================
// 性能基准测试：计算运行时间、内存带宽、验证结果
// ==============================================
void benchmark(int B, int N, KernelType type, const char* name, float& ms, float& bw) {
    // 限制：优化版要求列数是 4 的倍数（float4 对齐）
    if (N % 4 != 0) {
        printf("ERROR: N must be multiple of 4\n");
        exit(1);
    }

    // 计算总数据字节数
    size_t bytes = B * N * sizeof(float);
    // 分配设备内存
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    // 生成随机测试数据（范围 -10 ~ 10）
    std::vector<float> h_in(B*N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10, 10);
    for (auto& x : h_in) x = dist(gen);

    // 数据从主机拷贝到设备
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // CUDA 事件：高精度计时
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // 预热运行
    for (int i = 0; i < WARMUP_ITER; i++)
        launch_softmax(d_out, d_in, B, N, type);

    // 正式测试：计时
    cudaEventRecord(s);
    for (int i = 0; i < BENCH_ITER; i++)
        launch_softmax(d_out, d_in, B, N, type);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    // 计算平均耗时
    float total;
    cudaEventElapsedTime(&total, s, e);
    ms = total / BENCH_ITER;
    // 计算有效内存带宽（读 + 写 = 2 次数据搬运）
    bw = 2 * bytes / (ms / 1000.0f) / 1e9;

    // 打印性能信息
    printf("[%s] %.4f ms | %.2f GB/s | ", name, ms, bw);

    // 正确性验证
    launch_softmax(d_out, d_in, B, N, type);
    std::vector<float> gpu(B*N), cpu(B*N);
    cudaMemcpy(gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    softmax_cpu(cpu.data(), h_in.data(), B, N);

    float err;
    bool ok = verify_result(cpu.data(), gpu.data(), B*N, err);
    printf("Check: %s | err:%.6f\n", ok ? "PASS" : "FAIL", err);

    // 释放资源
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cudaFree(d_in);
    cudaFree(d_out);
}

// 打印当前使用的 GPU 信息
void print_gpu_info() {
    int d;
    cudaGetDevice(&d);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, d);
    printf("GPU: %s\n", prop.name);
}

// ==============================================
// 主函数：测试不同矩阵尺寸，对比原版与优化版性能
// ==============================================
int main() {
    print_gpu_info();

    // 测试矩阵尺寸：行数固定 4096，列数 1024/2048/4096/8192
    int shapes[][2] = {
        {4096, 1024},
        {4096, 2048},
        {4096, 4096},
        {4096, 8192}
    };

    // 遍历所有尺寸，分别测试原版与矢量优化版
    for (auto& sh : shapes) {
        int B = sh[0];
        int N = sh[1];
        float t1, t2, b1, b2;

        printf("\n===== Softmax %d x %d =====\n", B, N);
        benchmark(B, N, ORIGIN,   "Original ", t1, b1);    // 原版
        benchmark(B, N, OPT_SAFE, "OptFloat4", t2, b2);    // 优化版
        printf("Speedup: %.2fx\n", t1 / t2);                // 计算加速比
    }

    return 0;
}