#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "cuda_runtime.h"

// 强制忽略 CCCL 版本过时警告，解决编译兼容性问题
#define CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#include "cuda.h"
// CUB 高性能库：提供 CUDA 块级/设备级并行归约优化实现
#include "cub/cub.cuh"

// CUDA 错误检查宏：遇到错误立即打印信息并退出，方便调试
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); }

// 块归约使用的线程块大小（固定 128 线程/块，匹配 CUB 优化配置）
constexpr int kReduceBlockSize = 128;
// 预热迭代次数：让 GPU 进入稳定工作状态，排除首次运行开销
const int WARMUP_ITER = 10;
// 基准测试迭代次数：多次运行取平均，保证性能数据准确
const int BENCH_ITER = 50;

// ======================= 核心设备函数与归约逻辑 =======================
/**
 * @brief 设备端绝对值函数（通用模板，支持 float/double 等数值类型）
 * @param a 输入数值
 * @return 输入数值的绝对值
 */
template<typename T>
__device__ T abs_func(const T& a) {
  return abs(a);
}

/**
 * @brief 设备端最大值函数
 * @param a 第一个输入值
 * @param b 第二个输入值
 * @return 两个值中的较大者
 */
template<typename T>
__device__ T max_func(const T a, const T b) {
  return a > b ? a : b;
}

/**
 * @brief 绝对值最大值归约操作符（CUB 自定义归约算子）
 * 功能：计算两个数的绝对值，返回绝对值更大的那个数
 */
template<typename T>
struct AbsMaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max_func(abs_func(a), abs_func(b));
  }
};

/**
 * @brief 块级全归约：计算当前线程块内所有线程的【绝对值最大值】
 * @param val 当前线程携带的局部最大值
 * @return 整个线程块的最终绝对值最大值（所有线程都能获取到）
 */
template<typename T>
__inline__ __device__ T BlockAllReduceAbsMax(T val) {
  // 定义 CUB 块归约类型：指定数据类型和块大小
  typedef cub::BlockReduce<T, kReduceBlockSize> BlockReduce;
  // 声明 CUB 归约需要的共享内存临时存储
  __shared__ typename BlockReduce::TempStorage temp_storage;
  // 共享内存：存储块归约的最终结果
  __shared__ T final_result;

  // 执行块级归约：使用自定义 AbsMaxOp 计算块内绝对值最大值
  T result = BlockReduce(temp_storage).Reduce(val, AbsMaxOp<T>());

  // 线程 0 负责将归约结果写入共享内存
  if (threadIdx.x == 0) { final_result = result; }
  // 同步块内所有线程：保证所有线程都能读到最终结果
  __syncthreads();

  return final_result;
}

/**
 * @brief CUDA 核心核函数：按行计算绝对值最大值并归一化
 * 功能：对二维矩阵的每一行，先求该行所有元素的绝对值最大值，再用每个元素除以该最大值
 * @tparam T 数据类型（float）
 * @tparam IDX 索引类型（int）
 * @param x 输入/输出矩阵（设备内存）
 * @param row_size 矩阵行数
 * @param col_size 矩阵列数
 */
template<typename T, typename IDX>
__global__ void ReduceScaleBlockKernel(T* x, IDX row_size, IDX col_size) {
  // 网格跨步循环：处理行数 > 网格块数的情况（一个线程块处理多行）
  for(int32_t row = blockIdx.x, step=gridDim.x; row < row_size; row+= step){
    // 当前线程的局部最大值（初始化为 0）
    T thread_scale_factor = 0.0; 

    // 第一步：线程按列跨步遍历，计算本行内的局部绝对值最大值
    for(int32_t col=threadIdx.x; col < col_size; col+= blockDim.x){
      // 计算二维矩阵展平后的一维索引
      IDX idx = row * col_size + col; 
      // 读取当前元素
      T x_val = x[idx];
      // 更新线程局部最大值（保留更大的绝对值）
      thread_scale_factor = max_func(thread_scale_factor, abs_func(x_val)); 
    }

    // 第二步：块级归约 → 得到当前行的【全局绝对值最大值】
    T row_scale_factor = BlockAllReduceAbsMax<T>(thread_scale_factor); 

    // 第三步：用行最大值归一化当前行所有元素
    for(int32_t col=threadIdx.x; col < col_size; col+=blockDim.x){
      IDX idx = row * col_size + col; 
      // 元素 / 行绝对值最大值 → 归一化到 [-1, 1]
      x[idx] /= row_scale_factor;
    }
  }
}

// ======================= 主机端调用封装 =======================
/**
 * @brief 启动 CUDA 核函数的封装接口
 * @param d_x 设备端矩阵指针
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 */
void LaunchReduceScale(float* d_x, int rows, int cols) {
    // 配置线程块：每个块 128 线程
    dim3 block(kReduceBlockSize);
    // 配置网格：最多启动 8192 个块（GPU 资源友好）
    dim3 grid(std::min(rows, 8192));

    // 启动核函数
    ReduceScaleBlockKernel<float, int><<<grid, block>>>(d_x, rows, cols);
    // 检查核函数启动是否报错
    CHECK_CUDA(cudaGetLastError());
    // 等待核函数执行完成
    cudaDeviceSynchronize();
}

// ======================= CPU 参考实现（用于验证正确性） =======================
/**
 * @brief CPU 版本绝对值最大归一化（串行，结果作为标准答案）
 */
void CpuReduceScale(float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float max_abs = 0.0f;
        // 遍历行，求绝对值最大值
        for (int c = 0; c < cols; c++)
            max_abs = fmaxf(max_abs, fabs(x[r * cols + c]));
        // 归一化当前行
        for (int c = 0; c < cols; c++)
            x[r * cols + c] /= max_abs;
    }
}

/**
 * @brief 对比 CPU 和 GPU 结果，验证计算正确性
 * @param cpu CPU 计算结果
 * @param gpu GPU 计算结果
 * @param size 元素总个数
 * @param max_err 输出最大误差
 * @return 误差是否在允许范围内
 */
bool CheckResult(const float* cpu, const float* gpu, int size, float& max_err) {
    max_err = 0;
    for (int i = 0; i < size; i++)
        max_err = fmax(max_err, fabs(cpu[i] - gpu[i]));
    // 误差小于 1e-5 视为正确
    return max_err < 1e-5;
}

// ======================= 性能基准测试 =======================
/**
 * @brief 性能测试函数：计算运行时间和内存带宽
 * @param B 行数
 * @param N 列数
 * @param ms 输出单次运行耗时（毫秒）
 * @param bw 输出有效内存带宽（GB/s）
 */
void benchmark(int B, int N, float& ms, float& bw) {
    // 计算总数据字节数
    size_t bytes = B * N * sizeof(float);
    // 分配设备内存
    float *d_x;
    CHECK_CUDA(cudaMalloc(&d_x, bytes));

    // 生成随机测试数据（范围 -5 ~ 5）
    std::vector<float> h_x(B*N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : h_x) v = dist(gen);

    // 数据从主机拷贝到设备
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    // 创建 CUDA 事件（高精度计时）
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // 预热：让 GPU 进入稳定状态
    for (int i = 0; i < WARMUP_ITER; i++)
        LaunchReduceScale(d_x, B, N);

    // 正式性能测试
    cudaEventRecord(s);
    for (int i = 0; i < BENCH_ITER; i++)
        LaunchReduceScale(d_x, B, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    // 计算平均耗时和带宽
    float total;
    cudaEventElapsedTime(&total, s, e);
    ms = total / BENCH_ITER;
    // 带宽 = 2 * 数据量 / 时间（读 + 写）
    bw = 2 * bytes / (ms / 1000.0f) / 1e9;

    // 正确性验证
    LaunchReduceScale(d_x, B, N);
    std::vector<float> gpu(B*N);
    cudaMemcpy(gpu.data(), d_x, bytes, cudaMemcpyDeviceToHost);
    std::vector<float> cpu = h_x;
    CpuReduceScale(cpu.data(), B, N);

    float err;
    bool ok = CheckResult(cpu.data(), gpu.data(), B*N, err);
    printf("[AbsMaxScale] %.4f ms | %.2f GB/s | Check: %s | err:%.6f\n",
           ms, bw, ok ? "PASS" : "FAIL", err);

    // 释放资源
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cudaFree(d_x);
}

/**
 * @brief 打印当前使用的 GPU 信息
 */
void print_gpu_info() {
    int d;
    cudaGetDevice(&d);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, d);
    printf("GPU: %s\n", prop.name);
}

// ======================= 主函数 =======================
int main() {
    print_gpu_info();

    // 测试矩阵尺寸：行数固定 4096，列数从 1024 到 8192
    int shapes[][2] = {
        {4096, 1024},
        {4096, 2048},
        {4096, 4096},
        {4096, 8192}
    };

    // 遍历所有尺寸，执行测试
    for (auto& sh : shapes) {
        int B = sh[0];
        int N = sh[1];
        float t, b;

        printf("\n===== AbsMaxScale %d x %d =====\n", B, N);
        benchmark(B, N, t, b);
    }

    return 0;
}