import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton Kernel: Online Softmax（数值稳定）
# -----------------------------------------------------------------------------
@triton.jit
def online_softmax_kernel(
    output_ptr,
    input_ptr,
    stride_b,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = input_ptr + row_idx * stride_b
    out_ptr = output_ptr + row_idx * stride_b

    current_max = -float('inf')
    current_sum = 0.0

    # 分块在线计算 max & sum
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(row_ptr + cols, mask=mask, other=-float('inf'))

        block_max = tl.max(x, 0)
        new_max = tl.maximum(current_max, block_max)

        exp_vals = tl.exp(x - new_max)
        block_sum = tl.sum(exp_vals, 0)

        current_sum = current_sum * tl.exp(current_max - new_max) + block_sum
        current_max = new_max

    # 防除 0
    safe_sum = tl.where(current_sum < 1e-10, 1.0, current_sum)
    inv_sum = 1.0 / safe_sum

    # 写回结果
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(row_ptr + cols, mask=mask)
        val = tl.exp(x - current_max) * inv_sum
        tl.store(out_ptr + cols, val, mask=mask)

# -----------------------------------------------------------------------------
# 对外接口
# -----------------------------------------------------------------------------
def online_softmax_triton(x: torch.Tensor):
    B, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (B,)
    online_softmax_kernel[grid](
        out, x,
        stride_b=N,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# -----------------------------------------------------------------------------
# Benchmark 函数（和 CUDA 风格一致）
# -----------------------------------------------------------------------------
def benchmark(fn, x, warmup=10, test_iter=50):
    # 预热
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(test_iter):
        fn(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / test_iter
    B, N = x.shape
    bytes = B * N * 4 * 2  # float32 读+写
    gbs = bytes / ms / 1e6
    return ms, gbs

# -----------------------------------------------------------------------------
# 主函数：完全对齐你 CUDA 里的 4 种 shape
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    
    # 🔥 完全和你 C++ 代码一样的尺寸
    shapes = [
        (4096, 1024),
        (4096, 2048),
        (4096, 4096),
        (4096, 8192)
    ]

    for B, N in shapes:
        print(f"\n===== Softmax {B} x {N} =====")
        x = torch.randn(B, N, device='cuda', dtype=torch.float32)

        # 1. Triton
        t_tri, bw_tri = benchmark(online_softmax_triton, x)
        # 2. PyTorch
        t_torch, bw_torch = benchmark(lambda x: torch.softmax(x, dim=-1), x)

        # 精度验证
        y_tri = online_softmax_triton(x)
        y_torch = torch.softmax(x, dim=-1)
        max_err = torch.max(torch.abs(y_tri - y_torch)).item()
        check = "PASS" if max_err < 1e-4 else "FAIL"

        # 输出（和 CUDA 输出格式几乎一样）
        print(f"[Triton]  {t_tri:6.3f} ms | {bw_tri:6.2f} GB/s")
        print(f"[PyTorch] {t_torch:6.3f} ms | {bw_torch:6.2f} GB/s")
        print(f"Speedup: {t_torch / t_tri:.2f}x | Check: {check} | err: {max_err:.6f}")