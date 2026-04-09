cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX CUDA)

# C++ / CUDA 版本
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU 架构
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# ==============================================
# 自动找到 softmax 下所有 .cu 文件
# ==============================================
file(GLOB SOFTMAX_CUDA_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/softmax/*.cu"
)

# ==============================================
# 为每个 .cu 生成一个可执行文件
# 🔥 可执行文件直接放在 build/ 目录下
# ==============================================
foreach(file_path ${SOFTMAX_CUDA_FILES})
    # 获取不带后缀的文件名
    get_filename_component(exe_name ${file_path} NAME_WE)

    # 目标名
    set(target_name "softmax_${exe_name}")

    # 添加可执行文件
    add_executable(${target_name} ${file_path})

    # 🔥 关键：输出到【build/】目录
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endforeach()