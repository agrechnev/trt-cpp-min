cmake_minimum_required(VERSION 3.11)
project(trt-cpp-min)
set(CMAKE_CXX_STANDARD 14)
file(COPY model1.onnx DESTINATION .)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")


link_libraries(nvinfer nvonnxparser ${CUDA_LIBRARIES})

add_executable(example1 example1.cpp)
