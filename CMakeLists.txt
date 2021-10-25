cmake_minimum_required(VERSION 3.11)
project(trt-cpp-min)
set(CMAKE_CXX_STANDARD 14)
file(COPY model1.onnx DESTINATION .)
file(COPY model2.onnx DESTINATION .)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")


link_libraries(nvinfer nvonnxparser ${CUDA_LIBRARIES})

add_executable(example1 example1.cpp)
add_executable(example2 example2.cpp)
add_executable(example3-save example3-save.cpp)
add_executable(example3-load example3-load.cpp)
add_executable(example4 example4.cpp)
add_executable(example5 example5.cpp)
add_executable(example6 example6.cpp)
add_executable(example7 example7.cpp)
add_executable(example7prof example7prof.cpp)
