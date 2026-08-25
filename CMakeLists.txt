cmake_minimum_required(VERSION 3.15)
project(FastVLM_NPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FASTVLM_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtime" CACHE PATH
    "Directory containing the RKLLM/RKNN runtime headers and libraries")
set(FASTVLM_OPENCV_DIR "" CACHE PATH
    "Optional OpenCV_DIR (use the official aarch64 SDK when cross compiling)")

if(FASTVLM_OPENCV_DIR)
    set(OpenCV_DIR "${FASTVLM_OPENCV_DIR}")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(FASTVLM_RUNTIME_INCLUDE "${FASTVLM_RUNTIME_DIR}/include")
set(FASTVLM_RKLLM "${FASTVLM_RUNTIME_DIR}/lib/librkllmrt.so")
set(FASTVLM_RKNN "${FASTVLM_RUNTIME_DIR}/lib/librknnrt.so")
if(NOT EXISTS "${FASTVLM_RKLLM}" OR NOT EXISTS "${FASTVLM_RKNN}")
    message(FATAL_ERROR "runtime/lib must contain librkllmrt.so and librknnrt.so")
endif()

add_executable(FastVLM_NPU src/main.cpp src/image_encoder.cpp src/model_contract.cpp)
target_include_directories(FastVLM_NPU PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${FASTVLM_RUNTIME_INCLUDE}"
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(FastVLM_NPU PRIVATE
    "${FASTVLM_RKLLM}"
    "${FASTVLM_RKNN}"
    ${OpenCV_LIBS})
target_compile_options(FastVLM_NPU PRIVATE -Wall -Wextra -Wpedantic)
# Runtime lookup is deliberately controlled by LD_LIBRARY_PATH so the board can
# choose the project-local 1.3.0 libraries without embedding a host build path.

install(TARGETS FastVLM_NPU RUNTIME DESTINATION .)
install(DIRECTORY runtime/lib/ DESTINATION runtime/lib USE_SOURCE_PERMISSIONS)
