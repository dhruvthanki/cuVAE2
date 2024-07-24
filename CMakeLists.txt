cmake_minimum_required(VERSION 3.8)
project(cuVAE2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

find_package(CUDA REQUIRED)

add_executable(vae src/main.cu)

target_include_directories(vae PRIVATE include)

target_link_libraries(vae ${CUDA_LIBRARIES})
