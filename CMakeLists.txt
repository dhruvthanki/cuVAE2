cmake_minimum_required(VERSION 3.10)
project(VAE_MNIST LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(vae src/main.cu src/mnist_loader.cpp)
target_link_libraries(vae ${CUDA_LIBRARIES})
