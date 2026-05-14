cmake_minimum_required(VERSION 3.20)
project(KVCacheDSL LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

include(AddMLIR)
include(AddLLVM)
include(LLVMExternalProjectUtils)
include(HandleLLVMOptions)

include(LLVMConfig)
add_definitions(${LLVM_DEFINITIONS})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
