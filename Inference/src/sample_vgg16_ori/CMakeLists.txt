cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(sample)
set(CMAKE_CXX_STANDARD 14)


macro(find_library_create_target target_name lib libtype hints)
    message(STATUS "========================= Importing and creating target ${target_name} ==========================")
    message(STATUS "Looking for library ${lib}")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        find_library(${lib}_LIB_PATH ${lib}${TRT_DEBUG_POSTFIX} HINTS ${hints} NO_DEFAULT_PATH)
    endif()
    find_library(${lib}_LIB_PATH ${lib} HINTS ${hints} NO_DEFAULT_PATH)
    find_library(${lib}_LIB_PATH ${lib})
    message(STATUS "Library that was found ${${lib}_LIB_PATH}")
    add_library(${target_name} ${libtype} IMPORTED)
    set_property(TARGET ${target_name} PROPERTY IMPORTED_LOCATION ${${lib}_LIB_PATH})
    message(STATUS "==========================================================================================")
endmacro()


get_filename_component(TRT_DIR ../../ ABSOLUTE)

SET(SAMPLE_SOURCES sample_vgg16_ori.cpp)
SET(SAMPLE_HEADERS sample_vgg16_ori.h)
# SET(CUDA_FUNC_SOURCES ${TRT_DIR}/cuda_func/buffer_copy.cu)
# SET(CUDA_FUNC_HEADERS ${TRT_DIR}/cuda_func/buffer_copy.cuh)
SET(SAMPLE_OUT_DIR ${PROJECT_BINARY_DIR}/out)
SET(TRT_LIB_DIR /home/slzhang/projects/TensorRT-8.0.1.6/lib)
SET(TRT_OUT_DIR /home/slzhang/projects/ETBA/Inference/src/sample_vgg16_ori/build/out)
set(CUDA_INSTALL_DIR /usr/local/cuda CACHE STRING "CUDA directory")
set(ONNX_INCLUDE_DIR "${TRT_DIR}/parsers/onnx")
set(SAMPLES_DIR ${TRT_DIR}/common)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_INSTALL_DIR})
set(SAMPLES_COMMON_SOURCES ${SAMPLES_DIR}/logger.cpp)

add_subdirectory(${TRT_DIR}/cuda_func cuda_func.out)
set (EXTRA_LIBS ${EXTRA_LIBS} cuda_func)
# set(TORCH_LIB_DIR /home/slzhang/mytool/libtorch/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

add_executable(sample
    ${SAMPLE_SOURCES}
    ${SAMPLE_HEADERS}
    ${SAMPLES_COMMON_SOURCES}
)


target_include_directories(sample
    PUBLIC ${TRT_DIR}/include
    PUBLIC ${ONNX_INCLUDE_DIR}
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${SAMPLES_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

find_library_create_target(nvinfer nvinfer SHARED ${TRT_LIB_DIR})
find_library_create_target(nvonnxparser nvonnxparser SHARED ${TRT_LIB_DIR})
find_library(CUDNN_LIB cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)

# target_compile_options(${TARGET_NAME} PUBLIC "-fno-rtti")
set(SAMPLE_DEP_LIBS
    ${CUDART_LIB}
    ${CUBLAS_LIB}
    ${CUDNN_LIB}
    nvinfer
    nvonnxparser
    ${RT_LIB}
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${EXTRA_LIBS}
)

target_link_libraries(sample
    ${SAMPLE_DEP_LIBS}
    -Wl,--unresolved-symbols=ignore-in-shared-libs
)

# find_package(Torch REQUIRED)
find_package(Torch REQUIRED PATHS /home/slzhang/mytool/pytorch/torch)
message(STATUS ${TORCH_LIBRARIES})
# message(STATUS ${TORCH_INCLUDE_DIRS})
# include_directories(${TORCH_INCLUDE_DIRS})
target_link_libraries(sample ${TORCH_LIBRARIES})

set_target_properties(sample PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
# set_target_properties(sample PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(sample
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${SAMPLE_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${SAMPLE_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${SAMPLE_OUT_DIR}"
)

message(STATUS ${SAMPLE_OUT_DIR})

install(TARGETS sample
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)