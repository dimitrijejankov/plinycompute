# create the target
add_pdb_integration_test(TestMatrixMultiply)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestMatrixMultiply)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/tools/cmake")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
set(CMAKE_CUDA_FLAGS "-arch=sm_75")

set (CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set (CUDA_BLAS_LIB "/usr/lib/x86_64-linux-gnu/libcublas.so")
set (CUDA_DART_LIB "/usr/local/cuda-10.2/lib64/libcudart.so")

include_directories(${CUDA_INCLUDE_DIR})
include_directories("/usr/local/cuda-10.2/samples/common/inc/")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/cuda/headers/")

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestMatrixMultiply/sharedLibraries/source/*.cc")
file(GLOB cuda_source_files "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*/*.cu"
        "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*/*.cc"
        "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cu"
        "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cc")

foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    add_dependencies(shared-libraries ${fileName})
endforeach()

enable_language("CUDA")
add_library(CUDA_LIBRARIES SHARED ${cuda_source_files})

target_link_libraries(MatrixBlockData CUDA_LIBRARIES)
target_link_libraries(MatrixMultiplyJoin CUDA_LIBRARIES)

# add build dependencies to shared libraries it uses
add_dependencies(TestMatrixMultiply MatrixBlock)
add_dependencies(TestMatrixMultiply MatrixBlockData)
add_dependencies(TestMatrixMultiply MatrixBlockMeta)
add_dependencies(TestMatrixMultiply MatrixMultiplyAggregation)
add_dependencies(TestMatrixMultiply MatrixMultiplyJoin)
add_dependencies(TestMatrixMultiply MatrixScanner)
add_dependencies(TestMatrixMultiply MatrixWriter)

target_link_libraries(TestMatrixMultiply ${CUDA_DART_LIB})
target_link_libraries(TestMatrixMultiply ${CUDA_BLAS_LIB})
target_link_libraries(TestMatrixMultiply CUDA_LIBRARIES)