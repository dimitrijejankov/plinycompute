# create the target
add_pdb_integration_test(TestMatrixMultiply)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestMatrixMultiply)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.1/bin/nvcc")
set(CMAKE_CUDA_FLAGS "-arch=sm_61")

option(USE_CUDA "Use CUDA" ON)
# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestMatrixMultiply/sharedLibraries/source/*.cc")
file(GLOB GPU_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cu")

foreach (file ${GPU_SOURCE})
    get_filename_component(fileName "${file}" NAME_WE)
    if (USE_CUDA)
        enable_language("CUDA")
        add_library(${fileName} SHARED ${file})
        target_link_libraries(TestMatrixMultiply PDBCUDAMatrixMultiple)
    endif ()
endforeach ()


foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${GSL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()


# add build dependencies to shared libraries it uses
add_dependencies(TestMatrixMultiply MatrixBlock)
add_dependencies(TestMatrixMultiply MatrixBlockData)
add_dependencies(TestMatrixMultiply MatrixBlockMeta)
add_dependencies(TestMatrixMultiply MatrixMultiplyAggregation)
add_dependencies(TestMatrixMultiply MatrixMultiplyJoin)
add_dependencies(TestMatrixMultiply MatrixScanner)
add_dependencies(TestMatrixMultiply MatrixWriter)

target_link_libraries(MatrixMultiplyJoin PDBCUDAMatrixMultiple)
add_dependencies(MatrixMultiplyJoin PDBCUDAMatrixMultiple)