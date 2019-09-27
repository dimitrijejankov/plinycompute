# create the target
add_pdb_integration_test(TestMatrixMultiply)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestMatrixMultiply)

option(USE_CUDA "Use CUDA" ON)
# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestMatrixMultiply/sharedLibraries/source/*.cc")
file(GLOB GPU_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cu")

foreach (file ${GPU_SOURCE})
    get_filename_component(fileName "${file}" NAME_WE)
    if (USE_CUDA)
        enable_language("CUDA")
        add_library(${fileName} MODULE ${file})
        target_link_libraries(${fileName} pdb-shared-common)
        target_link_libraries(${fileName} ${GSL_LIBRARIES})

        add_library(${fileName}_share STATIC ${file})
        target_link_libraries(TestMatrixMultiply ${fileName}_share)

        add_dependencies(shared-libraries ${fileName})
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
