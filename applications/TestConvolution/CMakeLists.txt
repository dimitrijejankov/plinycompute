# create the target
add_pdb_integration_test(TestConvolution)
target_link_libraries(TestConvolution ${MKL_LIBRARIES})

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestConvolution)

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestConvolution/sharedLibraries/source/*.cc")
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${MKL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()

# add build dependencies to shared libraries it uses
add_dependencies(TestConvolution MatrixBlock)
add_dependencies(TestConvolution MatrixBlockData)
add_dependencies(TestConvolution MatrixBlockMeta)
add_dependencies(TestConvolution MatrixMultiplyAggregation)
add_dependencies(TestConvolution MatrixMultiplyJoin)
add_dependencies(TestConvolution MatrixScanner)
add_dependencies(TestConvolution MatrixWriter)