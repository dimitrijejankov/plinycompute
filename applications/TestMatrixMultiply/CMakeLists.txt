# create the target
add_pdb_integration_test(TestBMM)
add_pdb_integration_test(TestCMM)
add_pdb_integration_test(TestRMM)
target_link_libraries(TestBMM ${MKL_LIBRARIES})
target_link_libraries(TestCMM ${MKL_LIBRARIES})
target_link_libraries(TestRMM ${MKL_LIBRARIES})

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestBMM)
add_dependencies(build-integration-tests TestCMM)
add_dependencies(build-integration-tests TestRMM)

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestMatrixMultiply/sharedLibraries/source/*.cc")
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${MKL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()

# add build dependencies to shared libraries it uses
add_custom_target(TensorBasic)
add_dependencies(TensorBasic TensorBlock)
add_dependencies(TensorBasic TensorBlockData)
add_dependencies(TensorBasic TensorBlockMeta)
add_dependencies(TensorBasic TensorScanner)
add_dependencies(TensorBasic TensorWriter)

add_dependencies(TestBMM TensorBasic)
add_dependencies(TestBMM BCMMAggregation)
add_dependencies(TestBMM BCMMJoin)
add_dependencies(TestCMM TensorBasic)
add_dependencies(TestCMM BCMMAggregation)
add_dependencies(TestCMM BCMMJoin)
add_dependencies(TestRMM TensorBasic)
add_dependencies(TestRMM RMMDuplicateMultiSelection)
add_dependencies(TestRMM RMMAggregation)
add_dependencies(TestRMM RMMJoin)