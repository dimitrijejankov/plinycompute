# create the target
add_pdb_integration_test(TestComputation)
target_link_libraries(TestComputation ${MKL_LIBRARIES})

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestComputation)

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestComputation/sharedLibraries/source/*.cc")
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${MKL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()

# add build dependencies to shared libraries it uses
add_dependencies(TestComputation MatrixMultiplyAggregation)
add_dependencies(TestComputation MatrixMultiplyJoin)
add_dependencies(TestComputation MatrixSumJoin)
add_dependencies(TestComputation MatrixScanner)
add_dependencies(TestComputation MatrixWriter)