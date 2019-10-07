# adds one pdb integration test to CMAKE
function(add_pdb_application test-name)
    # get the directory
    get_filename_component(test-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

    # add the inlcude path
    include_directories("${test-path}/sharedLibraries/headers")

    # create the target
    add_executable(${test-name} "${test-path}/${test-name}.cc"
            $<TARGET_OBJECTS:logical-plan-parser>)

    # link it to the required libraries
    target_link_libraries(${test-name} pdb-tests-common)
    target_link_libraries(${test-name} ${GSL_LIBRARIES})
endfunction(add_pdb_application)



include(${PROJECT_SOURCE_DIR}/applications/TestLDA/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/applications/TestGMM/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/applications/TestMatrixMultiply/CMakeLists.txt)

#include(${PROJECT_SOURCE_DIR}/applications/TestSparseMultiply/CMakeLists.txt)
# TODO had to disable the linear algebra so it does not confilcit with my stuff
#include(${PROJECT_SOURCE_DIR}/applications/TestLinearAlgebra/CMakeLists.txt)
