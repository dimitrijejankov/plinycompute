# create the target
add_pdb_integration_test(TestBackpropLayer)
target_link_libraries(TestBackpropLayer ${MKL_LIBRARIES})

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestBackpropLayer)

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestBackpropLayer/sharedLibraries/source/*.cc")
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${MKL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()

# add build dependencies to shared libraries it uses
add_dependencies(TestBackpropLayer BPStrip)
add_dependencies(TestBackpropLayer BPStripData)
add_dependencies(TestBackpropLayer BPStripMeta)
add_dependencies(TestBackpropLayer BPMultiplyAggregation)
add_dependencies(TestBackpropLayer BPMultiplyJoin)
add_dependencies(TestBackpropLayer BPScanner)
add_dependencies(TestBackpropLayer BPWriter)