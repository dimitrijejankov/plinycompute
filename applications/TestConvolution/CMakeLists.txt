# create the target
add_pdb_integration_test(TestConvolution)
target_link_libraries(TestConvolution ${MKL_LIBRARIES})

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestConvolution)

find_package(Torch REQUIRED)

# process lib
add_library(ProcessLib ${PROJECT_SOURCE_DIR}/applications/TestConvolution/sharedLibraries/source/Process.cc)
target_link_libraries(ProcessLib "${TORCH_LIBRARIES}")
set_property(TARGET ProcessLib PROPERTY CXX_STANDARD 14)

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestConvolution/sharedLibraries/source/*.cc")
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ProcessLib)
    add_dependencies(shared-libraries ${fileName})
endforeach()

# add build dependencies to shared libraries it uses
add_dependencies(TestConvolution MatrixBlock3D)
target_link_libraries(TestConvolution ProcessLib)

add_dependencies(TestConvolution Matrix3DScanner)
add_dependencies(TestConvolution Matrix3DWriter)
add_dependencies(TestConvolution MatrixBlock3D)
add_dependencies(TestConvolution MatrixBlockData3D)
add_dependencies(TestConvolution MatrixBlockMeta3D)
add_dependencies(TestConvolution MatrixConv3DJoin)
add_dependencies(TestConvolution MatrixConvResult)