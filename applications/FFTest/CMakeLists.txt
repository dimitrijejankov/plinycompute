# create the target
add_pdb_integration_test(FFTest)
target_link_libraries(FFTest ${MKL_LIBRARIES})

# add a build dependency to build-tests target
add_dependencies(build-integration-tests FFTest)

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/FFTest/sharedLibraries/source/*.cc")
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${MKL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()

# add build dependencies to shared libraries it uses
add_dependencies(FFTest FFMatrixBlock)
add_dependencies(FFTest FFMatrixData)
add_dependencies(FFTest FFMatrixMeta)
add_dependencies(FFTest FFHiddenLayerJoin)
add_dependencies(FFTest FFInputLayerJoin)
add_dependencies(FFTest FFLayerAgg)
add_dependencies(FFTest FFMatrixScanner)
add_dependencies(FFTest FFMatrixWriter)
