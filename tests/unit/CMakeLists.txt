# include the google test
include(GoogleTest)
find_package(GTest REQUIRED)

#include pthreads
find_package(Threads)

# get the current directory
get_filename_component(unit-test-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# compile all the objects
file(GLOB files "${unit-test-path}/*.cc")

# sorts files alphabetically because some tests require
# files created in previous tests
list(SORT files)
add_custom_target(unit-tests)
foreach(file ${files})

    # grab the name of the test without the extension
    get_filename_component(fileName "${file}" NAME_WE)

    # create the test executable
    add_executable(${fileName} ${file}
                               $<TARGET_OBJECTS:logical-plan-parser>
                               $<TARGET_OBJECTS:client>)

    # link the stuff we need
    target_link_libraries(${fileName} pdb-tests-common)
    target_link_libraries(${fileName} ${GSL_LIBRARIES})
    target_link_libraries(${fileName} ${GTEST_BOTH_LIBRARIES} gmock ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${fileName} ${Boost_LIBRARIES})
    target_compile_definitions(${fileName} PRIVATE -DGTEST_LINKED_AS_SHARED_LIBRARY )

    # add the test as a dependency of the unit test target
    add_dependencies(unit-tests ${fileName})

endforeach()