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
                               $<TARGET_OBJECTS:linear-algebra-parser>
                               $<TARGET_OBJECTS:pdb-client>
                               $<TARGET_OBJECTS:linear-algebra-parser>)

    # link the stuff we need
    target_link_libraries(${fileName} pdb-tests-common)
    target_link_libraries(${fileName} ${GSL_LIBRARIES})
    target_link_libraries(${fileName} gtest_main gtest ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${fileName} gmock_main gmock ${CMAKE_THREAD_LIBS_INIT})

    # add the test as a dependency of the unit test target
    add_dependencies(unit-tests ${fileName})

endforeach()