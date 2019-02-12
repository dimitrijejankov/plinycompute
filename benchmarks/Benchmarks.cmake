
# find threads and google benchmark
find_package(Threads REQUIRED)
find_package(GoogleBenchmark REQUIRED)

# get the current directory
get_filename_component(bench-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# compile all the objects
file(GLOB files "${bench-path}/*.cc")

# this one builds all the benchmarks
add_custom_target(build-benchmarks)

# create all the benchmarks
foreach(file ${files})

    # get the name of the benchmark
    get_filename_component(fileName "${file}" NAME_WE)

    # add the executable
    add_executable(${fileName} ${file} $<TARGET_OBJECTS:logical-plan-parser>
                                       $<TARGET_OBJECTS:client>)

    # link the libraries
    target_link_libraries(${fileName} ${benchmark_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${fileName} pdb-tests-common)
    target_link_libraries(${fileName} ${GSL_LIBRARIES})

    # add it to build benchmarks
    add_dependencies(build-benchmarks ${fileName})

endforeach()
