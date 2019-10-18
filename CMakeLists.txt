cmake_minimum_required(VERSION 3.5.1)

# set clang as the default compiler
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
set(CMAKE_CUDA_FLAGS "-arch=sm_75")

set (BOOST_ROOT "/home/yuxin/snap/boost")
set (Boost_INCLUDE_DIR "/home/yuxin/snap/boost/include")
set (Boost_LIBRARY_DIRS "/home/yuxin/snap/boost/lib")

project(pdb)

option(USE_CUDA "Use CUDA" ON)

# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/libraries")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/tools/cmake")
set(CTEST_BINARY_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

# set the standard
set(CMAKE_CXX_STANDARD 14)

# check the operating system and add the appropriate flags
set(CMAKE_CXX_FLAGS "-fPIC -Wno-deprecated-declarations")

# if we are compiling for debug add the debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)

    # this is to make the standard lib object readable in the debugger, this will require a boost library compiled for that
    # you can provide it with -DUSE_CUSTOM_BOOST=ON and -DCUSTOM_BOOST_LOCATION=/path/to/boost
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_BUFFER_MANAGER -g")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -ldl" )
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -ldl")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic -ldl")

# if we are compiling for release
elseif(CMAKE_BUILD_TYPE MATCHES Release)

    # do -O3 and set the instruction set to the native one for this machine
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -ldl" )
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -ldl")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic -ldl")

# if we are compiling the unit tests
elseif(CMAKE_BUILD_TYPE MATCHES UNIT_TESTS)

    # the unit test have the same stuff enabled as the regular tests
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_BUFFER_MANAGER -g3")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic")

    # in the unit tests the minimum page size will be set to 8 bytes so that they can execute fast
    add_definitions(-DMIN_PAGE_SIZE=8)

endif ()

#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread -fPIE -g")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=thread -fPIE -g")

# set the definitions for the compiler
ADD_DEFINITIONS(-DINITIALIZE_ALLOCATOR_BLOCK)
ADD_DEFINITIONS(-DENABLE_SHALLOW_COPY)
ADD_DEFINITIONS(-DDEFAULT_BATCH_SIZE=100)
ADD_DEFINITIONS(-DREMOVE_SET_WITH_EVICTION)
ADD_DEFINITIONS(-DAUTO_TUNING)
ADD_DEFINITIONS(-DENABLE_LARGE_GRAPH)
ADD_DEFINITIONS(-DJOIN_HASH_TABLE_SIZE_RATIO=1.5)
ADD_DEFINITIONS(-DEVICT_STOP_THRESHOLD=0.95)
ADD_DEFINITIONS(-DEVICT_STOP_THRESHOLD=0.95)
ADD_DEFINITIONS(-DENABLE_COMPRESSION)
ADD_DEFINITIONS(-DPROFILING)

# used to enable debug
if (USE_DEBUG)
   message("PDB_DEBUG is ON")
   #ADD_DEFINITIONS(-DPDB_DEBUG)
elseif (NOT USE_DEBUG)
   REMOVE_DEFINITIONS(-DPDB_DEBUG)
   message("PDB_DEBUG is OFF")
endif (USE_DEBUG)

# set the directories with the common header files
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/bufferManager/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/builtInPDBObjects/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/client/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/communication/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/catalog/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/computations/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/computationServer/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/computationServer/headers/physicalOptimizer")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/configuration/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/distributedStorage/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/executionServer/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/executionServer/headers/physicalAlgorithms")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/lambdas/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/lambdas/headers/lambdas")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/logicalPlan/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/objectModel/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/pipeline/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/pipeline/headers/sinks")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/pipeline/headers/sources")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/pipeline/headers/executors")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/pipeline/headers/pipeline")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/pipeline/headers/processors")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/server/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/serverFunctionalities/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/sharedLibraries/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/storageManager/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/work/headers")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/cuda/headers")

# add the third party libraries as includes
include_directories("${PROJECT_SOURCE_DIR}/third-party/mustache/headers")
include_directories("${PROJECT_SOURCE_DIR}/third-party/qunit/headers")
include_directories("${PROJECT_SOURCE_DIR}/third-party/sqlite/headers")
include_directories("${PROJECT_SOURCE_DIR}/third-party/concurent_queue/headers")


# specify the source files
file(GLOB SERVER_FUNCTIONALITIES_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/serverFunctionalities/source/*.cc")
file(GLOB COMMUNICATION_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/communication/source/*.cc")
file(GLOB CATALOG_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/catalog/source/*.cc")
file(GLOB DISTRIBUTED_STORAGE_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/distributedStorage/source/*.cc")
file(GLOB SERVER_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/server/source/*.cc")
file(GLOB OBJECT_MODEL_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/objectModel/source/*.cc")
file(GLOB WORK_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/work/source/*.cc")
file(GLOB STORAGE_MANAGER_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/storageManager/source/*.cc")
file(GLOB BUILT_OBJECTS_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/builtInPDBObjects/source/*.cc")

file(GLOB PIPELINE_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/pipeline/source/*.cc" "${PROJECT_SOURCE_DIR}/pdb/src/pipeline/source/pipeline/*.cc")

file(GLOB DISTRIBUTION_MANAGER_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/distributionManager/source/*.cc")
file(GLOB LOGICAL_PLAN_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/logicalPlan/source/*.cc")
file(GLOB CLIENT_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/client/source/*.cc")
file(GLOB BUFFER_MANAGER_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/bufferManager/source/*.cc")
file(GLOB COMPUTATION_SERVER_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/computationServer/sources/*.cc"
                                    "${PROJECT_SOURCE_DIR}/pdb/src/computationServer/sources/physicalOptimizer/*.cc")
file(GLOB EXECUTION_SERVER_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/executionServer/sources/*.cc"
                                  "${PROJECT_SOURCE_DIR}/pdb/src/executionServer/sources/physicalAlgorithms/*.cc")

file(GLOB GPU_SOURCE "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cu")

set(SOURCE_FILES_SQL_LITE ${PROJECT_SOURCE_DIR}/third-party/sqlite/source/sqlite3.c)

# specify the hearder files
file(GLOB PDB_HEADERS "${PROJECT_SOURCE_DIR}/pdb/src/*/headers/*.h"
                      "${PROJECT_SOURCE_DIR}/pdb/src/*/headers/**/*.h"
                      "${PROJECT_SOURCE_DIR}/pdb/src/*/headers/**/**/*.h")


# specify the third party headers
file(GLOB THIRD_PARTY_HEADERS "${PROJECT_SOURCE_DIR}/third-party/*/headers/*.h"
                              "${PROJECT_SOURCE_DIR}/third-party/*/headers/**/*.h")

add_custom_target(HEADER_ONLY_TARGET SOURCES "${PDB_HEADERS}" "${THIRD_PARTY_HEADERS}")

# generate lexers
FIND_PACKAGE(FLEX REQUIRED)

# set the output paths
set(LogicalPlanFlexOutput ${CMAKE_SOURCE_DIR}/pdb/src/logicalPlan/source/Lexer.c)

# if you want to use a custom version of boost set -DUSE_CUSTOM_BOOST=ON and -DCUSTOM_BOOST_LOCATION=/path/to/boost
IF(USE_CUSTOM_BOOST)

   # set the custom location
   set(Boost_INCLUDE_DIRS ${CUSTOM_BOOST_LOCATION}/include)
   set(Boost_LIBRARY_DIRS ${CUSTOM_BOOST_LOCATION}/lib)
ENDIF()

# find the boost package
set(Boost_NO_BOOST_CMAKE ON)
FIND_PACKAGE(Boost COMPONENTS program_options filesystem system stacktrace_basic REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

# setup the commands
IF (FLEX_FOUND)
    # logical plan lexer commands
    ADD_CUSTOM_COMMAND(
            OUTPUT ${LogicalPlanFlexOutput}
            COMMAND ${FLEX_EXECUTABLE}
            --outfile=${LogicalPlanFlexOutput}
            ${CMAKE_SOURCE_DIR}/pdb/src/logicalPlan/source/Lexer.l
            COMMENT "Generating Logical Plan Lexer!"
            BUILD_BYPRODUCTS ${LogicalPlanFlexOutput}
    )
ENDIF ()

# generate the parser
FIND_PACKAGE(BISON REQUIRED)

# set the output paths
set(LogicalPlanParserOutput ${CMAKE_SOURCE_DIR}/pdb/src/logicalPlan/source/Parser.c)

# setup the commands
IF (BISON_FOUND)
    # logical plan parser commands
    ADD_CUSTOM_COMMAND(
            OUTPUT ${LogicalPlanParserOutput}
            COMMAND ${BISON_EXECUTABLE}
            --output=${LogicalPlanParserOutput}
            ${CMAKE_SOURCE_DIR}/pdb/src/logicalPlan/source/Parser.y
            COMMENT "Generating Logical Plan Parser"
            BUILD_BYPRODUCTS ${LogicalPlanParserOutput}
    )
ENDIF ()

# find gsl
FIND_PACKAGE(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})

#find snappy
FIND_PACKAGE(Snappy REQUIRED)

# the files generated from the type codes
set(BUILT_IN_OBJECT_TYPE_ID        ${CMAKE_SOURCE_DIR}/pdb/src/objectModel/headers/BuiltInObjectTypeIDs.h)
set(BUILT_IN_PDB_OBJECTS           ${CMAKE_SOURCE_DIR}/pdb/src/objectModel/headers/BuiltinPDBObjects.h)
set(BUILT_IN_PDB_OBJECTS_GENERATED ${CMAKE_SOURCE_DIR}/pdb/src/objectModel/headers/BuiltinPDBObjects.h
                                   ${CMAKE_SOURCE_DIR}/pdb/src/objectModel/headers/BuiltinPDBObjects.cc)

# mark the files as generated
SET_SOURCE_FILES_PROPERTIES(${BUILT_IN_PDB_OBJECTS_GENERATED} PROPERTIES GENERATED TRUE)
SET_SOURCE_FILES_PROPERTIES(${BUILT_IN_OBJECT_TYPE_ID} PROPERTIES GENERATED TRUE)

# add the commands to generate the type ids
add_custom_command (
        OUTPUT  object-type-ids-command
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/type-id-data
        COMMAND "${CMAKE_SOURCE_DIR}/scripts/internal/generateTypeCodes.py" BuiltInObjectTypeIDs ${CMAKE_CURRENT_BINARY_DIR}/type-id-data
        COMMAND "${CMAKE_SOURCE_DIR}/scripts/internal/generateTypeCodes.py" BuiltinPDBObjects ${CMAKE_CURRENT_BINARY_DIR}/type-id-data
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/pdb"
        COMMENT "Generating type codes!"
)

# create a target to run the command and make every other target depend on it so it gets to run first
add_custom_target(object-type-ids ALL DEPENDS object-type-ids-command)

# include the directory we just created in the clean target
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/type-id-data)

# find the threads library
find_package(Threads)

# setup the libraries
add_library(sql-lite OBJECT ${SOURCE_FILES_SQL_LITE})
add_library(server-functionalities OBJECT ${SERVER_FUNCTIONALITIES_SOURCE})
add_library(communication OBJECT ${COMMUNICATION_SOURCE})
add_library(catalog OBJECT ${CATALOG_SOURCE})
add_library(distributed-storage OBJECT ${DISTRIBUTED_STORAGE_SOURCE})
add_library(server OBJECT ${SERVER_SOURCE})
add_library(work OBJECT ${WORK_SOURCE})
add_library(storageManager OBJECT ${STORAGE_MANAGER_SOURCE})
add_library(pipeline OBJECT ${PIPELINE_SOURCE})
add_library(built-in-objects OBJECT ${BUILT_OBJECTS_SOURCE})
add_library(logical-plan OBJECT ${LOGICAL_PLAN_SOURCE})
add_library(client OBJECT ${CLIENT_SOURCE})
add_library(object-model OBJECT ${OBJECT_MODEL_SOURCE})
add_library(buffer-manager OBJECT ${BUFFER_MANAGER_SOURCE})
add_library(computation-server OBJECT ${COMPUTATION_SERVER_SOURCE})
add_library(execution-server OBJECT ${EXECUTION_SERVER_SOURCE})

# add the dependencies to the object type generationg
add_dependencies(server-functionalities object-type-ids)
add_dependencies(communication object-type-ids)
add_dependencies(object-model object-type-ids)
add_dependencies(client object-type-ids)
add_dependencies(logical-plan object-type-ids)
add_dependencies(server object-type-ids)
add_dependencies(work object-type-ids)
add_dependencies(execution-server object-type-ids)

# used to build the pdb-manager and pdb-worker
add_library(pdb-server-common $<TARGET_OBJECTS:catalog>
        $<TARGET_OBJECTS:distributed-storage>
        $<TARGET_OBJECTS:communication>
        $<TARGET_OBJECTS:logical-plan>
        $<TARGET_OBJECTS:server>
        $<TARGET_OBJECTS:client>
        $<TARGET_OBJECTS:object-model>
        $<TARGET_OBJECTS:sql-lite>
        $<TARGET_OBJECTS:server-functionalities>
        $<TARGET_OBJECTS:storageManager>
        $<TARGET_OBJECTS:pipeline>
        $<TARGET_OBJECTS:built-in-objects>
        $<TARGET_OBJECTS:buffer-manager>
        $<TARGET_OBJECTS:computation-server>
        $<TARGET_OBJECTS:execution-server>
        $<TARGET_OBJECTS:work>)

# used to link the shared libraries that the shared librarires use
add_library(pdb-shared-common $<TARGET_OBJECTS:catalog>
        $<TARGET_OBJECTS:distributed-storage>
        $<TARGET_OBJECTS:communication>
        $<TARGET_OBJECTS:logical-plan>
        $<TARGET_OBJECTS:server>
        $<TARGET_OBJECTS:client>
        $<TARGET_OBJECTS:sql-lite>
        $<TARGET_OBJECTS:server-functionalities>
        $<TARGET_OBJECTS:storageManager>
        $<TARGET_OBJECTS:pipeline>
        $<TARGET_OBJECTS:buffer-manager>
        $<TARGET_OBJECTS:computation-server>
        $<TARGET_OBJECTS:execution-server>
        $<TARGET_OBJECTS:work>)

# used to link the shared libraries that the test use
add_library(pdb-tests-common $<TARGET_OBJECTS:catalog>
        $<TARGET_OBJECTS:distributed-storage>
        $<TARGET_OBJECTS:communication>
        $<TARGET_OBJECTS:logical-plan>
        $<TARGET_OBJECTS:server>
        $<TARGET_OBJECTS:client>
        $<TARGET_OBJECTS:object-model>
        $<TARGET_OBJECTS:sql-lite>
        $<TARGET_OBJECTS:server-functionalities>
        $<TARGET_OBJECTS:storageManager>
        $<TARGET_OBJECTS:pipeline>
        $<TARGET_OBJECTS:built-in-objects>
        $<TARGET_OBJECTS:buffer-manager>
        $<TARGET_OBJECTS:computation-server>
        $<TARGET_OBJECTS:execution-server>
        $<TARGET_OBJECTS:work>)

# link the dependent libraries so that they are made of the public interface
target_link_libraries(pdb-server-common PRIVATE ${SNAPPY_LIBRARY})
target_link_libraries(pdb-server-common PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(pdb-server-common PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(pdb-server-common PRIVATE ${Boost_LIBRARIES})

target_link_libraries(pdb-tests-common PRIVATE ${SNAPPY_LIBRARY})
target_link_libraries(pdb-tests-common PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(pdb-tests-common PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(pdb-tests-common PRIVATE ${Boost_LIBRARIES})

# form the parsers for the logical plan
add_library(logical-plan-parser OBJECT ${LogicalPlanFlexOutput} ${LogicalPlanParserOutput})


# Find CUDA and compile all the .cu to pdb-gpu
if (("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
        ("${CMAKE_VERSION}" VERSION_GREATER 3.8))
    # Modern CMake
    if (USE_CUDA)
        enable_language("CUDA")
        add_library(pdb-gpu SHARED ${GPU_SOURCE})
    endif ()
else ()
    # Old CMake
    if (USE_CUDA)
        find_package(CUDA REQUIRED)
        if (NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else ()
            message(STATUS "CUDA found")
            CUDA_ADD_LIBRARY(pdb-gpu ${GPU_SOURCE} SHARED)
        endif ()
    endif ()
endif ()

# check if the target is unit tests
if(CMAKE_BUILD_TYPE MATCHES UNIT_TESTS)

    # include the unit tests
    include(tests/unit/CMakeLists.txt)

    # include benchmarks that are in the benchmarks directory
    include(benchmarks/Benchmarks.cmake)

else()

    # set the pdb-worker build target
    set(SOURCE_FILES_PDB_CLUSTER ${PROJECT_SOURCE_DIR}/pdb/src/mainServer/source/NodeMain.cc $<TARGET_OBJECTS:logical-plan-parser>)
    add_executable(pdb-node ${SOURCE_FILES_PDB_CLUSTER})
    target_link_libraries(pdb-node pdb-server-common)
    target_link_libraries(pdb-node ${GSL_LIBRARIES})
    target_link_libraries(pdb-node ${Boost_LIBRARIES})
    add_dependencies(pdb-node pdb-gpu)
    target_link_libraries(pdb-node pdb-gpu)
    target_link_libraries(pdb-node "/usr/local/cuda-10.0/lib64/libcublas.so")

    # set dependencies for pdb-main
    add_custom_target(pdb-main)
    add_dependencies(pdb-main pdb-node)

    # compile all the objects
    file(GLOB files "${PROJECT_SOURCE_DIR}/pdb/src/sharedLibraries/source/*.cc")
    add_custom_target(shared-libraries)
    foreach(file ${files})
        get_filename_component(fileName "${file}" NAME_WE)
        add_library(${fileName} MODULE ${file})
        target_link_libraries(${fileName} pdb-shared-common)
        add_dependencies(shared-libraries ${fileName})
    endforeach()

    # include tests that are in tests directory
    include(tests/Tests.cmake)

    # include tests that are in applications directory
    include(applications/Applications.cmake)

endif()


