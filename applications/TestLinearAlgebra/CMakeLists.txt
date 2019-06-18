


###
### linear algebra shared objects
###

add_custom_target(la-shared-libraries)

include_directories("${PROJECT_SOURCE_DIR}/applications/TestLinearAlgebra/sharedLibraries/headers")
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestLinearAlgebra/sharedLibraries/source/*.cc")
# compile all the shared library
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${GSL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
    add_dependencies(la-shared-libraries ${fileName})
endforeach()


###
### linear algebra dsl interpreter
###

include_directories("${PROJECT_SOURCE_DIR}/applications/TestLinearAlgebra/linearAlgebraDSL/headers")
file(GLOB LINEAR_ALGEBRA_DSL_SOURCE "${PROJECT_SOURCE_DIR}/applications/TestLinearAlgebra/linearAlgebraDSL/source/*.cc")

FIND_PACKAGE(FLEX REQUIRED)
SET(LinearAlgebraDSLFlexOutput ${CMAKE_SOURCE_DIR}/applications/TestLinearAlgebra/linearAlgebraDSL/source/LALexer.c)
IF (FLEX_FOUND)
    # linear algebra dsl lexer commands
    ADD_CUSTOM_COMMAND(
            OUTPUT ${LinearAlgebraDSLFlexOutput}
            COMMAND ${FLEX_EXECUTABLE}
            --outfile=${LinearAlgebraDSLFlexOutput}
            ${CMAKE_SOURCE_DIR}/applications/TestLinearAlgebra/linearAlgebraDSL/source/LALexer.l
            COMMENT "Generating Linear Algebra DSL Lexer!"
            BUILD_BYPRODUCTS ${LinearAlgebraDSLFlexOutput}
    )
ENDIF ()

FIND_PACKAGE(BISON REQUIRED)
SET(LinearAlgebraDSLParserOutput ${CMAKE_SOURCE_DIR}/applications/TestLinearAlgebra/linearAlgebraDSL/source/LAParser.c)
IF (BISON_FOUND)
    # linear algebra dsl parser commands
    ADD_CUSTOM_COMMAND(
            OUTPUT ${LinearAlgebraDSLParserOutput}
            COMMAND ${BISON_EXECUTABLE}
            --output=${LinearAlgebraDSLParserOutput}
            ${CMAKE_SOURCE_DIR}/applications/TestLinearAlgebra/linearAlgebraDSL/source/LAParser.y
            COMMENT "Generating Linear Algebra DSL Parser"
            BUILD_BYPRODUCTS ${LinearAlgebraDSLParserOutput}
    )
ENDIF ()

add_library(linear-algebra OBJECT ${LINEAR_ALGEBRA_DSL_SOURCE})
add_library(linear-algebra-parser OBJECT ${LinearAlgebraDSLFlexOutput} ${LinearAlgebraDSLParserOutput} ${BUILT_IN_OBJECT_TYPE_ID} ${BUILT_IN_PDB_OBJECTS})

# add a build dependency to build-la-tests target
add_dependencies(build-la-tests TestLA_unit01_Transpose)
add_dependencies(build-la-tests TestLA_unit02_Add)
add_dependencies(build-la-tests TestLA_unit03_Subtract)
add_dependencies(build-la-tests TestLA_unit04_Multiply)
add_dependencies(build-la-tests TestLA_unit05_MaxElement)
add_dependencies(build-la-tests TestLA_unit06_MinElement)
add_dependencies(build-la-tests TestLA_unit07_ElementwiseMultiply)
add_dependencies(build-la-tests TestLA_unit08_RowMax)
add_dependencies(build-la-tests TestLA_unit09_RowMin)
add_dependencies(build-la-tests TestLA_unit10_ColMax)
add_dependencies(build-la-tests TestLA_unit11_ColMin)
add_dependencies(build-la-tests TestLA_unit12_DuplicateRow)
add_dependencies(build-la-tests TestLA_unit13_DuplicateCol)
add_dependencies(build-la-tests TestLA_unit14_Inverse)
add_dependencies(build-la-tests TestLA_unit15_RowSum)
add_dependencies(build-la-tests TestLA_unit16_ColSum)
add_dependencies(build-la-tests TestLA_unit17_TransposeMultiply)
add_dependencies(build-la-tests TestLA_unit18_TransposeMultiply_Gram)


# adds linear algebra unit test to CMAKE
function(add_pdb_linear_algebra_unit_test test-name)
    # get the directory
    get_filename_component(test-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

    # create the target
    add_executable(${test-name} "${test-path}/tests/${test-name}.cc"
            $<TARGET_OBJECTS:logical-plan-parser>)

    # link it to the required libraries
    target_link_libraries(${test-name} pdb-tests-common)
    add_dependencies(${test-name} la-shared-libraries)
endfunction(add_pdb_linear_algebra_unit_test)


# adds linear algebra application to CMAKE
function(add_pdb_linear_algebra_application test-name)
    # get the directory
    get_filename_component(test-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

    # create the target
    add_executable(${test-name} "${test-path}/tests/${test-name}.cc"
            $<TARGET_OBJECTS:logical-plan-parser>
            $<TARGET_OBJECTS:linear-algebra>
            $<TARGET_OBJECTS:linear-algebra-parser>)

    # link it to the required libraries
    target_link_libraries(${test-name} pdb-tests-common)
    add_dependencies(${test-name} la-shared-libraries)
endfunction(add_pdb_linear_algebra_application)


###
### Linear algebra unit Tests
###

add_pdb_linear_algebra_unit_test(TestLA_unit01_Transpose)
add_pdb_linear_algebra_unit_test(TestLA_unit02_Add)
add_pdb_linear_algebra_unit_test(TestLA_unit03_Subtract)
add_pdb_linear_algebra_unit_test(TestLA_unit04_Multiply)
add_pdb_linear_algebra_unit_test(TestLA_unit05_MaxElement)
add_pdb_linear_algebra_unit_test(TestLA_unit06_MinElement)
add_pdb_linear_algebra_unit_test(TestLA_unit07_ElementwiseMultiply)
add_pdb_linear_algebra_unit_test(TestLA_unit08_RowMax)
add_pdb_linear_algebra_unit_test(TestLA_unit09_RowMin)
add_pdb_linear_algebra_unit_test(TestLA_unit10_ColMax)
add_pdb_linear_algebra_unit_test(TestLA_unit11_ColMin)
add_pdb_linear_algebra_unit_test(TestLA_unit12_DuplicateRow)
add_pdb_linear_algebra_unit_test(TestLA_unit13_DuplicateCol)
add_pdb_linear_algebra_unit_test(TestLA_unit14_Inverse)
add_pdb_linear_algebra_unit_test(TestLA_unit15_RowSum)
add_pdb_linear_algebra_unit_test(TestLA_unit16_ColSum)
add_pdb_linear_algebra_unit_test(TestLA_unit17_TransposeMultiply)
add_pdb_linear_algebra_unit_test(TestLA_unit18_TransposeMultiply_Gram)

###
###  Compile the actual interpreter
###

# Test if the parser works.
add_pdb_linear_algebra_application(TestLA_Parser)
# This is the actual interpreter that handles the .pdml code
add_pdb_linear_algebra_application(TestLA_Instance)