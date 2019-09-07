# create the target
add_pdb_integration_test(TestComplexUnion)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestComplexUnion)

# add build dependencies to shared libraries it uses
add_dependencies(TestComplexUnion ReadInt)
add_dependencies(TestComplexUnion IntUnion)
add_dependencies(TestComplexUnion IntWriter)
add_dependencies(TestComplexUnion UnionIntSelection)