# create the target
add_pdb_integration_test(TestDoubleUnion)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestDoubleUnion)

# add build dependencies to shared libraries it uses
add_dependencies(TestDoubleUnion ReadInt)
add_dependencies(TestDoubleUnion IntUnion)
add_dependencies(TestDoubleUnion IntWriter)