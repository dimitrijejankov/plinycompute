# create the target
add_pdb_integration_test(TestUnion)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestUnion)

# add build dependencies to shared libraries it uses
add_dependencies(TestUnion ReadInt)
add_dependencies(TestUnion IntUnion)
add_dependencies(TestUnion IntWriter)