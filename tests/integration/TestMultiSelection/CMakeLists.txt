# create the target
add_pdb_integration_test(TestMultiSelection)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestMultiSelection)

# add build dependencies to shared libraries it uses
add_dependencies(TestMultiSelection ReadStringIntPair)
add_dependencies(TestMultiSelection StringIntPairMultiSelection)
add_dependencies(TestMultiSelection WriteStringIntPair)