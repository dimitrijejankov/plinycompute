# create the target
add_pdb_integration_test(TestJoin)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestJoin)

# add build dependencies to shared libraries it uses
add_dependencies(TestJoin ReadInt)
add_dependencies(TestJoin ReadStringIntPair)
add_dependencies(TestJoin SillyJoinIntString)
add_dependencies(TestJoin SillyWriteIntString)