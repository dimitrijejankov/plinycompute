# create the target
add_pdb_integration_test(TestCartasianJoin)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestCartasianJoin)

# add build dependencies to shared libraries it uses
add_dependencies(TestCartasianJoin ReadInt)
add_dependencies(TestCartasianJoin ReadStringIntPair)
add_dependencies(TestCartasianJoin SillyCartasianJoinIntString)
add_dependencies(TestCartasianJoin SillyWriteIntString)