# create the target
add_pdb_integration_test(TestAggregationAllFunctors)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestAggregationAllFunctors)