# create the target
add_pdb_integration_test(TestAggregation)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestAggregation)

# add build dependencies to shared libraries it uses
add_dependencies(TestAggregation ScanSupervisorSet)
add_dependencies(TestAggregation SillyQuery)
add_dependencies(TestAggregation SillyAgg)
add_dependencies(TestAggregation FinalQuery)
add_dependencies(TestAggregation WriteSalaries)