# create the target
add_pdb_integration_test(TestUnion)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestUnion)

# add build dependencies to shared libraries it uses
add_dependencies(TestUnion ScanSupervisorSet)
add_dependencies(TestUnion SillyQuery)
add_dependencies(TestUnion SillyAgg)
add_dependencies(TestUnion FinalQuery)
add_dependencies(TestUnion WriteSalaries)