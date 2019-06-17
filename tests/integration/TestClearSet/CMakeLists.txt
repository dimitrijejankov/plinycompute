# create the target
add_pdb_integration_test(TestClearSet)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestClearSet)

# add build dependencies to shared libraries it uses
add_dependencies(TestClearSet ScanSupervisorSet)
add_dependencies(TestClearSet SillyQuery)
add_dependencies(TestClearSet SillyAgg)
add_dependencies(TestClearSet FinalQuery)
add_dependencies(TestClearSet WriteSalaries)