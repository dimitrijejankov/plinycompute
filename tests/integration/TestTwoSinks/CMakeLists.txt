# create the target
add_pdb_integration_test(TestTwoSinks)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestTwoSinks)

# add build dependencies to shared libraries it uses
add_dependencies(TestTwoSinks ScanSupervisorSet)
add_dependencies(TestTwoSinks SillyQuery)
add_dependencies(TestTwoSinks SillyAgg)
add_dependencies(TestTwoSinks FinalQuery)
add_dependencies(TestTwoSinks WriteSalaries)