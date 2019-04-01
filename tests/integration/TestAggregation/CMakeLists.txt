# create the target
add_pdb_integration_test(TestAggregation)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestAggregation)

# add build dependencies to shared libraries it uses
add_dependencies(TestAggregation SharedEmployee)
add_dependencies(TestAggregation EmployeeBuiltInIdentitySelection)
add_dependencies(TestAggregation WriteBuiltinEmployeeSet)
add_dependencies(TestAggregation ScanEmployeeSet)