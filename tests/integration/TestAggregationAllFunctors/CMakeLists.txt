# create the target
add_pdb_integration_test(TestAggregationAllFunctors)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestAggregationAllFunctors)

# add build dependencies to shared libraries it uses
add_dependencies(TestAggregationAllFunctors ScanEmployeeSet)
add_dependencies(TestAggregationAllFunctors DepartmentMax)
add_dependencies(TestAggregationAllFunctors MaxEmployeeAgg)
add_dependencies(TestAggregationAllFunctors WriteDepartmentMaxSet)

