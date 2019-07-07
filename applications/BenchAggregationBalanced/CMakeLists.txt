# create the target
add_pdb_application(BenchAggregationBalanced)
# Note: these shared libraries are in pdb/src/sharedLibraries
add_dependencies(BenchAggregationBalanced SillyAgg)
add_dependencies(BenchAggregationBalanced ScanEmployeeSet)
add_dependencies(BenchAggregationBalanced WriteDepartmentTotal)