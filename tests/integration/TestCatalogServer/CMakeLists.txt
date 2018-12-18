# create the target
add_pdb_integration_test(TestCatalogServer)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestCatalogServer)

# add build dependencies to shared libraries it uses
add_dependencies(TestCatalogServer DoubleVectorAggregation)
add_dependencies(TestCatalogServer SimpleJoin)
add_dependencies(TestCatalogServer ScanIntSet)
add_dependencies(TestCatalogServer ScanStringIntPairSet)
add_dependencies(TestCatalogServer ScanStringSet)
add_dependencies(TestCatalogServer WriteStringSet)
