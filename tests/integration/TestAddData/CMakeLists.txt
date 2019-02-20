# create the target
add_pdb_integration_test(TestAddData)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestAddData)

# add build dependencies to shared libraries it uses
add_dependencies(TestAddData SharedEmployee)
add_dependencies(TestAddData EmployeeBuiltInIdentitySelection)
add_dependencies(TestAddData WriteBuiltinEmployeeSet)
add_dependencies(TestAddData ScanEmployeeSet)