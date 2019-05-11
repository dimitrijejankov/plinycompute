# create the target
add_pdb_integration_test(TestSelection)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestSelection)

# add build dependencies to shared libraries it uses
add_dependencies(TestSelection SharedEmployee)
add_dependencies(TestSelection EmployeeBuiltInIdentitySelection)
add_dependencies(TestSelection WriteBuiltinEmployeeSet)
add_dependencies(TestSelection ScanEmployeeSet)