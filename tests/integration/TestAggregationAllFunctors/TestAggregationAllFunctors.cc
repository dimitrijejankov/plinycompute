//
// Created by vicram on 6/24/19.
//

#include <string>
#include <iostream>
#include <cstdlib>

#include <PDBClient.h>
#include <MaxEmployeeAgg.h>
#include <WriteDepartmentMaxSet.h>
#include "ScanEmployeeSet.h"

using namespace pdb;

int main(int argc, char* argv[]) {
  const size_t blockSize = 64;

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libScanEmployeeSet.so");
  pdbClient.registerType("libraries/libDepartmentMax.so");
  pdbClient.registerType("libraries/libMaxEmployeeAgg.so");
  pdbClient.registerType("libraries/libWriteDepartmentMaxSet.so");

  /// 2. Create the sets

  // now, create a new database
  std::string dbname("db");
  pdbClient.createDatabase(dbname);

  // now, create the input and output sets
  std::string inputname("inputset");
  std::string outputname("outputset");
  pdbClient.createSet<Employee>(dbname, inputname);
  pdbClient.createSet<int>(dbname, outputname);

  /// 3. Fill in the data

  {
    // We just use a single allocation block, so it has to be big enough
    const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};
    try {
      pdb::Handle<pdb::Vector<pdb::Handle<pdb::Employee>>> toSend = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Employee>>>();
      for (int dep = 1; dep <= 3; ++dep) {
        std::string departmentName = std::to_string(dep);
        for (int salary = 0; salary <= 5*dep; ++salary) {
          Handle<Employee> emp = makeObject<Employee>("Frank", 20, departmentName, (double)salary);
          toSend->push_back(emp);
        }

      }
      pdbClient.sendData<Employee>(dbname, inputname, toSend);
    } catch (NotEnoughSpace &e) {
      std::cout << "Allocation block is not big enough. Please increase the size of the allocation block." << std::endl;
      std::cout << "Exiting now." << std::endl;
      exit(1);
    }

  }

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  Handle<Computation> scanComp = makeObject<ScanEmployeeSet>(dbname, inputname);
  Handle<Computation> aggComp = makeObject<MaxEmployeeAgg>();
  Handle<Computation> writeComp = makeObject<WriteDepartmentMaxSet>(dbname, outputname);

  aggComp->setInput(scanComp);
  writeComp->setInput(aggComp);

  pdbClient.executeComputations({writeComp});

  /// 5. Iterate over set and print out the results

  std::cout << "Now printing out results. They should be:" << std::endl;
  std::cout << "Department 1: Max salary 5;    Department 2: Max salary 10;    Department 3: Max salary 15" << std::endl;

  // grab the iterator
  auto it = pdbClient.getSetIterator<DepartmentMax>(dbname, outputname);
  while(it->hasNextRecord()) {
    Handle<DepartmentMax> depMax = it->getNextRecord();

    std::cout << "Department: " << depMax->departmentName << "\tMax salary: " << depMax->max << std::endl << std::endl;
  }
  pdbClient.shutDownServer();

  return 0;
}