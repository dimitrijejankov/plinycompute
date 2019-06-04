//
// Created by vicram on 6/3/19.
//

/*
 * This benchmark is testing the Aggregation in the case where all the classes are balanced;
 * i.e. every department has the same number of Employees. This isn't an entirely realistic
 * case, but it's a good starting point and should test the "baseline" performance of the
 * Aggregation.
 */

#include <cstdlib>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <iomanip>

#include <benchmark/benchmark.h>

#include <PDBClient.h>
#include <Handle.h>
#include <PDBVector.h>
#include <InterfaceFunctions.h>
#include <Employee.h>
#include <DepartmentTotal.h>
#include <SillyAgg.h>
#include <ScanEmployeeSet.h>
#include <WriteDepartmentTotal.h>

// Global constants (these are the input params to the benchmark)
static const std::string managerHostname("localhost");
static const int managerPort = 8108;
static const bool validate = true;


static void BenchAggregationBalanced(benchmark::State& state) {
  int64_t numDepartments = state.range(0);
  int64_t totalNumEmployees = state.range(1);
  /// First is the setup code (won't be timed)

  // Set up client

  pdb::PDBClient pdbClient(managerPort, managerHostname);

  // Register shared libs
  pdbClient.registerType("libraries/libSillyAgg.so");
  pdbClient.registerType("libraries/libScanEmployeeSet.so");
  pdbClient.registerType("libraries/libWriteDepartmentTotal.so");

  // Create DB and sets
  const std::string dbname("BenchAggregationBalancedDB");
  const std::string inputSet("BenchAggregationBalancedInputSet");
  const std::string outputSet("BenchAggregationBalancedOutputSet");
  pdbClient.createDatabase(dbname);
  pdbClient.createSet<pdb::Employee>(dbname, inputSet);
  pdbClient.createSet<pdb::DepartmentTotal>(dbname, outputSet);

  // Fill input set with Employees

  // First, create a C++ vector of department names.
  std::vector<std::string> departments;

  for (int i = 0; i < numDepartments; ++i) {
    // This code is needed to ensure that all the strings are the same length, which
    // simplifies reasoning about performance.
    // https://stackoverflow.com/a/225435
    std::stringstream ss;
    ss << std::setw(15) << std::setfill('0') << i; // Pad with leading zeros to make string of length 15
    std::string s = ss.str();
    departments.push_back(s);
  }

  // Next, allocate the Employees and send them to storage.
  pdb::makeObjectAllocatorBlock(1024 * 1024, true); // 1 MiB
  int employeesSent = 0;
  while(employeesSent < totalNumEmployees) {
    pdb::Handle<pdb::Vector<pdb::Handle<pdb::Employee>>> toSend =
        pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Employee>>>();
    try {
      for(; employeesSent < totalNumEmployees; ++employeesSent) {
        // Assign the employees to departments in a round-robin manner.
        int departmentId = employeesSent % numDepartments;
        pdb::Handle<pdb::Employee> emp =
            pdb::makeObject<pdb::Employee>("Frank", 5, departments[departmentId], 1.0);
        // Name and age are arbitrary (not used), salary is 1
        toSend->push_back(emp);
      }
      // If we make it to the end without an exception, send and exit the loop
      pdbClient.sendData<pdb::Employee>(dbname, inputSet, toSend);
    } catch (pdb::NotEnoughSpace& e) {
      pdbClient.sendData<pdb::Employee>(dbname, inputSet, toSend);
      pdb::makeObjectAllocatorBlock(1024 * 1024, true);
    }
  }
//  pdbClient.flushData();

  /// Finally, run the work loop
  for(auto _ : state) {
    state.PauseTiming();
    // We don't want to measure setup time here, only time to execute the computations
    pdb::makeObjectAllocatorBlock(1024 * 1024, true);

    pdb::String myTCAPString = ""; // TODO

    pdb::Handle<pdb::Vector<pdb::Handle<pdb::Computation>>> myComputations =
        pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Computation>>>();

    pdb::Handle<pdb::Computation> scanComp =
        pdb::makeObject<ScanEmployeeSet>(dbname, inputSet);
    pdb::Handle<pdb::Computation> aggComp =
        pdb::makeObject<pdb::SillyAgg>();
    pdb::Handle<pdb::Computation> writeComp =
        pdb::makeObject<pdb::WriteDepartmentTotal>(dbname, outputSet);

    aggComp->setInput(scanComp);
    writeComp->setInput(aggComp);

    myComputations->push_back(scanComp);
    myComputations->push_back(aggComp);
    myComputations->push_back(writeComp);

    state.ResumeTiming();
    pdbClient.executeComputations(myComputations, myTCAPString);
    state.PauseTiming();

    if (validate) {
      // Because the employees were distributed round-robin to departments,
      // the number of employees in a department (which is equal to that department's
      // salary, because all employees have a salary of 1) can vary by at most 1
      // among the departments. Denoting the lower total as 'lowerBound', we can compute
      // how many departments should have a total of lowerBound and how many should have
      // a total of lowerBound + 1.
      int lowerBound = totalNumEmployees / numDepartments;
      int upperCount = totalNumEmployees % numDepartments; // # of depts which should have a total of lowerBound+1
      int lowerCount = numDepartments - upperCount; // # of depts which should have a total of lowerBound
      assert((lowerCount*lowerBound + upperCount*(lowerBound + 1)) == totalNumEmployees);

      bool everythingOk = true;
      auto iter = pdbClient.getSetIterator<pdb::DepartmentTotal>(dbname, outputSet);
      while (iter->hasNextRecord()) {
        pdb::Handle<pdb::DepartmentTotal> deptTotal = iter->getNextRecord();
        double total = *(deptTotal->getTotSales());
        long roundedTotal = std::lround(total);
        if (roundedTotal == lowerBound) {
          --lowerCount;
        } else if (roundedTotal == (lowerBound + 1)) {
          --upperCount;
        } else {
          std::stringstream errMsgStream;
          errMsgStream << "A department had an illegal total: instead of " << lowerBound << " or " << (lowerBound+1) <<
                       " the total was " << roundedTotal << "!";
          state.SkipWithError(errMsgStream.str().c_str());
          everythingOk = false;
          break;
        }
      }
      if (!everythingOk) {
        break;
      }
      if ((lowerCount != 0) || (upperCount != 0)) {
        std::stringstream errMsgStream;
        errMsgStream << "Incorrect counts for department totals: lowerCount = " << lowerCount <<
                     " and upperCount = " << upperCount << "!";
        state.SkipWithError(errMsgStream.str().c_str());
        break;
      }
    }

    /// Regardless of whether we validate, we need to clear the output set for the next iteration
    pdbClient.removeSet(dbname, outputSet); // TODO how to reset without being able to remove a set?
    pdbClient.createSet<pdb::DepartmentTotal>(dbname, outputSet);

    state.ResumeTiming();
  }

  /// Finally, shut down server.
  pdbClient.shutDownServer();
}


BENCHMARK(BenchAggregationBalanced)
    ->UseRealTime() // Wall-clock time better captures the network latency, compared to CPU time
    ->Args({100, 10000}); // {numDepartments, totalNumEmployees}

BENCHMARK_MAIN();