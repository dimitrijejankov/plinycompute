//
// Created by vicram on 6/3/19.
//

/*
 * This benchmark is testing the Aggregation in the case where all the classes are balanced;
 * i.e. every department has the same number of Employees. This isn't an entirely realistic
 * case, but it's a good starting point and should test the "baseline" performance of the
 * Aggregation.
 *
 * We measure time using wall-clock time in order to accurately capture the latency of
 * the cluster.
 */

#include <cstdlib>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ratio>
#include <utility>
#include <thread>

#include <boost/program_options.hpp>

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

/**
 * Wraps the Linux system(3) built-in function: https://linux.die.net/man/3/system
 * If the command exits with nonzero status, this function terminates the program.
 * Otherwise it will return silently.
 * @param command Command to be executed
 */
static void systemwrap(const char *command) {
  int status = system(command);
  if (status) {
    std::cout << "Command '" << command << "' exited with status " << status << "!" << std::endl;
    std::cout << "Now exiting" << std::endl;
    exit(1);
  }
}


static std::unique_ptr<std::vector<std::chrono::duration<double, std::milli>>>
BenchAggregationBalanced(const int numDepartments,
                         const int totalNumEmployees,
                         const int numReps,
                         const bool validate,
                         const bool standalone) {
  assert(numDepartments > 0);
  assert(totalNumEmployees > 0);
  assert(numReps > 0);
  if (!standalone) {
    std::cout << "Only standalone mode works at this time. Please call with standalone=true" << std::endl;
    exit(1);
  }
  /// First is the setup code (won't be timed)
  // Start the server
  std::cout << "Starting manager node" << std::endl;
  if (!fork()) {
    systemwrap("./bin/pdb-node -m");
    exit(0);
  }
  // Wait 10 seconds
  std::this_thread::sleep_for(std::chrono::seconds(10));
  if (standalone) {
    std::cout << "Starting 1 worker node" << std::endl;
    if (!fork()) {
      systemwrap("./bin/pdb-node -p 8109 -r ./pdbRootW1");
      exit(0);
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }
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
  // Here we're allocating the vector that will be returned
  std::unique_ptr<std::vector<std::chrono::duration<double, std::milli>>>
      times = std::make_unique<std::vector<std::chrono::duration<double, std::milli>>>(numReps);
  std::cout << "Now starting the work loop!" << std::endl;
  for(int rep = 0; rep < numReps; ++rep) {
    // We don't want to measure setup time here, only time to execute the computations
    pdb::makeObjectAllocatorBlock(1024 * 1024, true);

    pdb::Handle<pdb::Computation> scanComp =
        pdb::makeObject<ScanEmployeeSet>(dbname, inputSet);
    pdb::Handle<pdb::Computation> aggComp =
        pdb::makeObject<pdb::SillyAgg>();
    pdb::Handle<pdb::Computation> writeComp =
        pdb::makeObject<pdb::WriteDepartmentTotal>(dbname, outputSet);

    aggComp->setInput(scanComp);
    writeComp->setInput(aggComp);

    // See example here of recording execution time:
    // https://en.cppreference.com/w/cpp/chrono/duration/duration_cast
    auto start = std::chrono::system_clock::now(); // system_clock is the "real" (wall-clock) system time
    pdbClient.executeComputations({writeComp});
    auto end = std::chrono::system_clock::now();

    // The type parameters here are: the numerical type used to store the
    // duration, and the period (in seconds) of this duration's unit.
    // Because std::milli represents 1/1000, this number is in units of
    // 1/1000 seconds, or milliseconds.
    // https://en.cppreference.com/w/cpp/chrono/duration
    std::chrono::duration<double, std::milli> millis = end - start;
    (*times)[rep] = millis;

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
        int roundedTotal = std::lround(total);
        if (roundedTotal == lowerBound) {
          --lowerCount;
        } else if (roundedTotal == (lowerBound + 1)) {
          --upperCount;
        } else {
          std::cout << "A department had an illegal total: instead of " << lowerBound << " or " << (lowerBound+1) <<
                    " the total was " << roundedTotal << "!" << std::endl;
          std::cout << "Exiting now" << std::endl;
          everythingOk = false;
          break;
        }
      }
      if (!everythingOk) {
        break;
      }
      if ((lowerCount != 0) || (upperCount != 0)) {
        std::cout << "Incorrect counts for department totals: lowerCount = " << lowerCount <<
                  " and upperCount = " << upperCount << "!" << std::endl;
        std:: cout << "Exiting now" << std::endl;
        break;
      }
    }

    /// Regardless of whether we validate, we need to clear the output set for the next iteration
    pdbClient.clearSet(dbname, outputSet);

  }

  /// Finally, shut down server.
  std::cout << "Finished the work loop. Now shutting down server." << std::endl;
  pdbClient.shutDownServer();
  return times;
}



int main(int argc, char *argv[]) {
  // Code for parsing program options is modified from here:
  // http://www.radmangames.com/programming/how-to-use-boost-program_options
//  try {
//    /** Define and parse the program options
//     */
//    namespace po = boost::program_options;
//    po::options_description desc("Options");
//    desc.add_options()
//        ("help,h", "Prints usage information")
//        ()
//  }
  int numReps = 5;
  bool validate = true;
  bool standalone = true;
  std::vector<std::pair<int, int>> column_values; // Contains pairs of (numDepartments, totalNumEmployees)
#define ADDPAIR(x,y) column_values.push_back(std::make_pair<int,int>((x), (y)))
  ADDPAIR(100, 10000);

  auto numcols = column_values.size();
  double table[numReps][numcols];
  // Each column in table holds numReps doubles, which are the times (in milliseconds)
  // of all the reps for that column's numDepartments and totalNumEmployees values.

  for(int col = 0; col < numcols; ++col) {
    std::pair<int,int> elem = column_values[col];
    auto vectorOfDurations = BenchAggregationBalanced(elem.first, elem.second, numReps, validate, standalone);
    for(int row = 0; row < numReps; ++row) {
      std::chrono::duration<double, std::milli> d = (*vectorOfDurations)[row];
      double ms = d.count();
      table[row][col] = ms;
    }
  }

  // Now printing the table of recorded times
  std::cout << "The table below is formatted as follows:" << std::endl;
  std::cout << "Each column is labeled with two numbers. The first number is "
            << "the number of departments, and the second number is the "
            << "total number of employees in all departments. Each department "
            << "has the same number of employees." << std::endl;
  std::cout << "The rows are the time, in milliseconds, that it took to run the "
            << "Aggregation. The same Aggregation is run num_rows times on the same "
            << "instance of PDB, and the times are listed in order from top to bottom." << std::endl;
  std::cout << std::endl;
  // Print header line
  std::cout << "idx";
  int colwidth = 30; // Width of each column, in characters
  for (auto elem : column_values) {
    std::cout <<  ", " << std::setw(30) << std::setfill(' ') << elem.first << "/" << elem.second;
  }
  std::cout << std::endl;
  // Print the numbers
  for (int row = 0; row < numReps; ++row) {
    std::cout << row+1;
    for (int col = 0; col < numcols; ++col) {
      std::cout << ", ";
      // The below line is modified from https://stackoverflow.com/q/49295185
      std::cout << std::scientific << std::setprecision(6) << std::setw(30) << std::setfill(' ') << table[row][col];
    }
    std::cout << std::endl;
  }
}