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
 *
 * It is important to note that the benchmarks are done sequentially on the same cluster
 * without restarting any of the nodes. This means that earlier tests could potentially
 * alter the performance of later tests.
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

// Global constants
static const std::string managerHostname("localhost");
static const int managerPort = 8108;

/**
 * This function creates a connection to the cluster, creates and sends some test
 * data, and runs the same Aggregation multiple times, returning the time taken for each
 * repetition.
 *
 * The Aggregation being run takes Employees and aggregates them by department. Each department
 * will have the same number of Employees. If the number of Employees is not divisible by the
 * number of departments, then some departments will have one more Employee than some other
 * departments.
 *
 * This function will also print out an estimate for the number of bytes per Employee. This estimate
 * is computed by counting up the number of allocation blocks created, and multiplying that by the
 * number of bytes per allocation block. Because the last allocation block is likely not entirely
 * full when it is sent, this is a slight overestimate. Therefore, you should only trust the estimate
 * if there are more than, say, 10 allocation blocks sent.
 *
 * @param numDepartments Number of departments to distribute the Employees among. This is equivalent to the
 * number of unique aggregation keys.
 * @param totalNumEmployees Total number of employees among all departments.
 * @param numReps Number of times to run the Aggregation query on this data size.
 * @param validate If this is true, then after each Aggregation query, this function will
 * iterate over every Department in the output set and ensure that the results are correct,
 * or otherwise print an error message and return immediately.
 * @return A vector containing the time duration it took to run each query, in seconds.
 * This only times the query itself; it does not time the creation of the input data set or
 * the validation step.
 */
static std::unique_ptr<std::vector<std::chrono::duration<double>>>
BenchAggregationBalanced(const int numDepartments,
                         const int totalNumEmployees,
                         const int numReps,
                         const bool validate) {
  assert(numDepartments > 0);
  assert(totalNumEmployees > 0);
  assert(numReps > 0);

  std::cout << "Now beginning benchmark of " << numReps << " reps for " << numDepartments << " departments and " << totalNumEmployees << " employees." << std::endl;

  /// First is the setup code (won't be timed)

  // Set up client

  pdb::PDBClient pdbClient(managerPort, managerHostname);

  // Register shared libs
  pdbClient.registerType("libraries/libSillyAgg.so");
  pdbClient.registerType("libraries/libScanEmployeeSet.so");
  pdbClient.registerType("libraries/libWriteDepartmentTotal.so");

  // Create DB and sets
  const std::string suffix = std::to_string(numDepartments) + std::string("_") + std::to_string(totalNumEmployees);
  const std::string dbname = std::string("BenchAggregationBalancedDB") + suffix;
  const std::string inputSet = std::string("BenchAggregationBalancedInputSet") + suffix;
  const std::string outputSet = std::string("BenchAggregationBalancedOutputSet") + suffix;
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
  // Here we count up the number of allocation blocks in order to estimate
  // the number of bytes per Employee.
  unsigned int numbytes = 1024 * 1024 * 64; // 64 MiB
  pdb::makeObjectAllocatorBlock(numbytes, true);
  int employeesSent = 0;
  int blocksSent = 0;
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
      ++blocksSent;
    } catch (pdb::NotEnoughSpace& e) {
      pdbClient.sendData<pdb::Employee>(dbname, inputSet, toSend);
      ++blocksSent;
      pdb::makeObjectAllocatorBlock(numbytes, true);
    }
  }
  double totalBytes = (double)(numbytes * blocksSent);
  double bytesPerEmp = totalBytes / totalNumEmployees;
  std::cout << "Each Employee takes up approximately " << bytesPerEmp << " bytes." << std::endl;
  std::cout << "NOTE: this approximation is only meaningful if a significant number of allocation blocks had to be created and sent." << std::endl;

  /// Finally, run the work loop
  // Here we're allocating the vector that will be returned
  std::unique_ptr<std::vector<std::chrono::duration<double>>>
      times = std::make_unique<std::vector<std::chrono::duration<double>>>(numReps);
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

    // The type parameters for std::chrono::duration are: the numerical type used to store the
    // duration, and the period (in seconds) of this duration's unit. The default period is 1,
    // so to get a duration in seconds we only need to specify the type.
    // https://en.cppreference.com/w/cpp/chrono/duration
    std::chrono::duration<double> seconds = end - start;
    (*times)[rep] = seconds;

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
  pdbClient.removeSet(dbname, inputSet);
  pdbClient.removeSet(dbname, outputSet);

  return times;
}



int main(int argc, char *argv[]) {
  // Code for parsing program options is modified from here:
  // http://www.radmangames.com/programming/how-to-use-boost-program_options
  int numReps;
  bool validate;
  try {
    /** Define and parse the program options
     */
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Prints usage information")
        ("numreps,n", po::value<int>(&numReps)->default_value(5), "Number of times the Aggregation is repeated for each numDepartments/totalNumEmployees combination")
        ("validate,v", po::bool_switch(&validate), "Every rep, iterate over the aggregated output and check whether it looks correct");

    po::variables_map vm;
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
      }

      po::notify(vm);
    } catch (po::error& e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return 1;
    }
  } catch(std::exception& e) {
    std::cerr << "Unhandled exception caught while parsing command line args: " << e.what() << std::endl;
    std::cerr << "This program will now exit." << std::endl;
    return 2;
  }

  // Before running the benchmark, we will test that the timing code is correct
  // and returns the time taken in the correct unit (seconds).
  std::cout << "Testing timing code. Will sleep for 3 seconds." << std::endl;
  auto start = std::chrono::system_clock::now(); // system_clock is the "real" (wall-clock) system time
  std::this_thread::sleep_for(std::chrono::seconds(3)); // Sleep for at least 3 seconds
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> durationInSeconds = end - start;
  double numSeconds = durationInSeconds.count();
  std::cout << "The duration slept is " << numSeconds << " seconds. It should've been a little over 3 seconds." << std::endl;


  std::vector<std::pair<int, int>> column_values; // Contains pairs of (numDepartments, totalNumEmployees)
#define ADDPAIR(x,y) column_values.push_back(std::make_pair<int,int>((x), (y)))
  // NOTE: modify combinations here if desired
  ADDPAIR(100, 10000);
  ADDPAIR(50, 50000);
  ADDPAIR(1000, 100000); // 100k employees
  ADDPAIR(1000, 1000000); // 1 million employees
  ADDPAIR(1000, 10000000); // 10 million employees
  ADDPAIR(2000, 10000000); // 10 million employees
  ADDPAIR(10000, 10000000); // 10 million employees
//  ADDPAIR(10000, 100000000); // 100 million employees
//  ADDPAIR(10000, 1000000000); // 1 billion employees
  // TODO: try uncommenting the two lines above

  std::cout << "Combinations that will be benchmarked:" << std::endl;
  for (auto elem : column_values) {
    std::cout << elem.first << " departments and " << elem.second << " employees" << std::endl;
  }

  auto numcols = column_values.size();
  double table[numReps][numcols];
  // Each column in table holds numReps doubles, which are the times (in seconds)
  // of all the reps for that column's numDepartments and totalNumEmployees values.

  for(int col = 0; col < numcols; ++col) {
    std::pair<int,int> elem = column_values[col];
    auto vectorOfDurations = BenchAggregationBalanced(elem.first, elem.second, numReps, validate);
    for(int row = 0; row < numReps; ++row) {
      std::chrono::duration<double> d = (*vectorOfDurations)[row];
      double secs = d.count();
      table[row][col] = secs;
    }
  }

  // Now printing the table of recorded times
  std::cout << "The table below is formatted as follows:" << std::endl;
  std::cout << "Each column is labeled with two numbers. The first number is "
            << "the number of departments, and the second number is the "
            << "total number of employees in all departments. Each department "
            << "has the same number of employees." << std::endl;
  std::cout << "The rows are the time, in seconds, that it took to run the "
            << "Aggregation. The same Aggregation is run num_rows times on the same "
            << "instance of PDB, and the times are listed in order from top to bottom." << std::endl;
  std::cout << "All whitespace should be ignored when parsing the table as a CSV." << std::endl;
  std::cout << std::endl;
  // Print header line
  std::cout << "idx";
  int colwidth = 30; // Width of each column, in characters
  for (auto elem : column_values) {
    std::cout <<  "," << std::setw(30) << std::setfill(' ') << (std::to_string(elem.first) + std::string("/") + std::to_string(elem.second));
  }
  std::cout << std::endl;
  // Print the numbers
  for (int row = 0; row < numReps; ++row) {
    std::cout << std::setw(3) << std::setfill(' ') << row+1; // This line is to insert spaces to ensure that the numbers in the first column are all
                                                             // aligned with the column header, 'idx'
    for (int col = 0; col < numcols; ++col) {
      std::cout << ",";
      // The below line is modified from https://stackoverflow.com/q/49295185
      std::cout << std::setw(30) << std::setfill(' ') << std::scientific << std::setprecision(6) << table[row][col];
    }
    std::cout << std::endl;
  }
}
