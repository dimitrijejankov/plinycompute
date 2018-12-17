/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#include <benchmark/benchmark.h>

#include "InterfaceFunctions.h"
#include "Employee.h"
#include "Supervisor.h"

using namespace pdb;

/**
 * The Employee class to bench std vector
 */
class EmployeeNative {

  std::string *name{nullptr};
  int age{};

 public:

  ~EmployeeNative() {
    delete name;
  }

  EmployeeNative() = default;

  EmployeeNative(const std::string &nameIn, int ageIn) {
    name = new std::string(nameIn);
    age = ageIn;
  }
};

/**
 * The Supervisor class to bench std vector
 */
class SupervisorNative {

 private:
  EmployeeNative *me{nullptr};
  std::vector<EmployeeNative *> myGuys;

 public:

  SupervisorNative() {
    for (auto a : myGuys) {
      delete a;
    }
  }

  ~SupervisorNative() = default;

  SupervisorNative(std::string name, int age) {
    me = new EmployeeNative(name, age);
  }

  void addEmp(EmployeeNative *addMe) {
    myGuys.push_back(addMe);
  }
};

static void BenchPDBObjectCreation(benchmark::State& state) {

    // bench
    for (auto _ : state) {

        // create one million allocators and one million objects
        try {

            // create an allocator
            makeObjectAllocatorBlock(1024 * 24, true);

            // create the supervisor
            Handle<Supervisor> super = makeObject<Supervisor>("Joe Johnson", 57);

            // create the Employees
            for (int j = 0; j < 10; j++) {
                Handle<Employee> temp = makeObject<Employee>("Steve Stevens", 57);
                super->addEmp(temp);
            }

            // do not optimize out the value
            benchmark::DoNotOptimize(super);

        } catch (NotEnoughSpace& e) {

            std::cout << "This is bad.  Why did I run out of RAM?\n";
            exit(-1);
        }
    }
}

static void BenchNativeObjectCreation(benchmark::State& state) {

  // employee name
  const std::string name = "Steve Stevens";

  // bench
  for (auto _ : state) {

    // create the supervisor
    auto *mySup = new SupervisorNative("Joe Johnson", 23);

    for (int j = 0; j < 10; j++) {
      auto *temp = new EmployeeNative(name, 57);
      mySup->addEmp(temp);
    }

    // do not optimize out the value
    benchmark::DoNotOptimize(mySup);

    // delete the object
    delete mySup;
  }
}


// Register the function as a benchmark
BENCHMARK(BenchPDBObjectCreation);
BENCHMARK(BenchNativeObjectCreation);

// create the main function
BENCHMARK_MAIN();